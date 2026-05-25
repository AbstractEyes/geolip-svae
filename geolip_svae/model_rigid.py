"""rigid_patch_svae.py — the rigidity formula embedded structurally in the
REAL PatchSVAE.

This is NOT a reinvented architecture. It extends geolip_svae.model.PatchSVAE
(the actual Spectral Autoencoder: enc → row-norm M → SVD/linear-readout →
SpectralCrossAttention → decode) by attaching the rigidity formula exactly
where it structurally belongs: on the row-normalized M-row codebook
directions — the V rows on S^(D-1) that `extract_codebook` reads and that
every experiment in the catalog measured.

Grounded in the session scratchpad + catalog experiments:

  • The codebook = `_row_normalize(enc_out output, 'sphere')` rows on S^(D-1),
    sign-canonicalized to RP^(D-1). NOT a vMF latent, NOT a separate VQ
    codebook — the M-rows the encoder already produces.

  • deviation = mean_projective_angle(M_rows) − uniform_projective_angle(D).
    The polytope-class statute is deviation POSITIVE (more spread than
    uniform), per 000115.

  • TWO valid statute regimes (000115 reconciliation):
      Regime 1  |dev| < dev_critical(D)=0.02√D  → dense H1 topology,
                the rigidity envelope, "spine" expression
      Regime 2  |dev| > dev_critical(D)         → degenerate topology
                (n_H1→0) but STILL a valid omega-class reconstruction.
                byte_trigram_proto_64_patch_2_v1 (dev +0.083 at D=4,
                MSE 3.57e-7) is the canonical Regime-2 example.
    "Collapse" in the rigidity sense ≠ task failure. The formula classifies
    the regime; it does not declare the model broken.

  • CV pin: the M-row pairwise-angle CV is pulled toward the 0.20–0.23
    pentachoron band (the "star" / CV attractor), validated across 17+
    architectures.

  • p50 offset: within Regime 1, pairwise p50 sits ~2.5° above
    uniform_baseline(D) — the training-driven signature (|z|=3.5 at D=4).
    Read-only diagnostic; NOT enforced (let it emerge — the C2 test).

  • H2-class = linear_readout=True, svd_mode='none' (omega-class statute via
    the learned sphere-solver readout).

The formula is added as auxiliary losses on top of the existing
reconstruction; the spectral core (SVD/readout, cross-attention, decode) is
the real PatchSVAE untouched. This is the "build the full formula
structurally where it needs to be" — on the codebook the architecture
already produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The real architecture. Import guarded so the RigidityFormula module
# (which needs only torch) is testable even where geolip_svae's solver
# dependencies (geolip-core / FLEigh) aren't installed.
try:
    from geolip_svae.model import PatchSVAE
    HAVE_PATCHSVAE = True
except Exception:
    PatchSVAE = object  # type: ignore
    HAVE_PATCHSVAE = False


# ════════════════════════════════════════════════════════════════════════
#  Geometry primitives (mirror geolip_svae.inference.codebook exactly)
# ════════════════════════════════════════════════════════════════════════

def canonicalize_sign(v: torch.Tensor) -> torch.Tensor:
    """First nonzero coordinate positive, along the last dim. Vectorized
    form of inference.codebook._canonicalize_sign."""
    eps = 1e-6
    nonzero = v.abs() > eps
    first_idx = nonzero.float().argmax(dim=-1, keepdim=True)
    first_val = torch.gather(v, -1, first_idx)
    sign = torch.where(first_val < 0, -1.0, 1.0)
    return v * sign


_UNIFORM_CACHE: Dict[int, float] = {}
_UNIFORM_P50_CACHE: Dict[int, float] = {}


def uniform_projective_angle(D: int, n_samples: int = 4096,
                              seed: int = 0) -> float:
    """MEAN pairwise projective angle for uniform points on RP^(D-1).
    Mirrors inference.codebook.uniform_projective_angle. Cached per D.
    This is the baseline `deviation` is measured against."""
    if D in _UNIFORM_CACHE:
        return _UNIFORM_CACHE[D]
    g = torch.Generator().manual_seed(int(seed))
    pts = torch.randn(n_samples, D, generator=g)
    pts = pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pts = canonicalize_sign(pts)
    cos = (pts @ pts.T).clamp(-1, 1)
    angles = torch.acos(cos.abs())
    iu = torch.triu_indices(n_samples, n_samples, offset=1)
    vals = angles[iu[0], iu[1]]
    _UNIFORM_CACHE[D] = float(vals.mean())
    _UNIFORM_P50_CACHE[D] = float(vals.median())
    return _UNIFORM_CACHE[D]


def uniform_projective_p50(D: int, n_samples: int = 4096,
                            seed: int = 0) -> float:
    """MEDIAN pairwise projective angle for uniform points on RP^(D-1).

    Distinct from uniform_projective_angle (the mean): the folded
    projective distribution is right-skewed, so median > mean structurally.
    The catalog's training-driven p50 gap was trained_p50 − null_p50 (both
    medians), so p50_offset must be measured against THIS, not the mean
    baseline. Comparing the codebook median against the uniform MEAN would
    fold the structural median-mean gap into what looks like a training
    signal."""
    if D not in _UNIFORM_P50_CACHE:
        uniform_projective_angle(D, n_samples, seed)  # populates both caches
    return _UNIFORM_P50_CACHE[D]


def dev_critical(D: int, coeff: float = 0.02) -> float:
    """Rigidity envelope boundary: dev_critical(D) = coeff × √D.
    Empirical fit D=4→0.040, D=16→0.080 (5–7% agreement)."""
    return coeff * math.sqrt(D)


def _pairwise_proj_angles(axes: torch.Tensor) -> torch.Tensor:
    """Upper-triangular pairwise projective angles (radians) of unit axes.
    axes: (N, D). Returns 1-D tensor of N·(N-1)/2 angles in [0, π/2]."""
    a = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (a @ a.T).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos.abs())                      # fold to [0, π/2]
    n = axes.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=axes.device)
    return ang[iu[0], iu[1]]


# ════════════════════════════════════════════════════════════════════════
#  RigidityFormula — the structural formula on the M-row codebook
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RigidityConfig:
    dev_critical_coeff: float = 0.02        # dev_critical(D) = coeff × √D
    envelope_margin: float = 0.7            # hinge starts at margin × dev_critical
    cv_attractor: float = 0.215             # the star: pentachoron band center
    cv_band: Tuple[float, float] = (0.20, 0.23)
    uniform_samples: int = 4096
    # Loss weights for the structural formula terms
    w_envelope: float = 0.1                 # pull into Regime 1 (rigidity envelope)
    w_cv: float = 0.05                      # pin CV to the band (the star)
    # Antipodal collapse: dedup near-duplicate ± rows before measuring,
    # matching extract_codebook's antipodal handling.
    antipodal_collapse: bool = True
    antipodal_thresh_deg: float = 1.0       # rows within this |angle| are merged


class RigidityFormula(nn.Module):
    """Computes the rigidity formula on a codebook of M-row directions.

    The codebook is the row-normalized M (V rows on S^(D-1)). This module
    is stateless except for the cached uniform baseline; it produces the
    structural losses (envelope, cv) and the diagnostic readout (deviation,
    p50 offset, statute class) that the assessment harness consumes.
    """

    def __init__(self, D: int, cfg: RigidityConfig):
        super().__init__()
        self.D = D
        self.cfg = cfg
        ub = uniform_projective_angle(D, cfg.uniform_samples)
        up50 = uniform_projective_p50(D, cfg.uniform_samples)
        self.register_buffer('uniform_baseline', torch.tensor(ub))
        self.register_buffer('uniform_p50', torch.tensor(up50))
        self.register_buffer('dev_crit',
                             torch.tensor(dev_critical(D, cfg.dev_critical_coeff)))

    def _collapse_antipodal(self, axes: torch.Tensor) -> torch.Tensor:
        """Optionally merge near-duplicate sign-canonicalized rows so the
        measured codebook matches extract_codebook's antipodal-collapsed set.
        Greedy, differentiable-safe (selects a subset, no in-place)."""
        if not self.cfg.antipodal_collapse or axes.shape[0] < 2:
            return axes
        a = canonicalize_sign(
            axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12))
        thresh = math.cos(math.radians(self.cfg.antipodal_thresh_deg))
        keep = []
        used = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
        sim = (a @ a.T).abs()
        for i in range(a.shape[0]):
            if used[i]:
                continue
            keep.append(i)
            dup = sim[i] > thresh
            used = used | dup
        return axes[torch.tensor(keep, device=axes.device)]

    def measure(self, M_rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        """M_rows: (V, D) codebook directions (already row-normalized
        upstream, but we renormalize defensively). Returns differentiable
        formula quantities."""
        axes = self._collapse_antipodal(M_rows)
        angles = _pairwise_proj_angles(axes)
        mean_ang = angles.mean()
        deviation = mean_ang - self.uniform_baseline
        cv = angles.std(unbiased=False) / mean_ang.clamp_min(1e-8)
        return {
            'n_axes': torch.tensor(float(axes.shape[0])),
            'mean_proj_angle': mean_ang,
            'deviation': deviation,
            'cv': cv,
            'p50': angles.median(),
            'uniform_baseline': self.uniform_baseline,
            'dev_critical': self.dev_crit,
        }

    def envelope_penalty(self, deviation: torch.Tensor) -> torch.Tensor:
        """Hinge pulling toward Regime 1 (the rigidity envelope). Zero
        inside margin × dev_critical, quadratic ramp toward the boundary.
        Soft — does not forbid Regime 2, just biases toward the dense-
        topology regime."""
        thresh = self.cfg.envelope_margin * self.dev_crit
        over = (deviation.abs() - thresh).clamp_min(0.0)
        return over.pow(2)

    def cv_pin_penalty(self, cv: torch.Tensor) -> torch.Tensor:
        """Pull the CV toward the pentachoron band center (the star)."""
        return (cv - self.cfg.cv_attractor).pow(2)

    def forward(self, M_rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        m = self.measure(M_rows)
        l_env = self.envelope_penalty(m['deviation'])
        l_cv = self.cv_pin_penalty(m['cv'])
        loss = self.cfg.w_envelope * l_env + self.cfg.w_cv * l_cv
        return {**m, 'envelope_penalty': l_env, 'cv_penalty': l_cv,
                'formula_loss': loss}

    def classify_statute(self, m: Dict[str, torch.Tensor]) -> str:
        """Statute classification per the 000115 taxonomy. Read-only."""
        dev = float(m['deviation'])
        absdev = abs(dev)
        crit = float(self.dev_crit)
        lo, hi = self.cfg.cv_band
        cv = float(m['cv'])
        in_band = lo <= cv <= hi
        if absdev < 0.05 and absdev < crit:
            base = 'uniform_class'         # near uniform RP^(D-1)
        elif dev > 0 and absdev < crit:
            base = 'polytope_class_R1'     # Regime 1: dense-topology envelope
        elif dev > 0:
            base = 'polytope_class_R2'     # Regime 2: valid but degenerate topo
        else:
            base = 'sub_uniform'           # clumped (rare / degenerate)
        return f"{base}{'+cv_band' if in_band else ''}"

    def readout(self, M_rows: torch.Tensor) -> Dict:
        with torch.no_grad():
            m = self.measure(M_rows)
            return {
                'D': self.D,
                'n_axes': int(m['n_axes']),
                'deviation': float(m['deviation']),
                'dev_critical': float(self.dev_crit),
                'in_envelope': abs(float(m['deviation'])) < float(self.dev_crit),
                'mean_proj_angle_deg': math.degrees(float(m['mean_proj_angle'])),
                'uniform_baseline_deg': math.degrees(float(self.uniform_baseline)),
                'p50_deg': math.degrees(float(m['p50'])),
                'uniform_p50_deg': math.degrees(float(self.uniform_p50)),
                'p50_offset_deg': math.degrees(
                    float(m['p50'] - self.uniform_p50)),
                'cv': float(m['cv']),
                'cv_target': self.cfg.cv_attractor,
                'statute': self.classify_statute(m),
            }


# ════════════════════════════════════════════════════════════════════════
#  RigidPatchSVAE — the real PatchSVAE with the formula wired structurally
# ════════════════════════════════════════════════════════════════════════

class RigidPatchSVAE(PatchSVAE):
    """The real PatchSVAE, extended so the rigidity formula is computed on
    the M-row codebook every forward pass and exposed as structural losses.

    The spectral core (enc → row-norm M → SVD/readout → SpectralCrossAttention
    → decode → stitch → smooth) is the parent's, untouched. We only:
      1. read svd['M'] from the parent forward (the row-normalized codebook),
      2. measure the formula on it,
      3. surface envelope + cv losses for the trainer to add to reconstruction.
    """

    def __init__(self, *args, rigidity: Optional[RigidityConfig] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rigidity_cfg = rigidity or RigidityConfig()
        self.formula = RigidityFormula(self.D, self.rigidity_cfg)

    def _codebook_rows(self, svd: Dict) -> torch.Tensor:
        """Extract the codebook directions from the forward's svd dict.

        svd['M'] is (B, N, V, D) row-normalized. The codebook is the set of
        V row-directions; we pool across the batch×patch axis by averaging
        each row's direction over (B, N) then renormalizing. This matches
        the spirit of extract_codebook, which reads the encoder's M-row
        directions as the codebook.
        """
        M = svd['M']                                  # (B, N, V, D) or (B*N, V, D)
        if M.dim() == 4:
            B, N, V, D = M.shape
            M = M.reshape(B * N, V, D)
        # Mean direction per codebook row across the batch×patch axis
        rows = M.mean(dim=0)                          # (V, D)
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return canonicalize_sign(rows)

    def forward(self, images: torch.Tensor) -> dict:
        out = super().forward(images)                 # {'recon', 'svd'}
        rows = self._codebook_rows(out['svd'])
        out['codebook_rows'] = rows
        out['rigidity'] = self.formula(rows)          # measures + losses
        return out

    def loss(self, images: torch.Tensor,
             w_recon: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(images)
        recon = out['recon']
        l_recon = F.mse_loss(recon, images)
        rg = out['rigidity']
        total = w_recon * l_recon + rg['formula_loss']
        logs = {
            'loss': float(total.detach()),
            'recon': float(l_recon.detach()),
            'envelope': float(rg['envelope_penalty'].detach()),
            'cv_pin': float(rg['cv_penalty'].detach()),
            'deviation': float(rg['deviation'].detach()),
            'cv': float(rg['cv'].detach()),
            'p50_offset_deg': math.degrees(
                float((rg['p50'] - self.formula.uniform_p50).detach())),
        }
        return total, logs

    def assess(self, images: Optional[torch.Tensor] = None) -> Dict:
        """Curated assessment — formula adherence on the codebook. If images
        given, measures on the live codebook from that batch; else uses the
        encoder's current M-row directions on a zero probe is not meaningful,
        so images should be provided during training."""
        if images is None:
            raise ValueError("assess() needs an image batch to read the "
                             "live M-row codebook")
        with torch.no_grad():
            out = self.forward(images)
            rows = out['codebook_rows']
            return self.formula.readout(rows)


# ════════════════════════════════════════════════════════════════════════
#  Config presets (H2-class = linear_readout + svd_mode='none')
# ════════════════════════════════════════════════════════════════════════

def h2_class_kwargs(D: int = 4, V: int = 32, ps: int = 4, hidden: int = 64,
                    depth: int = 1, n_cross: int = 1,
                    channels: int = 3) -> Dict:
    """The H2-class sphere-solver config: linear_readout=True, svd_mode='none'.
    Defaults mirror byte_trigram_proto_64_patch_2_v1's regime (D=4, V=32)."""
    return dict(
        V=V, D=D, ps=ps, hidden=hidden, depth=depth, n_cross=n_cross,
        channels=channels,
        linear_readout=True, svd_mode='none', match_params=True,
        row_norm='sphere',
    )


def build_rigid_h2(D: int = 4, rigidity: Optional[RigidityConfig] = None,
                   **overrides) -> 'RigidPatchSVAE':
    """Build an H2-class RigidPatchSVAE at the given D."""
    if not HAVE_PATCHSVAE:
        raise RuntimeError(
            "geolip_svae.model.PatchSVAE not importable in this environment. "
            "Run where the geolip-svae package + solver deps are installed.")
    kw = h2_class_kwargs(D=D)
    kw.update(overrides)
    return RigidPatchSVAE(rigidity=rigidity, **kw)


# ════════════════════════════════════════════════════════════════════════
#  Smoke test
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"PatchSVAE importable: {HAVE_PATCHSVAE}")

    # The RigidityFormula module is testable standalone (torch only).
    print("\n── RigidityFormula standalone test ──")
    for D in (4, 8, 16):
        cfg = RigidityConfig()
        formula = RigidityFormula(D, cfg)
        # Synthetic codebook: random directions on S^(D-1)
        torch.manual_seed(0)
        V = {4: 32, 8: 48, 16: 256}[D]
        M = torch.randn(V, D)
        out = formula(M)
        ro = formula.readout(M)
        print(f"  D={D:2d} V={V:3d}: "
              f"dev={ro['deviation']:+.4f} "
              f"(crit={ro['dev_critical']:.4f}, "
              f"in_env={ro['in_envelope']}) "
              f"cv={ro['cv']:.4f} "
              f"p50_off={ro['p50_offset_deg']:+.2f}° "
              f"statute={ro['statute']}")
        print(f"           formula_loss={float(out['formula_loss']):.5f} "
              f"(env={float(out['envelope_penalty']):.5f}, "
              f"cv={float(out['cv_penalty']):.5f})")

    # Verify gradient flows through the formula to the codebook
    print("\n── gradient test ──")
    M = torch.randn(32, 4, requires_grad=True)
    formula = RigidityFormula(4, RigidityConfig())
    loss = formula(M)['formula_loss']
    loss.backward()
    print(f"  formula_loss.backward() OK, "
          f"grad norm on codebook = {M.grad.norm():.4f}")

    # Full model only if the real PatchSVAE is importable
    if HAVE_PATCHSVAE:
        print("\n── RigidPatchSVAE full model test ──")
        model = build_rigid_h2(D=4)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  RigidPatchSVAE (H2-class D=4): {n_params:,} params")
        img = torch.randn(2, 3, 64, 64)
        total, logs = model.loss(img)
        print(f"  loss OK: {logs}")
        total.backward()
        print(f"  backward OK — gradients flow through spectral core + formula")
        print(f"  assessment: {model.assess(img)}")
    else:
        print("\n  (Skipping full-model test — run where geolip_svae installs.)")
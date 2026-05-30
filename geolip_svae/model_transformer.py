"""
geolip_svae.model_transformer
=============================
The geolip-svae-transformer MODEL — architecture only, no trainer.

A two-recon, frozen-boundary transformer built on the H2 PatchSVAE microcosm:

    images
      -> Battery (FULL PatchSVAE autoencoder)  -> internal_recon, M
      -> M.detach()  (the STEM boundary; downstream is decoupled)
      -> SingleLens  (fixed isometric lift  D_base -> D_lens)
      -> SpectralAlphaStack (coordination)  -> GeoDecoder  -> external_recon
      -> AlephAssessor (read-only signed-projective address)

Trainable surface and the three-optimizer separation are the TRAINER's concern
and live outside this module (e.g. a train_transformer.py). This file exposes
only the model and its components for use and implementation:

    GeoConfig, GeoSVAETransformer, build_model,
    Battery, SingleLens, AlephAssessor,
    SpectralAlphaAttention, SpectralAlphaStack, GeoDecoder,
    canon, extract_patches, stitch_patches, byte_recovery,
    uniform_projective_angle, dev_critical, intrinsic_deviation, rigidity_barrier
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Real H2 battery from the same package. Battery also imports PatchSVAE locally;
# this flag lets the lean fallback engage when the repo isn't importable (e.g.
# shell-only experimentation outside the package).
try:
    from geolip_svae.model import PatchSVAE   # noqa: F401
    _HAVE_REPO = True
except Exception:
    _HAVE_REPO = False

# ════════════════════════════════════════════════════════════════════════
#  Projective helpers (vendored / aligned with the verified code)
# ════════════════════════════════════════════════════════════════════════

def canon(v: torch.Tensor) -> torch.Tensor:
    """Sign-canonicalize onto the projective representative: first nonzero
    coordinate positive. Antipodes -> the same point (a line through 0)."""
    nz = v.abs() > 1e-6
    fi = nz.float().argmax(dim=-1, keepdim=True)
    fv = torch.gather(v, -1, fi)
    return v * torch.where(fv < 0, -1.0, 1.0)


_UMEAN: Dict[int, float] = {}


def uniform_projective_angle(D: int, n: int = 4096, seed: int = 0) -> float:
    """Mean pairwise acos|cos| of uniform sign-canonicalized directions on
    S^(D-1) — the rigid (uniform-packing) baseline."""
    if D in _UMEAN:
        return _UMEAN[D]
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, D, generator=g)
    pts = canon(pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12))
    cos = (pts @ pts.T).clamp(-1, 1)
    iu = torch.triu_indices(n, n, offset=1)
    _UMEAN[D] = float(torch.acos(cos.abs())[iu[0], iu[1]].mean())
    return _UMEAN[D]


def dev_critical(D: int, coeff: float = 0.02) -> float:
    """Envelope half-width: how far the projective mean-angle may sit from the
    uniform baseline before the frame is 'out of envelope'."""
    return coeff * math.sqrt(D)


@torch.no_grad()
def intrinsic_deviation(M_rows: torch.Tensor, baseline_D: int) -> float:
    """Projective mean-angle deviation of a (V, D) frame from uniform. Monitor."""
    a = M_rows / M_rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (a @ a.T).abs().clamp(0, 1 - 1e-7)
    n = M_rows.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=M_rows.device)
    return float(torch.acos(cos[iu[0], iu[1]]).mean()) - uniform_projective_angle(baseline_D)


def rigidity_barrier(M: torch.Tensor, D_base: int, margin_frac: float = 0.5) -> torch.Tensor:
    """Hinge edge-wall on per-patch projective deviation: ZERO inside the
    envelope (never fights recon), quadratic once deviation exceeds
    margin_frac * crit. Architectural constraint, not the optimizer crutch.
    M: (B, N, V, D)."""
    B, N, V, D = M.shape
    a = F.normalize(M, dim=-1).reshape(B * N, V, D)
    cos = (a @ a.transpose(1, 2)).abs().clamp(0, 1 - 1e-7)
    iu = torch.triu_indices(V, V, offset=1, device=M.device)
    mean_ang = torch.acos(cos[:, iu[0], iu[1]]).mean(dim=1)          # (BN,)
    dev = (mean_ang - uniform_projective_angle(D_base)).abs()
    return F.relu(dev - margin_frac * dev_critical(D_base)).pow(2).mean()


def extract_patches(images: torch.Tensor, ps: int):
    B, C, H, W = images.shape
    gh, gw = H // ps, W // ps
    p = images.reshape(B, C, gh, ps, gw, ps).permute(0, 2, 4, 1, 3, 5).contiguous()
    return p.reshape(B, gh * gw, C * ps * ps), gh, gw


def stitch_patches(patches: torch.Tensor, gh: int, gw: int, ps: int, C: int):
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, C, ps, ps)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, C, gh * ps, gw * ps)


# ════════════════════════════════════════════════════════════════════════
#  Geometric front-end — patch -> sphere-normalized M (V, D_base)
# ════════════════════════════════════════════════════════════════════════

class LeanGeometricEncoder(nn.Module):
    """Sandbox encoder mirroring PatchSVAE: enc_in -> residual blocks ->
    enc_out -> reshape(V, D) -> row sphere-normalize -> canon."""

    def __init__(self, patch_dim: int, V: int, D: int, hidden: int, depth: int = 1):
        super().__init__()
        self.V, self.D = V, D
        self.enc_in = nn.Linear(patch_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(depth)])
        self.enc_out = nn.Linear(hidden, V * D)
        nn.init.orthogonal_(self.enc_out.weight)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, _ = patches.shape
        h = F.gelu(self.enc_in(patches.reshape(B * N, -1)))
        for blk in self.blocks:
            h = h + blk(h)
        M = self.enc_out(h).reshape(B * N, self.V, self.D)
        return canon(F.normalize(M, dim=-1)).reshape(B, N, self.V, self.D)


class Battery(nn.Module):
    """The H2 PatchSVAE used as a FULL autoencoder — the small pot.

    encode -> M -> decode -> internal recon. Trained on its OWN internal recon
    (pure MSE), exactly the core system the H2 dictates, so it converges and
    stays rigid on its own. Exposes the stem M that the shell reads (detached).

    Real path: the installed geolip_svae.PatchSVAE — its native forward already
    returns image-space recon + the svd dict containing M. Lean path: a
    self-contained encode/decode so this file runs standalone."""

    def __init__(self, cfg: "GeoConfig"):
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.ps * cfg.ps
        self.real = cfg.use_real_svae and _HAVE_REPO
        if self.real:
            try:
                from geolip_svae.model import PatchSVAE
                self.svae = PatchSVAE(V=cfg.V, D=cfg.D_base, ps=cfg.ps, hidden=cfg.hidden,
                                      channels=cfg.channels, depth=1, n_cross=1,
                                      linear_readout=True, svd_mode='none',
                                      match_params=True, row_norm='sphere')
            except Exception as e:
                print(f"  [battery] real PatchSVAE unavailable ({e}); using lean battery")
                self.real = False
        if not self.real:
            self.enc = LeanGeometricEncoder(patch_dim, cfg.V, cfg.D_base, cfg.hidden)
            self.dec_in = nn.Linear(cfg.V * cfg.D_base, cfg.hidden)
            self.dec_blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(),
                              nn.Linear(cfg.hidden, cfg.hidden))])
            self.dec_out = nn.Linear(cfg.hidden, patch_dim)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.real:
            o = self.svae(images)                          # {'recon': (B,C,H,W), 'svd': {'M', ...}}
            return {'internal_recon': o['recon'], 'M': o['svd']['M']}
        patches, gh, gw = extract_patches(images, self.cfg.ps)
        M = self.enc(patches)                              # (B,N,V,D_base) sphere
        B, N, V, D = M.shape
        h = F.gelu(self.dec_in(M.reshape(B * N, V * D)))
        for blk in self.dec_blocks:
            h = h + blk(h)
        dec = self.dec_out(h).reshape(B, N, -1)            # (B,N,patch_dim)
        recon = stitch_patches(dec, gh, gw, self.cfg.ps, self.cfg.channels)
        return {'internal_recon': recon, 'M': M}


# ════════════════════════════════════════════════════════════════════════
#  The single lens — the GUARANTEE. Fixed orthonormal isometric lift.
# ════════════════════════════════════════════════════════════════════════

class SingleLens(nn.Module):
    """Fixed orthonormal isometric lift  D_base -> D_lens:  <Ex, Ey> = <x, y>.

    Rows enter on S^(D_base-1) and exit on a great subsphere of S^(D_lens-1) —
    the geometry is carried up EXACTLY, the envelope cannot be left, and nothing
    here learns. This is the wall. (D_lens == D_base is allowed: a pure fixed
    rotation, still isometric.) The whole architecture's stability rests here."""

    def __init__(self, D_base: int, D_lens: int, seed: int = 0, sign_mode: str = 'canon'):
        super().__init__()
        assert D_lens >= D_base, "isometric lift requires D_lens >= D_base"
        assert sign_mode in ('canon', 'signed')
        g = torch.Generator().manual_seed(seed)
        E = torch.linalg.qr(torch.randn(D_lens, D_base, generator=g))[0]   # (D_lens, D_base)
        self.register_buffer('E', E)
        self.D_base, self.D_lens, self.sign_mode = D_base, D_lens, sign_mode

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        # M: (B, N, V, D_base) sphere -> (B, N, V, D_lens) sphere
        lifted = F.normalize(M @ self.E.T, dim=-1)
        # 'canon'  : project to RP^(D-1) — per-row sign DROPPED (the control, ~90%).
        # 'signed' : keep S^(D-1) — per-row sign PRESERVED to the decoder. The
        #   omega channel is sign-invariant either way, so this changes ONLY what
        #   the decoder sees: a clean isolation of the sign channel that rigidity
        #   (computed on |cos|) leaves free to carry per-input content.
        return canon(lifted) if self.sign_mode == 'canon' else lifted


# ════════════════════════════════════════════════════════════════════════
#  Aleph assessor — READ-ONLY signed-projective address (the "aleph-logit").
#  D1 KNOB: the functional form of the address readout.
# ════════════════════════════════════════════════════════════════════════

class AlephAssessor(nn.Module):
    """Read-only discriminative readout over the SIGNED PROJECTIVE assignment
    of M against a FIXED codebook of axes. NOT a trainable logit head: M is
    detached, so this never backprops into the lens and never drags values
    toward zero. It is the metric for a fractally-bound space where cosine
    saturates — we assess by ADDRESS (which axis + which sign), not by raw angle.

    Returns per-row: axis (argmax projective margin), sign (the ± bit), margin
    (|<M, ref>|), and a soft assignment. Two inputs that cosine collapses can
    still differ in their signed address pattern across the V rows.

    D1: swap `codebook` for an EXTRACTED/farmed set (explicit discovery), or
    change the readout (margins vs sign-agreement vs Cayley-Menger volume vs
    finite/infinite assessment). Default = deterministic axes (finite capacity).
    """

    def __init__(self, D: int, K: int, codebook: Optional[torch.Tensor] = None,
                 temp: float = 0.1, seed: int = 1):
        super().__init__()
        if codebook is None:
            g = torch.Generator().manual_seed(seed)
            codebook = canon(F.normalize(torch.randn(K, D, generator=g), dim=-1))
        self.register_buffer('codebook', codebook)             # (K, D) projective axes
        self.temp = temp

    @torch.no_grad()
    def forward(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        a = canon(F.normalize(M.detach(), dim=-1))             # (..., D), read-only
        proj = a @ self.codebook.T                             # (..., K) signed projection
        margin = proj.abs().clamp(0, 1 - 1e-6)                 # antipode-invariant margin
        axis = margin.argmax(dim=-1)                           # (...) the address
        sign = torch.sign(proj.gather(-1, axis.unsqueeze(-1)).squeeze(-1))  # the sign bit
        assign = F.softmax(margin / self.temp, dim=-1)         # soft assignment
        return {'axis': axis, 'sign': sign, 'margin': margin, 'assign': assign}

    @torch.no_grad()
    def code(self, M: torch.Tensor) -> torch.Tensor:
        """Signed integer aleph address per row: axis * sign. (..., V)."""
        o = self.forward(M)
        return (o['axis'] + 1) * o['sign'].long()              # ±(axis+1), 0 only if sign 0


# ════════════════════════════════════════════════════════════════════════
#  Spectral-alpha cross-patch attention (vendored — the only attention the
#  omegas are aligned to). Bounded, multiplicative, near-identity at init.
# ════════════════════════════════════════════════════════════════════════

class SpectralAlphaAttention(nn.Module):
    """S_out = S * (1 + alpha_d * tanh(out_proj(SDPA(qkv(norm(S))))_d)).
    Per-mode alpha in [0, max_alpha], init sigmoid(-2)*0.2 ~ 0.024 -> starts as
    near-identity and ENGAGES gradually as the omegas form. Multiplicative, so
    it can only modulate the spectrum, never inject into it."""

    def __init__(self, D, n_heads=4, max_alpha=0.2, alpha_init=-2.0):
        super().__init__()
        assert D % n_heads == 0, f"D={D} must be divisible by n_heads={n_heads}"
        self.n_heads, self.head_dim, self.max_alpha = n_heads, D // n_heads, max_alpha
        self.qkv = nn.Linear(D, 3 * D)
        self.out_proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.alpha_logits = nn.Parameter(torch.full((D,), float(alpha_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.max_alpha * torch.sigmoid(self.alpha_logits)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, N, D = S.shape
        qkv = self.qkv(self.norm(S)).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        out = out.transpose(1, 2).reshape(B, N, D)
        return S * (1.0 + self.alpha.view(1, 1, -1) * torch.tanh(self.out_proj(out)))


class SpectralAlphaStack(nn.Module):
    def __init__(self, D, n_heads, n_layers, max_alpha=0.2, alpha_init=-2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SpectralAlphaAttention(D, n_heads, max_alpha, alpha_init) for _ in range(n_layers)])

    def forward(self, S):
        for layer in self.layers:
            S = layer(S)
        return S

    def mean_alpha(self) -> float:
        with torch.no_grad():
            return float(torch.stack([l.alpha.mean() for l in self.layers]).mean())


# ════════════════════════════════════════════════════════════════════════
#  Decoder — modulated frame -> patch (the reconstruction path; pure MSE)
# ════════════════════════════════════════════════════════════════════════

class GeoDecoder(nn.Module):
    """M_dec = M_lens * S_att (broadcast over V rows) mirrors U*S; a small net
    reads the modulated frame to patch_dim."""

    def __init__(self, V, D_lens, patch_dim, hidden, depth=1):
        super().__init__()
        self.dec_in = nn.Linear(V * D_lens, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(depth)])
        self.dec_out = nn.Linear(hidden, patch_dim)

    def forward(self, M_lens: torch.Tensor, S_att: torch.Tensor):
        B, N, V, D = M_lens.shape
        M_dec = M_lens * S_att.unsqueeze(2)                    # (B,N,V,D_lens)
        h = F.gelu(self.dec_in(M_dec.reshape(B * N, V * D)))
        for blk in self.blocks:
            h = h + blk(h)
        return self.dec_out(h).reshape(B, N, -1)


# ════════════════════════════════════════════════════════════════════════
#  Config + model
# ════════════════════════════════════════════════════════════════════════

@dataclass
class GeoConfig:
    img_size: int = 16
    channels: int = 3
    # ── battery: the MICROCOSM (proven H2 config — do NOT scale) ──
    ps: int = 2
    V: int = 32
    D_base: int = 4
    hidden: int = 64                 # battery (PatchSVAE) hidden — stays small
    # ── shell: the MACRO-SHAPE (snap the microcosm into a large projection) ──
    D_lens: int = 256                # isometric lift D_base -> D_lens (massive macro shell)
    shell_hidden: int = 512          # shell decoder hidden
    n_heads: int = 8                 # shell spectral heads (must divide D_lens)
    n_layers: int = 6                # shell spectral depth
    max_alpha: float = 0.2
    alpha_init: float = -2.0
    use_real_svae: bool = True
    use_spectral: bool = True
    # lens sign channel — DEFAULT 'signed': keep S^(D-1), per-row sign preserved to
    # the decoder (the +8.78% address channel). 'canon' = project to RP^(D-1), sign
    # dropped (now the ABLATION, not the baseline).
    lens_sign: str = 'signed'
    # ── read-only assessment substructure ──
    codebook_K: int = 64             # fixed projective axes (cross-contrast + aleph monitors)
    cc_temp: float = 0.10            # commitment temperature              [D2 knob]
    seed: int = 0


class GeoSVAETransformer(nn.Module):
    """Plant pot, two recons.

      BATTERY (small pot): the PatchSVAE autoencoder. Trained on its INTERNAL
        recon (pure MSE) — encode -> M -> decode. Converges and stays rigid on
        its own. Exposes the stem M.
      SHELL (upward projection / large pot): reads the DETACHED stem, lifts it
        isometrically, coordinates across patches, produces the EXTERNAL recon.
        Trained on the external recon under its OWN optimizer. The detach is why
        the external recon's gradient can never reach the battery's frame.
      GROWTH: reads the detached stem, own loss, own optimizer. Parked until a
        stencil is registered.
      Aleph / cross-contrast / rigidity: READ-ONLY monitors.

    TWO recons are monitored. The INTERNAL recon is the DIRECT probe for battery
    wobble — a foreign pressure shows there immediately, in the battery's own
    decoder, where the EXTERNAL recon can stay deceptively healthy because the
    shell decoder compensates (which is why last run's cross-contrast wobble was
    invisible). The guarantee — small-pot coverage within large-pot projection —
    is exactly: watch both and see the small structure stay intact upward."""

    def __init__(self, cfg: "GeoConfig", device: str = 'cpu'):
        super().__init__()
        self.cfg = cfg
        self.D_lens = cfg.D_lens
        patch_dim = cfg.channels * cfg.ps * cfg.ps
        self.patch_dim = patch_dim

        self.battery = Battery(cfg)                              # small pot — internal recon
        self.lens = SingleLens(cfg.D_base, cfg.D_lens, seed=cfg.seed, sign_mode=cfg.lens_sign)  # upward projection (fixed, no params)
        self.transformer = SpectralAlphaStack(cfg.D_lens, cfg.n_heads, cfg.n_layers,
                                              cfg.max_alpha, cfg.alpha_init)
        self.decoder = GeoDecoder(cfg.V, cfg.D_lens, patch_dim, cfg.shell_hidden)  # external recon (macro shell)

        # read-only assessment codebook (buffer — never trains the battery)
        g = torch.Generator().manual_seed(cfg.seed + 7)
        cb = canon(F.normalize(torch.randn(cfg.codebook_K, cfg.D_base, generator=g), dim=-1))
        self.register_buffer('codebook', cb)
        self.aleph = AlephAssessor(cfg.D_base, cfg.codebook_K, codebook=cb.clone(), temp=cfg.cc_temp)

        self.growth = nn.ModuleDict()                           # parked until a stencil exists

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> Dict:
        bat = self.battery(images)                              # internal recon + the stem
        M = bat['M']                                            # (B,N,V,D_base) — battery's, grad ON
        M_stem = M.detach()                                     # ── stem boundary ──
        M_lens = self.lens(M_stem)                              # (B,N,V,D_lens) isometric lift
        omega = M_lens.norm(dim=-2)                             # (B,N,D_lens)
        S_att = self.transformer(omega) if self.cfg.use_spectral else omega
        dec = self.decoder(M_lens, S_att)                       # (B,N,patch_dim)
        gh = gw = self.cfg.img_size // self.cfg.ps
        external_recon = stitch_patches(dec, gh, gw, self.cfg.ps, self.cfg.channels)
        return {'internal_recon': bat['internal_recon'], 'external_recon': external_recon,
                'M': M, 'M_stem': M_stem, 'M_lens': M_lens, 'omega': omega, 'S_att': S_att,
                'aleph': self.aleph(M_stem),                    # READ-ONLY
                'mean_alpha': self.transformer.mean_alpha() if self.cfg.use_spectral else 0.0}

    # ── losses (two recons + parked growth) ────────────────────────────────
    def internal_recon_loss(self, out: Dict, images: torch.Tensor) -> torch.Tensor:
        """Battery's own recon — the small pot. Pure MSE; trains the battery only.
        The direct battery-wobble probe."""
        return F.mse_loss(out['internal_recon'], images)

    def external_recon_loss(self, out: Dict, images: torch.Tensor) -> torch.Tensor:
        """Shell's recon from the DETACHED stem — the large-pot projection. Pure
        MSE; trains the shell only (M detached, so the battery never sees it)."""
        return F.mse_loss(out['external_recon'], images)

    def growth_loss(self, out: Dict) -> Optional[torch.Tensor]:
        """Cultivation loss on the DETACHED stem (out['M_stem']); trains growth
        params only. None while no stencil is registered."""
        if len(self.growth) == 0:
            return None
        raise NotImplementedError(
            "growth registered but growth_loss not wired — define the stencil's "
            "loss on out['M_stem'] here.")

    # ── read-only monitors (never in any backward) ─────────────────────────
    @torch.no_grad()
    def cross_contrast_monitor(self, out: Dict) -> float:
        M = out['M_stem']
        a = canon(F.normalize(M, dim=-1))
        margin = (a @ self.codebook.T).abs().clamp(0, 1 - 1e-6)
        p = F.softmax(margin / self.cfg.cc_temp, dim=-1)
        within = -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
        marg = p.mean(dim=(0, 1, 2))
        cross = -(marg * marg.clamp_min(1e-9).log()).sum()
        return float(within - cross)

    @torch.no_grad()
    def rigidity_monitor(self, out: Dict, margin_frac: float = 0.25) -> float:
        return float(rigidity_barrier(out['M_stem'], self.cfg.D_base, margin_frac))

    @torch.no_grad()
    def in_envelope(self, out: Dict) -> bool:
        dev = abs(intrinsic_deviation(out['M_stem'][0, 0], self.cfg.D_base))
        return dev < dev_critical(self.cfg.D_base)

    # ── parameter groups: three concerns, three gradients ──────────────────
    def battery_parameters(self):
        """The PatchSVAE — trained on INTERNAL recon (pure MSE) only."""
        return list(self.battery.parameters())

    def shell_parameters(self):
        """Lens (fixed, no params) + spectral + decoder — trained on EXTERNAL recon."""
        return list(self.transformer.parameters()) + list(self.decoder.parameters())

    def growth_parameters(self):
        return list(self.growth.parameters())

    # ── processor passthrough (if repo present) ────────────────────────────
    def encode_text(self, text: str) -> torch.Tensor:
        return text_to_image(text, self.cfg.img_size, self.cfg.ps, channels=self.cfg.channels)

    @torch.no_grad()
    def adjudicate(self, text: str, which: str = 'internal') -> str:
        img = self.encode_text(text).unsqueeze(0).to(next(self.parameters()).device)
        out = self(img)
        key = 'internal_recon' if which == 'internal' else 'external_recon'
        rec = out[key][0].detach().cpu()
        return image_to_text(rec, self.cfg.ps, len(text.encode('utf-8')), self.cfg.channels)


def build_model(device: str = 'cuda', **geo_kwargs) -> GeoSVAETransformer:
    cfg = GeoConfig(**{k: v for k, v in geo_kwargs.items()
                       if k in GeoConfig.__dataclass_fields__})
    dev = device if torch.cuda.is_available() else 'cpu'
    return GeoSVAETransformer(cfg, device=dev).to(dev)


# ════════════════════════════════════════════════════════════════════════
#  Trainer — ACTUAL separation (not implied). Battery optimizer sees PURE MSE
#  only, with LR annealing for precision; growth optimizer steps ONLY on
#  growth_loss over the DETACHED stem. Two optimizers + the detach boundary make
#  the separation real, not a sum. cross-contrast / rigidity / in_envelope are
#  read-only monitors — never in any backward.
# ════════════════════════════════════════════════════════════════════════

def byte_recovery(recon: torch.Tensor, img: torch.Tensor) -> float:
    a = (recon * 127.5 + 127.5).round().clamp(0, 255)
    b = (img * 127.5 + 127.5).round().clamp(0, 255)
    return float((a == b).float().mean())
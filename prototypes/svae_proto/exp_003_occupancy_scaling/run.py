"""exp_003_occupancy_scaling.py — SELF-CONTAINED occupancy/lens scaling.

Independence requirement (Phil 2026-05-24): each experiment lives in its
own location, is independently installable, and carries its own copy of
the not-yet-in-core rigidity code ("pypackage deviance capacity"). So this
file:

  • imports ONLY from the installed core: geolip_svae.model.PatchSVAE +
    cv_of (and geolip_svae.inference noise generators, with a fallback).
  • VENDORS everything rigidity-related inline — geometry primitives,
    RigidityFormula, the RigidPatchSVAE wrapper, the train/assess machinery,
    the Colab helpers. No import from exp_002, no dependency on
    model_rigid.py living anywhere.

When the core officially absorbs the rigidity additions, the vendored
blocks below can be replaced with `from geolip_svae.rigidity import ...`.
Until then, this experiment runs standalone.

═══════════════════════════════════════════════════════════════════════
The experiment: enlarge the occupancy lens functional space and confirm
the formula still functions within the lens.

  OCCUPANCY (V) — codebook directions filling RP^(D-1)
  LENS (D)      — spectral dimension; dev_critical(D)=0.02√D widens with D,
                  cv_of band shifts (~0.85–1.05 at D=4 → ~0.20–0.23 at D=16)

Ladder: (4,32) anchor → (4,64) → (4,128) → (8,64) → (8,128) → (16,256).
Per rung: converged? deviation in envelope? cv_of in D-band? p50 emerging?
Verdict: does the emergent formula-adherence SURVIVE enlargement (the
precondition for attention structure)? D=8 rungs settle conjecture C1.

Reference (exp_002, D=4 V=32): converged MSE 0.0048, deviation +0.011 in
envelope (crit 0.040), cv_of 0.888 in band, p50 +1.79° — all emergent.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── installed core (the ONLY external dependency) ──
from geolip_svae.model import PatchSVAE, cv_of


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: geometry primitives (mirror geolip_svae.inference.codebook)
# ════════════════════════════════════════════════════════════════════════

def canonicalize_sign(v: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    nz = v.abs() > eps
    first_idx = nz.float().argmax(dim=-1, keepdim=True)
    first_val = torch.gather(v, -1, first_idx)
    sign = torch.where(first_val < 0, -1.0, 1.0)
    return v * sign


_UNIFORM_MEAN: Dict[int, float] = {}
_UNIFORM_P50: Dict[int, float] = {}


def _populate_uniform(D: int, n_samples: int = 4096, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(int(seed))
    pts = torch.randn(n_samples, D, generator=g)
    pts = pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pts = canonicalize_sign(pts)
    cos = (pts @ pts.T).clamp(-1, 1)
    angles = torch.acos(cos.abs())
    iu = torch.triu_indices(n_samples, n_samples, offset=1)
    vals = angles[iu[0], iu[1]]
    _UNIFORM_MEAN[D] = float(vals.mean())
    _UNIFORM_P50[D] = float(vals.median())


def uniform_projective_angle(D: int) -> float:
    """MEAN pairwise projective angle for uniform RP^(D-1). Deviation baseline."""
    if D not in _UNIFORM_MEAN:
        _populate_uniform(D)
    return _UNIFORM_MEAN[D]


def uniform_projective_p50(D: int) -> float:
    """MEDIAN pairwise projective angle for uniform RP^(D-1). p50-offset
    baseline (median, not mean — the folded distribution is right-skewed,
    so comparing the codebook median against the uniform mean would fold the
    structural median-mean gap into a phantom training signal)."""
    if D not in _UNIFORM_P50:
        _populate_uniform(D)
    return _UNIFORM_P50[D]


def dev_critical(D: int, coeff: float = 0.02) -> float:
    """Rigidity envelope boundary: 0.02√D. Empirical D=4→0.040, D=16→0.080."""
    return coeff * math.sqrt(D)


def _pairwise_proj_angles(axes: torch.Tensor) -> torch.Tensor:
    a = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (a @ a.T).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos.abs())
    n = axes.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=axes.device)
    return ang[iu[0], iu[1]]


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: RigidityFormula (on the M-row codebook)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RigidityConfig:
    dev_critical_coeff: float = 0.02
    envelope_margin: float = 0.7
    uniform_samples: int = 4096
    w_envelope: float = 0.0        # default OFF — measure-only emergent test
    w_cv: float = 0.0              # default OFF
    antipodal_collapse: bool = True
    antipodal_thresh_deg: float = 1.0


# D-appropriate pentachoron-volume CV band (cv_of), per measured basins
def cv_band_for(D: int) -> Tuple[float, float]:
    if D <= 8:
        return (0.85, 1.05)        # h2-class basin (measured)
    return (0.20, 0.23)            # Fresnel/Johanna basin


class RigidityFormula(nn.Module):
    """Formula on a codebook of M-row directions. Stateless except cached
    baselines. Note: CV here is the pairwise-ANGLE CV (a secondary number);
    the CANONICAL CV the bands refer to is cv_of (pentachoron-VOLUME),
    measured separately in assess_codebook()."""

    def __init__(self, D: int, cfg: RigidityConfig):
        super().__init__()
        self.D = D
        self.cfg = cfg
        self.register_buffer('uniform_baseline',
                             torch.tensor(uniform_projective_angle(D)))
        self.register_buffer('uniform_p50',
                             torch.tensor(uniform_projective_p50(D)))
        self.register_buffer('dev_crit',
                             torch.tensor(dev_critical(D, cfg.dev_critical_coeff)))

    def _collapse_antipodal(self, axes: torch.Tensor) -> torch.Tensor:
        if not self.cfg.antipodal_collapse or axes.shape[0] < 2:
            return axes
        a = canonicalize_sign(
            axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12))
        thresh = math.cos(math.radians(self.cfg.antipodal_thresh_deg))
        sim = (a @ a.T).abs()
        keep, used = [], torch.zeros(a.shape[0], dtype=torch.bool,
                                     device=a.device)
        for i in range(a.shape[0]):
            if used[i]:
                continue
            keep.append(i)
            used = used | (sim[i] > thresh)
        return axes[torch.tensor(keep, device=axes.device)]

    def measure(self, M_rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        axes = self._collapse_antipodal(M_rows)
        angles = _pairwise_proj_angles(axes)
        mean_ang = angles.mean()
        return {
            'n_axes': torch.tensor(float(axes.shape[0])),
            'mean_proj_angle': mean_ang,
            'deviation': mean_ang - self.uniform_baseline,
            'angle_cv': angles.std(unbiased=False) / mean_ang.clamp_min(1e-8),
            'p50': angles.median(),
        }

    def losses(self, m: Dict[str, torch.Tensor]) -> torch.Tensor:
        thresh = self.cfg.envelope_margin * self.dev_crit
        over = (m['deviation'].abs() - thresh).clamp_min(0.0)
        l_env = over.pow(2)
        lo, hi = cv_band_for(self.D)
        l_cv = (m['angle_cv'] - (lo + hi) / 2).pow(2)
        return self.cfg.w_envelope * l_env + self.cfg.w_cv * l_cv

    def classify_statute(self, m: Dict[str, torch.Tensor]) -> str:
        dev = float(m['deviation'])
        crit = float(self.dev_crit)
        if abs(dev) < 0.05 and abs(dev) < crit:
            return 'uniform_class'
        if dev > 0 and abs(dev) < crit:
            return 'polytope_class_R1'
        if dev > 0:
            return 'polytope_class_R2'
        return 'sub_uniform'

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
                'p50_deg': math.degrees(float(m['p50'])),
                'p50_offset_deg': math.degrees(
                    float(m['p50'] - self.uniform_p50)),
                'angle_cv': float(m['angle_cv']),
                'statute': self.classify_statute(m),
            }


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: RigidPatchSVAE — core PatchSVAE + formula on the M-rows
# ════════════════════════════════════════════════════════════════════════

class RigidPatchSVAE(PatchSVAE):
    """Core PatchSVAE with the rigidity formula computed on the M-row
    codebook each forward. Spectral core untouched; only reads svd['M']."""

    def __init__(self, *args, rigidity: Optional[RigidityConfig] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rigidity_cfg = rigidity or RigidityConfig()
        self.formula = RigidityFormula(self.D, self.rigidity_cfg)

    def _codebook_rows(self, svd: Dict) -> torch.Tensor:
        M = svd['M']
        if M.dim() == 4:
            B, N, V, D = M.shape
            M = M.reshape(B * N, V, D)
        rows = M.mean(dim=0)
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return canonicalize_sign(rows)

    def forward(self, images: torch.Tensor) -> dict:
        out = super().forward(images)
        rows = self._codebook_rows(out['svd'])
        out['codebook_rows'] = rows
        m = self.formula.measure(rows)
        out['rigidity'] = {**m, 'formula_loss': self.formula.losses(m)}
        return out


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: substrate batch, convergence threshold, assessment
# ════════════════════════════════════════════════════════════════════════

SUBSTRATE_MSE_THRESHOLD = [
    ('16types', 1e-2), ('curriculum', 1e-2), ('omega_noise', 1e-2),
    ('noise', 1e-2), ('tiny_imagenet', 1e-3), ('imagenet', 1e-3),
    ('byte_trigram', 1e-3), ('wikitext', 1e-3), ('wikipedia', 1e-3),
    ('sentencepiece', 1e-3), ('gaussian', 1e-3),
]


def threshold_for(dataset: str) -> float:
    s = dataset.lower()
    for pat, thr in SUBSTRATE_MSE_THRESHOLD:
        if pat in s:
            return thr
    return 4e-3


def make_batch(dataset: str, batch_size: int, img_size: int,
               channels: int, step: int, seed: int = 0) -> torch.Tensor:
    s = dataset.lower()
    sd = seed * 100000 + step
    try:
        from geolip_svae.inference import gen_sixteen_noise, gen_gaussian
        if 'omega' in s or '16' in s or s == 'noise':
            base = gen_sixteen_noise(n=batch_size, size=img_size, seed=sd)
        else:
            base = gen_gaussian(n=batch_size, size=img_size, seed=sd)
    except Exception:
        g = torch.Generator().manual_seed(sd)
        base = torch.randn(batch_size, 3, img_size, img_size,
                           generator=g).clamp(-4, 4)
    c = base.shape[1]
    if c != channels:
        if c > channels:
            base = base[:, :channels]
        else:
            base = torch.cat(
                [base, base[:, :1].repeat(1, channels - c, 1, 1)], dim=1)
    return base


def assess_codebook(model: 'RigidPatchSVAE', images: torch.Tensor,
                    D: int) -> Dict:
    """Formula readout + canonical cv_of (pentachoron-volume) on live codebook."""
    model.eval()
    with torch.no_grad():
        out = model(images)
        M = out['svd']['M']
        first_cb = M[0, 0] if M.dim() == 4 else M[0]
        canonical_cv = float(cv_of(first_cb))
        readout = model.formula.readout(out['codebook_rows'])
    model.train()
    lo, hi = cv_band_for(D)
    readout['canonical_cv_of'] = canonical_cv
    readout['cv_band'] = [lo, hi]
    readout['cv_in_band'] = lo <= canonical_cv <= hi
    return readout


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: single-arm training (soft-hand CV, the cousin mechanism)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ArmConfig:
    D: int = 4
    V: int = 32
    ps: int = 4
    hidden: int = 64
    depth: int = 1
    n_cross: int = 1
    channels: int = 3
    dataset: str = 'omega_noise'
    img_size: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 8
    steps_per_epoch: int = 200
    target_cv: float = 0.95
    cv_weight: float = 0.3
    boost: float = 0.5
    sigma: float = 0.15
    cv_measure_every: int = 50
    assess_every: int = 50
    enforce_formula: bool = False
    w_envelope: float = 0.1
    w_cv: float = 0.05
    seed: int = 0


def train_arm(cfg: ArmConfig, device: torch.device) -> Dict:
    torch.manual_seed(cfg.seed)
    rcfg = RigidityConfig(
        w_envelope=cfg.w_envelope if cfg.enforce_formula else 0.0,
        w_cv=cfg.w_cv if cfg.enforce_formula else 0.0,
    )
    model = RigidPatchSVAE(
        V=cfg.V, D=cfg.D, ps=cfg.ps, hidden=cfg.hidden, depth=cfg.depth,
        n_cross=cfg.n_cross, channels=cfg.channels,
        linear_readout=True, svd_mode='none', match_params=True,
        row_norm='sphere', rigidity=rcfg,
    ).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [train] {n_params:,} params, "
          f"enforce_formula={cfg.enforce_formula}")

    traj: List[Dict] = []
    last_cv, last_prox, step, best_mse = cfg.target_cv, 0.0, 0, float('inf')

    for epoch in range(cfg.epochs):
        ep_mse = 0.0
        for bi in range(cfg.steps_per_epoch):
            images = make_batch(cfg.dataset, cfg.batch_size, cfg.img_size,
                                cfg.channels, step, cfg.seed).to(device)
            opt.zero_grad()
            out = model(images)
            recon_loss = F.mse_loss(out['recon'], images)
            with torch.no_grad():
                if bi % cfg.cv_measure_every == 0:
                    M = out['svd']['M']
                    cb = M[0, 0] if M.dim() == 4 else M[0]
                    cur = cv_of(cb)
                    if cur > 0:
                        last_cv = cur
                    delta = last_cv - cfg.target_cv
                    last_prox = math.exp(-delta ** 2 / (2 * cfg.sigma ** 2))
            recon_w = 1.0 + cfg.boost * last_prox
            cv_pen = cfg.cv_weight * (1.0 - last_prox)
            loss = recon_w * recon_loss + cv_pen * (last_cv - cfg.target_cv) ** 2
            if cfg.enforce_formula:
                loss = loss + out['rigidity']['formula_loss']
            loss.backward()
            opt.step()

            mse_val = float(recon_loss.detach())
            ep_mse += mse_val
            best_mse = min(best_mse, mse_val)
            if step % cfg.assess_every == 0:
                a = assess_codebook(model, images, cfg.D)
                a.update(step=step, epoch=epoch, recon_mse=mse_val)
                traj.append(a)
            step += 1
        print(f"  [train] epoch {epoch:2d}: "
              f"mse={ep_mse/cfg.steps_per_epoch:.5f} "
              f"cv_of={last_cv:.3f} best={best_mse:.6f}")

    return {
        'n_params': n_params, 'best_mse': best_mse,
        'mse_threshold': threshold_for(cfg.dataset),
        'converged': best_mse < threshold_for(cfg.dataset),
        'final_cv_of': last_cv, 'trajectory': traj,
    }


# ════════════════════════════════════════════════════════════════════════
#  Occupancy / lens scaling sweep
# ════════════════════════════════════════════════════════════════════════

DEFAULT_LADDER: List[Tuple[int, int]] = [
    (4, 32), (4, 64), (4, 128), (8, 64), (8, 128), (16, 256),
]


@dataclass
class SweepConfig:
    dataset: str = 'omega_noise'
    epochs: int = 8
    steps_per_epoch: int = 200
    enforce_formula: bool = False
    out_dir: str = './exp_003_results'
    seed: int = 0


def sweep(cfg: SweepConfig,
          ladder: Optional[List[Tuple[int, int]]] = None) -> Dict:
    ladder = ladder or DEFAULT_LADDER
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("exp_003 occupancy/lens scaling (self-contained)")
    print(f"  dataset={cfg.dataset} epochs/rung={cfg.epochs} "
          f"enforce_formula={cfg.enforce_formula} device={device}")
    print(f"  ladder: {ladder}")
    print("=" * 70)

    rungs: List[Dict] = []
    t0 = time.time()
    for i, (D, V) in enumerate(ladder):
        print(f"\n── rung {i+1}/{len(ladder)}: D={D} V={V} "
              f"(cv band {cv_band_for(D)}, dev_crit {dev_critical(D):.4f}) ──")
        arm_cfg = ArmConfig(
            D=D, V=V, dataset=cfg.dataset, epochs=cfg.epochs,
            steps_per_epoch=cfg.steps_per_epoch,
            enforce_formula=cfg.enforce_formula, seed=cfg.seed,
        )
        lo, hi = cv_band_for(D)
        arm_cfg.target_cv = (lo + hi) / 2
        res = train_arm(arm_cfg, device)
        final = res['trajectory'][-1] if res['trajectory'] else {}
        rungs.append({
            'D': D, 'V': V,
            'converged': res['converged'], 'best_mse': res['best_mse'],
            'deviation': final.get('deviation'),
            'dev_critical': final.get('dev_critical'),
            'in_envelope': final.get('in_envelope'),
            'statute': final.get('statute'),
            'cv_of': final.get('canonical_cv_of'),
            'cv_in_band': final.get('cv_in_band'),
            'p50_offset_deg': final.get('p50_offset_deg'),
            'n_params': res['n_params'],
            'trajectory': res['trajectory'],
        })

    dt = time.time() - t0
    all_conv = all(r['converged'] for r in rungs)
    all_env = all(r['in_envelope'] for r in rungs)
    all_band = all(r['cv_in_band'] for r in rungs)
    n_p50 = sum(1 for r in rungs if (r['p50_offset_deg'] or 0) > 1.0)

    verdict = {
        'formula_scales': all_conv and all_env and all_band,
        'all_converged': all_conv, 'all_in_envelope': all_env,
        'all_cv_in_band': all_band, 'n_rungs_p50_emerging': n_p50,
        'n_rungs': len(rungs),
        'dev_critical_check': [
            {'D': r['D'], 'deviation': r['deviation'],
             'dev_critical': r['dev_critical'], 'in_envelope': r['in_envelope']}
            for r in rungs],
    }
    report = {
        'sweep_config': asdict(cfg), 'ladder': ladder,
        'verdict': verdict,
        'rungs': [{k: v for k, v in r.items() if k != 'trajectory'}
                  for r in rungs],
        'elapsed_sec': dt,
    }
    with open(out_dir / f'occupancy_scaling_{cfg.dataset}.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open(out_dir / f'occupancy_scaling_{cfg.dataset}_traj.json', 'w') as f:
        json.dump({f"d{r['D']}_v{r['V']}": r['trajectory'] for r in rungs},
                  f, indent=2)

    print("\n" + "=" * 70)
    print("SCALING VERDICT")
    print("=" * 70)
    print(f"  formula_scales: {verdict['formula_scales']} "
          f"(conv={all_conv}, env={all_env}, band={all_band})  "
          f"p50 emerging {n_p50}/{len(rungs)}")
    print(f"\n  {'D':>3} {'V':>4} {'conv':>5} {'mse':>9} {'dev':>8} "
          f"{'crit':>7} {'env':>4} {'cv_of':>7} {'band':>5} {'p50°':>6} "
          f"{'statute':>18}")
    for r in rungs:
        print(f"  {r['D']:>3} {r['V']:>4} {str(r['converged']):>5} "
              f"{r['best_mse']:>9.6f} {(r['deviation'] or 0):>+8.4f} "
              f"{(r['dev_critical'] or 0):>7.4f} {str(r['in_envelope']):>4} "
              f"{(r['cv_of'] or 0):>7.4f} {str(r['cv_in_band']):>5} "
              f"{(r['p50_offset_deg'] or 0):>+6.2f} {str(r['statute']):>18}")
    print(f"\n  elapsed: {dt:.1f}s")
    print(f"  report: {out_dir / f'occupancy_scaling_{cfg.dataset}.json'}")
    return report


# ════════════════════════════════════════════════════════════════════════
#  VENDORED: Colab-proof entry points
# ════════════════════════════════════════════════════════════════════════

def _is_jupyter_kernel() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def _filter_jupyter_args(argv):
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == '-f':
            skip = True
            continue
        if arg.startswith('-f=') or arg.endswith('.json'):
            continue
        out.append(arg)
    return out


def run(**kwargs):
    """Notebook entry — no CLI parsing.
        from exp_003_occupancy_scaling import run
        run(dataset='omega_noise', epochs=8)
        run(ladder=[(4,32),(8,64),(16,256)])
    """
    ladder = kwargs.pop('ladder', None)
    cfg = SweepConfig(**{k: v for k, v in kwargs.items()
                         if k in SweepConfig.__dataclass_fields__})
    return sweep(cfg, ladder=ladder)


def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv[1:]
    if _is_jupyter_kernel():
        argv = _filter_jupyter_args(argv)
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='omega_noise')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--steps-per-epoch', type=int, default=200)
    p.add_argument('--enforce-formula', action='store_true')
    p.add_argument('--out-dir', default='./exp_003_results')
    args, _unknown = p.parse_known_args(argv)
    cfg = SweepConfig(dataset=args.dataset, epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch,
                      enforce_formula=args.enforce_formula,
                      out_dir=args.out_dir)
    return sweep(cfg)


if __name__ == '__main__':
    main()
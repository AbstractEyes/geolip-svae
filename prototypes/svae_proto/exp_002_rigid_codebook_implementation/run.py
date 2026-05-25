"""exp_002_rigid_convergence.py — train model_rigid + assess convergence.

Question (Phil 2026-05-24): does the RigidPatchSVAE converge, and do its
elements behave like their adjacent SVAE structural cousins, or do they
fail?

Design — the first run MEASURES, it does not over-enforce:

  • Train with the EXISTING soft-hand CV mechanism (the way the real
    trainer trains every cousin): cv_of measured under no_grad every K
    batches, modulating reconstruction weight via a Gaussian proximity
    kernel. NOT a direct differentiable geometry gradient. This is how
    Freckles/Johanna/Fresnel/h2 all converged — match it so the
    comparison is honest.

  • Measure the rigidity formula as DIAGNOSTICS every N steps:
      - cv_of (pentachoron-volume CV) vs the D-appropriate band
            D=4  (h2-class):     ~0.85–1.05
            D=16 (Fresnel/Joh):  ~0.20–0.23
      - deviation = mean_proj_angle(M_rows) − uniform_baseline(D),
            vs dev_critical(D) = 0.02√D  (Regime 1 / Regime 2 boundary)
      - p50_offset = codebook_p50 − uniform_p50  (the training-driven
            signal; +2.5° at D=4 envelope in the catalog)
      - statute class (uniform / polytope_R1 / polytope_R2 / sub_uniform)

  • Optional --baseline: also train a plain PatchSVAE cousin with
    identical config and compare convergence + formula trajectories. The
    cousin tells us what "normal" looks like for this substrate/D.

  • Optional --enforce-formula: turn on the differentiable envelope/cv
    formula losses (the aggressive path) to test whether direct geometry
    gradients help or hurt vs the soft-hand cousin mechanism.

Convergence verdict: substrate-adaptive MSE threshold (from audit v2):
  noise/curriculum/16types → 1e-2 ; imagenet/text/gaussian → 1e-3.

Formula-adherence verdict: did the codebook land in the predicted regime
(deviation sign + envelope), CV band, and does the p50 offset emerge?

NOTE ON model_rigid.py CV: the pushed model_rigid.py pins a pairwise-ANGLE
CV to 0.215. The canonical CV (the 0.20–0.23 / 0.85–1.05 band) is cv_of =
pentachoron-VOLUME CV, and it's D-dependent. This experiment tracks the
canonical cv_of and flags the discrepancy; recommend updating
model_rigid.py's cv_attractor to cv_of with D-appropriate bands after this
run confirms where the codebook naturally lands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ExpConfig:
    # Model (H2-class defaults; the D=4 sphere-solver regime)
    D: int = 4
    V: int = 64
    ps: int = 16
    hidden: int = 64
    depth: int = 1
    n_cross: int = 1
    channels: int = 3
    linear_readout: bool = True
    svd_mode: str = 'none'
    match_params: bool = True
    row_norm: str = 'sphere'

    # Training
    dataset: str = 'omega_noise'
    img_size: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 10
    steps_per_epoch: int = 200          # for synthetic-noise substrates

    # Soft-hand CV (the cousin mechanism) — D-appropriate target
    target_cv: float = 0.9              # D=4 h2 basin; use 0.215 for D=16
    cv_weight: float = 0.3
    boost: float = 0.5
    sigma: float = 0.15
    cv_measure_every: int = 50          # batches between cv_of measurements

    # Formula diagnostics / enforcement
    assess_every: int = 50              # steps between full formula readouts
    enforce_formula: bool = False       # if True, add differentiable env+cv losses
    w_envelope: float = 0.1             # only used if enforce_formula
    w_cv: float = 0.05                  # only used if enforce_formula

    # Convergence verdict
    mse_threshold: float = 1e-2         # substrate-adaptive; set from dataset

    # Output
    out_dir: str = './exp_002_results'
    run_name: str = 'rigid_d4_omega'
    seed: int = 0


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


# D-appropriate CV band (pentachoron-volume cv_of)
def cv_band_for(D: int) -> tuple:
    if D <= 8:
        return (0.85, 1.05)             # h2-class basin (measured)
    return (0.20, 0.23)                 # Fresnel/Johanna basin


# ════════════════════════════════════════════════════════════════════════
#  Substrate batch generator (synthetic; matches the trainer's noise path)
# ════════════════════════════════════════════════════════════════════════

def make_batch(cfg: ExpConfig, step: int) -> torch.Tensor:
    """Substrate-appropriate batch. For noise substrates uses the real
    gen_sixteen_noise if available, else Gaussian. Channel-conformed."""
    seed = cfg.seed * 100000 + step
    g = torch.Generator().manual_seed(seed)
    ds = cfg.dataset.lower()
    try:
        from geolip_svae.inference import gen_sixteen_noise, gen_gaussian
        if 'omega' in ds or '16' in ds or ds == 'noise':
            base = gen_sixteen_noise(n=cfg.batch_size, size=cfg.img_size,
                                      seed=seed)
        else:
            base = gen_gaussian(n=cfg.batch_size, size=cfg.img_size, seed=seed)
    except Exception:
        base = torch.randn(cfg.batch_size, 3, cfg.img_size, cfg.img_size,
                            generator=g).clamp(-4, 4)
    # channel-conform
    c = base.shape[1]
    if c != cfg.channels:
        if c > cfg.channels:
            base = base[:, :cfg.channels]
        else:
            pad = base[:, :1].repeat(1, cfg.channels - c, 1, 1)
            base = torch.cat([base, pad], dim=1)
    return base


# ════════════════════════════════════════════════════════════════════════
#  Assessment — formula adherence on the live codebook
# ════════════════════════════════════════════════════════════════════════

def assess_codebook(model, images: torch.Tensor, D: int) -> Dict:
    """Full formula readout on the live M-row codebook + canonical cv_of.

    Combines the RigidityFormula readout (deviation, p50-offset, statute)
    with the canonical cv_of (pentachoron-volume CV) measured the way the
    trainer measures it.
    """
    from geolip_svae.model import cv_of
    model.eval()
    with torch.no_grad():
        out = model(images)
        svd = out['svd']
        # Canonical CV: pentachoron-volume CV on the first patch codebook,
        # matching train.py's cv_of(out['svd']['M'][0,0]).
        M = svd['M']
        first_cb = M[0, 0] if M.dim() == 4 else M[0]
        canonical_cv = cv_of(first_cb)

        readout = {}
        # Formula readout if the model carries the RigidityFormula
        if hasattr(model, 'formula') and 'codebook_rows' in out:
            readout = model.formula.readout(out['codebook_rows'])
        elif hasattr(model, 'formula'):
            rows = model._codebook_rows(svd)
            readout = model.formula.readout(rows)
    model.train()

    lo, hi = cv_band_for(D)
    readout['canonical_cv_of'] = float(canonical_cv)
    readout['cv_band'] = [lo, hi]
    readout['cv_in_band'] = lo <= float(canonical_cv) <= hi
    return readout


# ════════════════════════════════════════════════════════════════════════
#  Training
# ════════════════════════════════════════════════════════════════════════

def build_model(cfg: ExpConfig, rigid: bool):
    """Build either RigidPatchSVAE (model_rigid) or plain PatchSVAE cousin."""
    kw = dict(
        V=cfg.V, D=cfg.D, ps=cfg.ps, hidden=cfg.hidden, depth=cfg.depth,
        n_cross=cfg.n_cross, channels=cfg.channels,
        linear_readout=cfg.linear_readout, svd_mode=cfg.svd_mode,
        match_params=cfg.match_params, row_norm=cfg.row_norm,
    )
    if rigid:
        # The pushed module is model_rigid.py
        try:
            from geolip_svae.model_rigid import RigidPatchSVAE, RigidityConfig
        except Exception:
            from model_rigid import RigidPatchSVAE, RigidityConfig  # local fallback
        rcfg = RigidityConfig(
            w_envelope=cfg.w_envelope if cfg.enforce_formula else 0.0,
            w_cv=cfg.w_cv if cfg.enforce_formula else 0.0,
        )
        return RigidPatchSVAE(rigidity=rcfg, **kw)
    else:
        from geolip_svae.model import PatchSVAE
        return PatchSVAE(**kw)


def train_arm(cfg: ExpConfig, rigid: bool, device: torch.device) -> Dict:
    """Train one arm (rigid or baseline). Returns the trajectory log."""
    from geolip_svae.model import cv_of

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, rigid=rigid).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in model.parameters())

    arm = 'rigid' if rigid else 'baseline'
    print(f"\n[{arm}] {n_params:,} params, D={cfg.D} V={cfg.V} "
          f"linear_readout={cfg.linear_readout} svd_mode={cfg.svd_mode}")
    print(f"[{arm}] soft-hand CV: target={cfg.target_cv}, "
          f"enforce_formula={cfg.enforce_formula and rigid}")

    traj: List[Dict] = []
    last_cv = cfg.target_cv
    last_prox = 0.0
    global_step = 0
    best_mse = float('inf')

    for epoch in range(cfg.epochs):
        epoch_mse = 0.0
        for batch_idx in range(cfg.steps_per_epoch):
            images = make_batch(cfg, global_step).to(device)
            opt.zero_grad()
            out = model(images)
            recon_loss = F.mse_loss(out['recon'], images)

            # Soft-hand CV (the cousin mechanism) — cv_of under no_grad
            with torch.no_grad():
                if batch_idx % cfg.cv_measure_every == 0:
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

            # Optional: add the differentiable formula losses (rigid + enforce)
            if rigid and cfg.enforce_formula and 'rigidity' in out:
                loss = loss + out['rigidity']['formula_loss']

            loss.backward()
            opt.step()

            mse_val = float(recon_loss.detach())
            epoch_mse += mse_val
            best_mse = min(best_mse, mse_val)

            # Periodic full assessment
            if global_step % cfg.assess_every == 0:
                a = assess_codebook(model, images, cfg.D) if rigid else \
                    _baseline_assess(model, images, cfg.D)
                a['step'] = global_step
                a['epoch'] = epoch
                a['recon_mse'] = mse_val
                traj.append(a)

            global_step += 1

        epoch_mse /= cfg.steps_per_epoch
        print(f"[{arm}] epoch {epoch:2d}: mse={epoch_mse:.5f} "
              f"cv_of={last_cv:.3f} best_mse={best_mse:.6f}")

    thr = cfg.mse_threshold
    converged = best_mse < thr
    return {
        'arm': arm,
        'n_params': n_params,
        'best_mse': best_mse,
        'mse_threshold': thr,
        'converged': converged,
        'final_cv_of': last_cv,
        'trajectory': traj,
    }


def _baseline_assess(model, images: torch.Tensor, D: int) -> Dict:
    """Assessment for the plain PatchSVAE cousin — measures the same formula
    quantities on its M-rows so we can compare, even though it has no
    RigidityFormula attached."""
    from geolip_svae.model import cv_of
    # Lazily borrow the formula machinery
    try:
        from geolip_svae.model_rigid import RigidityFormula, RigidityConfig
    except Exception:
        from model_rigid import RigidityFormula, RigidityConfig
    formula = RigidityFormula(D, RigidityConfig()).to(images.device)

    model.eval()
    with torch.no_grad():
        out = model(images)
        svd = out['svd']
        M = svd['M']
        first_cb = M[0, 0] if M.dim() == 4 else M[0]
        canonical_cv = cv_of(first_cb)
        rows = M.reshape(-1, M.shape[-2], M.shape[-1]).mean(0) \
            if M.dim() == 4 else M.mean(0)
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
        readout = formula.readout(rows)
    model.train()
    lo, hi = cv_band_for(D)
    readout['canonical_cv_of'] = float(canonical_cv)
    readout['cv_band'] = [lo, hi]
    readout['cv_in_band'] = lo <= float(canonical_cv) <= hi
    return readout


# ════════════════════════════════════════════════════════════════════════
#  Verdict
# ════════════════════════════════════════════════════════════════════════

def verdict(cfg: ExpConfig, rigid_result: Dict,
            baseline_result: Optional[Dict]) -> Dict:
    """Convergence + formula-adherence verdict, with cousin comparison."""
    final = rigid_result['trajectory'][-1] if rigid_result['trajectory'] else {}

    v = {
        'converged': rigid_result['converged'],
        'best_mse': rigid_result['best_mse'],
        'mse_threshold': rigid_result['mse_threshold'],
        'final_deviation': final.get('deviation'),
        'final_dev_critical': final.get('dev_critical'),
        'final_in_envelope': final.get('in_envelope'),
        'final_statute': final.get('statute'),
        'final_canonical_cv': final.get('canonical_cv_of'),
        'final_cv_in_band': final.get('cv_in_band'),
        'final_p50_offset_deg': final.get('p50_offset_deg'),
    }

    # Formula adherence: did it land where the catalog predicts?
    adherence = []
    if final.get('in_envelope'):
        adherence.append('deviation_in_envelope')
    if final.get('cv_in_band'):
        adherence.append('cv_in_band')
    p50 = final.get('p50_offset_deg')
    if p50 is not None and p50 > 1.0:
        adherence.append('p50_offset_emerging')
    v['formula_adherence'] = adherence

    # Cousin comparison
    if baseline_result is not None:
        bfinal = (baseline_result['trajectory'][-1]
                  if baseline_result['trajectory'] else {})
        v['cousin_comparison'] = {
            'rigid_converged': rigid_result['converged'],
            'baseline_converged': baseline_result['converged'],
            'rigid_best_mse': rigid_result['best_mse'],
            'baseline_best_mse': baseline_result['best_mse'],
            'mse_ratio': (rigid_result['best_mse']
                          / max(baseline_result['best_mse'], 1e-12)),
            'rigid_deviation': final.get('deviation'),
            'baseline_deviation': bfinal.get('deviation'),
            'rigid_cv': final.get('canonical_cv_of'),
            'baseline_cv': bfinal.get('canonical_cv_of'),
            'behaves_like_cousin': (
                rigid_result['converged'] == baseline_result['converged']
                and 0.5 < (rigid_result['best_mse']
                           / max(baseline_result['best_mse'], 1e-12)) < 2.0
            ),
        }
    return v


# ════════════════════════════════════════════════════════════════════════
#  Colab / Jupyter arg safety
# ════════════════════════════════════════════════════════════════════════

def _is_jupyter_kernel() -> bool:
    """Detect IPython/Jupyter/Colab kernel. Jupyter injects
    `-f /path/to/kernel.json` into argv, which breaks naive argparse."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def _filter_jupyter_args(argv):
    """Strip the injected `-f /path/to/kernel.json` pair and any other
    kernel-launcher noise so argparse sees only real flags."""
    out = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == '-f':
            skip_next = True
            continue
        if arg.startswith('-f=') or arg.endswith('.json'):
            continue
        out.append(arg)
    return out


def run(**kwargs):
    """Direct entry point for notebooks — bypasses CLI entirely.

    Usage in Colab:
        from exp_002_rigid_convergence import run
        run(D=4, dataset='omega_noise', epochs=10, baseline=True)

    Any ExpConfig field is accepted, plus the run-flags `baseline` and
    `enforce_formula`. No argv parsing, so Colab's injected args can't
    interfere.
    """
    baseline = kwargs.pop('baseline', False)
    enforce_formula = kwargs.pop('enforce_formula', False)

    cfg = ExpConfig(**{k: v for k, v in kwargs.items()
                       if k in ExpConfig.__dataclass_fields__})
    cfg.enforce_formula = enforce_formula
    cfg.mse_threshold = threshold_for(cfg.dataset)
    if 'target_cv' not in kwargs:
        lo, hi = cv_band_for(cfg.D)
        cfg.target_cv = (lo + hi) / 2
    if 'run_name' not in kwargs:
        cfg.run_name = f"rigid_d{cfg.D}_{cfg.dataset}"
    return _execute(cfg, baseline=baseline)


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv[1:]
    # Colab/Jupyter safety: strip injected kernel args before parsing
    if _is_jupyter_kernel():
        argv = _filter_jupyter_args(argv)

    p = argparse.ArgumentParser()
    p.add_argument('--D', type=int, default=4)
    p.add_argument('--V', type=int, default=32)
    p.add_argument('--dataset', default='omega_noise')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=200)
    p.add_argument('--target-cv', type=float, default=None,
                   help='soft-hand CV target; default D-appropriate')
    p.add_argument('--baseline', action='store_true',
                   help='also train a plain PatchSVAE cousin for comparison')
    p.add_argument('--enforce-formula', action='store_true',
                   help='turn on differentiable envelope+cv formula losses')
    p.add_argument('--out-dir', default='./exp_002_results')
    p.add_argument('--run-name', default=None)
    # parse_known_args so any stray injected flag can't crash the run
    args, _unknown = p.parse_known_args(argv)

    cfg = ExpConfig(
        D=args.D, V=args.V, dataset=args.dataset, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        enforce_formula=args.enforce_formula, out_dir=args.out_dir,
    )
    cfg.mse_threshold = threshold_for(cfg.dataset)
    if args.target_cv is not None:
        cfg.target_cv = args.target_cv
    else:
        lo, hi = cv_band_for(cfg.D)
        cfg.target_cv = (lo + hi) / 2          # D-appropriate band center
    cfg.run_name = args.run_name or f"rigid_d{cfg.D}_{cfg.dataset}"

    return _execute(cfg, baseline=args.baseline)


def _execute(cfg: ExpConfig, baseline: bool = False) -> Dict:
    """Shared run body — called by both main() (CLI) and run() (notebook)."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 64)
    print(f"exp_002 rigid convergence — {cfg.run_name}")
    print(f"  D={cfg.D} V={cfg.V} dataset={cfg.dataset}  device={device}")
    print(f"  mse_threshold={cfg.mse_threshold} "
          f"target_cv={cfg.target_cv:.3f} (band {cv_band_for(cfg.D)})")
    print(f"  enforce_formula={cfg.enforce_formula} baseline={baseline}")
    print("=" * 64)

    t0 = time.time()
    rigid_result = train_arm(cfg, rigid=True, device=device)
    baseline_result = None
    if baseline:
        baseline_result = train_arm(cfg, rigid=False, device=device)

    v = verdict(cfg, rigid_result, baseline_result)
    dt = time.time() - t0

    report = {
        'config': asdict(cfg),
        'verdict': v,
        'rigid_result': {k: val for k, val in rigid_result.items()
                          if k != 'trajectory'},
        'baseline_result': ({k: val for k, val in baseline_result.items()
                             if k != 'trajectory'}
                            if baseline_result else None),
        'elapsed_sec': dt,
    }
    with open(out_dir / f'{cfg.run_name}_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open(out_dir / f'{cfg.run_name}_trajectory.json', 'w') as f:
        json.dump({
            'rigid': rigid_result['trajectory'],
            'baseline': (baseline_result['trajectory']
                         if baseline_result else None),
        }, f, indent=2)

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  converged:        {v['converged']} "
          f"(best_mse={v['best_mse']:.6f} vs threshold {v['mse_threshold']})")
    if v['final_deviation'] is not None:
        print(f"  final deviation:  {v['final_deviation']:+.4f} "
              f"(dev_crit {v['final_dev_critical']:.4f}, "
              f"in_envelope={v['final_in_envelope']})")
        print(f"  final statute:    {v['final_statute']}")
        print(f"  canonical cv_of:  {v['final_canonical_cv']:.4f} "
              f"(in_band={v['final_cv_in_band']})")
        print(f"  p50 offset:       {v['final_p50_offset_deg']:+.2f}°")
    print(f"  formula adherence: {v['formula_adherence']}")
    if 'cousin_comparison' in v:
        cc = v['cousin_comparison']
        print(f"\n  cousin comparison:")
        print(f"    behaves_like_cousin: {cc['behaves_like_cousin']}")
        print(f"    rigid mse {cc['rigid_best_mse']:.6f} vs "
              f"baseline {cc['baseline_best_mse']:.6f} "
              f"(ratio {cc['mse_ratio']:.2f})")
        if cc['rigid_deviation'] is not None and cc['baseline_deviation'] is not None:
            print(f"    rigid dev {cc['rigid_deviation']:+.4f} vs "
                  f"baseline dev {cc['baseline_deviation']:+.4f}")
    print(f"\n  elapsed: {dt:.1f}s")
    print(f"  report:  {out_dir / f'{cfg.run_name}_report.json'}")
    return report


if __name__ == '__main__':
    main()
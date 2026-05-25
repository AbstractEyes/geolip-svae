"""exp_003_occupancy_scaling.py — does the formula hold as we enlarge the
occupancy lens functional space?

Phil 2026-05-24: "Can we simply enlarge the occupancy lens functional space
of this, and it function how we expect it now within the lens?"

The exp_002 run confirmed the formula holds at the smallest h2 regime
(D=4, V=32): converged (MSE 0.0048), deviation in envelope (+0.011 vs
crit 0.040), cv_of in the D=4 band (0.888 ∈ [0.85, 1.05]), p50 offset
emerging (+1.79°) — all WITHOUT enforcement, purely emergent under the
soft-hand cousin training.

This sweep enlarges the functional space along two axes and asks whether
that emergent formula-adherence survives:

  • OCCUPANCY (V) — how many codebook directions fill the projective
    sphere. Enlarging V at fixed D increases occupancy density.

  • LENS (D) — the spectral dimension, the dimensionality of the working
    space. Enlarging D changes the basin: dev_critical(D)=0.02√D widens,
    and the cv_of band shifts (~0.85–1.05 at D=4 → ~0.20–0.23 at D=16).

The enlargement ladder (anchor → progressively larger occupancy + lens):

    (D=4,  V=32)   anchor — the proven point
    (D=4,  V=64)   2× occupancy, same lens
    (D=4,  V=128)  4× occupancy, same lens
    (D=8,  V=64)   enlarge lens to D=8  (+ conjecture-C1 data point)
    (D=8,  V=128)  enlarge lens + occupancy
    (D=16, V=256)  full Fresnel/Johanna-scale lens

For each rung: converged? deviation in envelope? cv_of in D-band? p50
emerging? The verdict is whether the formula SCALES — whether enlarging
the lens preserves the emergent rigidity the catalog described, which is
the precondition for building attention structure on top of it.

Reuses exp_002.run() per rung (no duplicated training code). Colab-proof:
run(**kwargs) for notebooks, jupyter-arg-filtered main() for CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# exp_002 is the training+assessment engine; import its run() + helpers.
try:
    from exp_002_rigid_codebook_implementation import run as run_arm, cv_band_for, \
        threshold_for, _is_jupyter_kernel, _filter_jupyter_args
except Exception:
    # package-qualified fallback if dropped into the prototypes tree
    from geolip_svae.experimental.exp_002_rigid_codebook_implementation import (  # type: ignore
        run as run_arm, cv_band_for, threshold_for,
        _is_jupyter_kernel, _filter_jupyter_args)


# (D, V) enlargement ladder — occupancy (V) × lens (D)
DEFAULT_LADDER: List[Tuple[int, int]] = [
    (4, 32),     # anchor — the proven point
    (4, 64),     # 2× occupancy, same lens
    (4, 128),    # 4× occupancy, same lens
    (8, 64),     # enlarge lens to D=8  (+ C1 data point)
    (8, 128),    # enlarge lens + occupancy
    (16, 256),   # full Fresnel/Johanna-scale lens
]


@dataclass
class SweepConfig:
    dataset: str = 'omega_noise'
    epochs: int = 8                 # per rung; trim for sweep speed
    steps_per_epoch: int = 200
    enforce_formula: bool = False   # measure-only by default (emergent test)
    out_dir: str = './exp_003_results'


def _rung_adherence(verdict: Dict) -> Dict:
    """Pull the formula-adherence signals out of an exp_002 verdict."""
    return {
        'converged': verdict.get('converged'),
        'best_mse': verdict.get('best_mse'),
        'deviation': verdict.get('final_deviation'),
        'dev_critical': verdict.get('final_dev_critical'),
        'in_envelope': verdict.get('final_in_envelope'),
        'statute': verdict.get('final_statute'),
        'cv_of': verdict.get('final_canonical_cv'),
        'cv_in_band': verdict.get('final_cv_in_band'),
        'p50_offset_deg': verdict.get('final_p50_offset_deg'),
        'adherence': verdict.get('formula_adherence', []),
    }


def sweep(cfg: SweepConfig,
          ladder: Optional[List[Tuple[int, int]]] = None) -> Dict:
    ladder = ladder or DEFAULT_LADDER
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("exp_003 occupancy/lens scaling sweep")
    print(f"  dataset={cfg.dataset}  epochs/rung={cfg.epochs}  "
          f"enforce_formula={cfg.enforce_formula}")
    print(f"  ladder: {ladder}")
    print("=" * 70)

    rungs: List[Dict] = []
    t0 = time.time()
    for i, (D, V) in enumerate(ladder):
        band = cv_band_for(D)
        print(f"\n── rung {i+1}/{len(ladder)}: D={D} V={V} "
              f"(cv band {band}, dev_crit {0.02 * D ** 0.5:.4f}) ──")
        report = run_arm(
            D=D, V=V, dataset=cfg.dataset, epochs=cfg.epochs,
            steps_per_epoch=cfg.steps_per_epoch,
            enforce_formula=cfg.enforce_formula,
            out_dir=str(out_dir),
            run_name=f"rung_d{D}_v{V}_{cfg.dataset}",
        )
        a = _rung_adherence(report['verdict'])
        a['D'] = D
        a['V'] = V
        rungs.append(a)

    dt = time.time() - t0

    # ── Scaling verdict ──
    all_converged = all(r['converged'] for r in rungs)
    all_in_env = all(r['in_envelope'] for r in rungs)
    all_cv_band = all(r['cv_in_band'] for r in rungs)
    n_p50 = sum(1 for r in rungs
                if (r['p50_offset_deg'] or 0) > 1.0)

    # Does the formula hold as occupancy enlarges (fixed D)?
    by_D: Dict[int, List[Dict]] = {}
    for r in rungs:
        by_D.setdefault(r['D'], []).append(r)
    occupancy_stable = {}
    for D, rs in by_D.items():
        if len(rs) < 2:
            continue
        devs = [r['deviation'] for r in rs if r['deviation'] is not None]
        cvs = [r['cv_of'] for r in rs if r['cv_of'] is not None]
        occupancy_stable[D] = {
            'n_rungs': len(rs),
            'all_in_envelope': all(r['in_envelope'] for r in rs),
            'all_cv_in_band': all(r['cv_in_band'] for r in rs),
            'deviation_range': [min(devs), max(devs)] if devs else None,
            'cv_range': [min(cvs), max(cvs)] if cvs else None,
        }

    # Does dev_critical(D)=0.02√D track across the lens enlargement?
    # (the C1 check — D=8 rungs give the missing point)
    dev_critical_check = [
        {'D': r['D'], 'deviation': r['deviation'],
         'dev_critical': r['dev_critical'],
         'in_envelope': r['in_envelope']}
        for r in rungs
    ]

    verdict = {
        'formula_scales': all_converged and all_in_env and all_cv_band,
        'all_converged': all_converged,
        'all_in_envelope': all_in_env,
        'all_cv_in_band': all_cv_band,
        'n_rungs_p50_emerging': n_p50,
        'n_rungs': len(rungs),
        'occupancy_stability_per_D': occupancy_stable,
        'dev_critical_check': dev_critical_check,
    }

    report = {
        'sweep_config': cfg.__dict__,
        'ladder': ladder,
        'verdict': verdict,
        'rungs': rungs,
        'elapsed_sec': dt,
    }
    with open(out_dir / f'occupancy_scaling_{cfg.dataset}.json', 'w') as f:
        json.dump(report, f, indent=2)

    # ── Print ──
    print("\n" + "=" * 70)
    print("SCALING VERDICT")
    print("=" * 70)
    print(f"  formula_scales: {verdict['formula_scales']}  "
          f"(converged={all_converged}, in_env={all_in_env}, "
          f"cv_band={all_cv_band})")
    print(f"  p50 emerging: {n_p50}/{len(rungs)} rungs")
    print(f"\n  {'D':>3} {'V':>4} {'conv':>5} {'mse':>9} {'dev':>8} "
          f"{'crit':>7} {'env':>4} {'cv_of':>7} {'band':>5} {'p50°':>6} "
          f"{'statute':>18}")
    for r in rungs:
        print(f"  {r['D']:>3} {r['V']:>4} "
              f"{str(r['converged']):>5} "
              f"{r['best_mse']:>9.6f} "
              f"{(r['deviation'] or 0):>+8.4f} "
              f"{(r['dev_critical'] or 0):>7.4f} "
              f"{str(r['in_envelope']):>4} "
              f"{(r['cv_of'] or 0):>7.4f} "
              f"{str(r['cv_in_band']):>5} "
              f"{(r['p50_offset_deg'] or 0):>+6.2f} "
              f"{str(r['statute']):>18}")
    print(f"\n  elapsed: {dt:.1f}s")
    print(f"  report: {out_dir / f'occupancy_scaling_{cfg.dataset}.json'}")
    return report


# ════════════════════════════════════════════════════════════════════════
#  Entry points (Colab-proof)
# ════════════════════════════════════════════════════════════════════════

def run(**kwargs):
    """Notebook entry point — no CLI parsing.

        from exp_003_occupancy_scaling import run
        run(dataset='omega_noise', epochs=8)
        run(ladder=[(4,32),(8,64),(16,256)])     # custom ladder
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

    cfg = SweepConfig(
        dataset=args.dataset, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        enforce_formula=args.enforce_formula, out_dir=args.out_dir,
    )
    return sweep(cfg)


if __name__ == '__main__':
    main()
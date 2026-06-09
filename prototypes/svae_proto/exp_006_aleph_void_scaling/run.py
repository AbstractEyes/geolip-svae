"""exp_006_aleph_void_scaling.py — SELF-CONTAINED lifted-cloud topology sweep.

Question (multiscale aleph-void series, exp 006 of 3): does the void/topology
fingerprint (beta2/axis on RP^(D_lens-1)) of the LIFTED M_lens axis cloud survive
— and trend with — the isometric multiscale lift?

Design — REUSE core, do not reimplement:
  • Shell training: geolip_svae.train_aleph.train_aleph_transformer(...).
  • Antipodal collapse: geolip_svae.inference.codebook.identify_antipodal_pairs
    + collapse_to_axes (+ uniform_projective_angle / codebook_mean_projective_angle).
  • Topology probes: geolip_svae.inference.train_codebook.run_topology_analysis
    -> TopologyReport (kNN/PCA always; ripser H2 optional via HAVE_RIPSER).

NOVEL piece (why core can't be used as-is): extract_codebook/create_codebook run
on the frozen aleph's M at D_base. Void SCALING needs the topology of the LIFTED
cloud, so we pull M_lens from AlephTransformer.forward(), aggregate to a
[V, D_lens] codebook, collapse, and run the probes on the D_lens axes.

beta2/axis = TopologyReport.persistence_n_finite['H2'] / n_axes.
No new latent math; no core edits. ripser is an OPTIONAL dep (exp_006 extra).

Colab:
    from svae_proto.exp_006_aleph_void_scaling.run import run
    report = run(ladder=[16, 64, 256], epochs=4)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


DEFAULT_LADDER: List[int] = [16, 64, 256]
DEFAULT_ALEPH_VERSION = "aleph_byte_trigram_tied_hard_K64"


@dataclass
class SweepConfig:
    aleph_hf_version: str = DEFAULT_ALEPH_VERSION
    aleph_repo: str = "AbstractPhil/geolip-aleph-void"
    dataset: str = "byte_trigram"
    ladder: List[int] = field(default_factory=lambda: list(DEFAULT_LADDER))
    stem: str = "m_hat"
    lens_sign: str = "signed"
    n_layers: int = 6
    shell_hidden: int = 512
    epochs: int = 4
    ds_size: int = 200_000
    val_size: int = 4_000
    batch_size: int = 2048
    topo_batches: int = 4              # batches of M_lens aggregated per rung
    ripser_thresh_deg: float = 20.0
    collapse_threshold: float = -0.9
    out_dir: str = "./experiments/exp_006_results"
    upload: bool = False
    seed: int = 0
    device: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════
#  Helpers (self-contained copies)
# ════════════════════════════════════════════════════════════════════════

def _valid_n_heads(d_lens: int, cap: int = 8) -> int:
    for h in range(min(cap, d_lens), 0, -1):
        if d_lens % h == 0:
            return h
    return 1


def _resolve_device(dev: Optional[str]) -> str:
    return dev or ("cuda" if torch.cuda.is_available() else "cpu")


def _build_eval_loader(cfg: SweepConfig, aleph):
    from geolip_svae.train_aleph import PRESETS
    from geolip_svae.dataset_presets import get_dataset_bundle
    ev = dict(PRESETS[cfg.dataset])
    ev.update(patch_size=aleph.patch_size, channels=aleph.channels,
              batch_size=cfg.batch_size,
              ds_size=max(cfg.batch_size, cfg.batch_size * cfg.topo_batches),
              val_size=max(cfg.batch_size, cfg.batch_size * cfg.topo_batches))
    return get_dataset_bundle(ev, channels=aleph.channels).test_loader


@torch.no_grad()
def _lifted_codebook(model, loader, device: str, max_batches: int) -> torch.Tensor:
    """Aggregate M_lens (B,N,V,D_lens) over batches -> a [V, D_lens] codebook
    (mean over all B*N rows), matching how extract_codebook means M to [V, D]."""
    model.eval()
    acc, count = None, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        M_lens = model(images)["M_lens"]                  # (B,N,V,D_lens)
        flat = M_lens.reshape(-1, M_lens.shape[-2], M_lens.shape[-1])  # (B*N,V,D)
        s = flat.sum(0)
        acc = s if acc is None else acc + s
        count += flat.shape[0]
    return (acc / max(count, 1)).cpu()                    # (V, D_lens)


def _topology_of_axes(cb: torch.Tensor, cfg: SweepConfig) -> Dict:
    """Antipodal-collapse a [V, D] codebook then run the core topology probes."""
    from geolip_svae.inference.codebook import (
        identify_antipodal_pairs, collapse_to_axes,
        uniform_projective_angle, codebook_mean_projective_angle,
    )
    from geolip_svae.inference.train_codebook import run_topology_analysis

    pairs, unpaired = identify_antipodal_pairs(cb, threshold=cfg.collapse_threshold)
    axes = collapse_to_axes(cb, pairs, unpaired)          # (n_axes, D)
    n_axes = int(axes.shape[0])
    D = int(cb.shape[1])
    if n_axes < 3:
        return dict(n_axes=n_axes, D=D, beta2_per_axis=None,
                    note="too few axes for topology")

    # The core kNN probe needs scipy; ripser adds H2. Missing deps must not
    # discard already-trained shells — degrade gracefully with a clear note.
    try:
        rep = run_topology_analysis(axes, ripser_thresh_deg=cfg.ripser_thresh_deg)
    except ModuleNotFoundError as e:
        from geolip_svae.inference.codebook import (
            uniform_projective_angle, codebook_mean_projective_angle)
        deviation = float(codebook_mean_projective_angle(axes)
                          - uniform_projective_angle(D))
        return dict(n_axes=n_axes, D=D, n_pairs=len(pairs), n_unpaired=len(unpaired),
                    beta2_per_axis=None, deviation=deviation,
                    note=f"topology unavailable ({e.name} missing) — "
                         f"install svae-proto[exp_006]")
    n_h2 = int(rep.persistence_n_finite.get("H2", 0)) if rep.ripser_available else None
    n_h1 = int(rep.persistence_n_finite.get("H1", 0)) if rep.ripser_available else None
    mean_ang = codebook_mean_projective_angle(axes)
    deviation = float(mean_ang - uniform_projective_angle(D))
    if abs(deviation) < 0.05:
        statute = "uniform_class"
    elif deviation > 0:
        statute = "polytope_class"
    else:
        statute = "sub_uniform"
    return dict(
        n_axes=n_axes, D=D, n_pairs=len(pairs), n_unpaired=len(unpaired),
        ripser_available=bool(rep.ripser_available),
        beta2=n_h2, beta1=n_h1,
        beta2_per_axis=(n_h2 / n_axes if n_h2 is not None else None),
        percolation_thresh_deg=rep.percolation_thresh_deg,
        local_dim_pr_p50=rep.local_dim_pr_p50,
        angular_dist_p50_deg=rep.angular_dist_p50_deg,
        deviation=deviation, statute=statute,
    )


# ════════════════════════════════════════════════════════════════════════
#  Sweep
# ════════════════════════════════════════════════════════════════════════

def sweep(cfg: SweepConfig) -> Dict:
    from geolip_svae.inference.loading import load_model
    from geolip_svae.train_aleph import train_aleph_transformer

    device = _resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"exp_006 void scaling — {cfg.aleph_repo}/{cfg.aleph_hf_version}")
    print(f"  ladder={cfg.ladder}  stem={cfg.stem}  lens_sign={cfg.lens_sign}  "
          f"device={device}")
    print("=" * 72)

    aleph0, _ = load_model(hf_version=cfg.aleph_hf_version,
                           repo_id=cfg.aleph_repo, device=device)
    for p in aleph0.parameters():
        p.requires_grad_(False)
    eval_loader = _build_eval_loader(cfg, aleph0)

    rungs: List[Dict] = []
    t0 = time.time()
    for i, d_lens in enumerate(cfg.ladder):
        if d_lens < aleph0.D:
            print(f"  [skip] D_lens={d_lens} < D_base={aleph0.D}")
            continue
        n_heads = _valid_n_heads(d_lens)
        print(f"\n── rung {i+1}/{len(cfg.ladder)}: D_lens={d_lens} "
              f"n_heads={n_heads}x{cfg.n_layers} ──")
        model, history = train_aleph_transformer(
            cfg.aleph_hf_version,
            D_lens=d_lens, dataset=cfg.dataset, stem=cfg.stem, lens_sign=cfg.lens_sign,
            n_heads=n_heads, n_layers=cfg.n_layers, shell_hidden=cfg.shell_hidden,
            device=device, upload=cfg.upload, aleph_repo=cfg.aleph_repo,
            hf_version=f"exp_006_d{d_lens}_{cfg.stem}_{cfg.lens_sign}",
            save_dir=str(out_dir / f"d{d_lens}_ckpt"), tb_dir=str(out_dir / "tb"),
            cfg_overrides=dict(epochs=cfg.epochs, ds_size=cfg.ds_size,
                               val_size=cfg.val_size, batch_size=cfg.batch_size),
        )
        row = history[-1] if history else (0,) * 7
        cb = _lifted_codebook(model, eval_loader, device, cfg.topo_batches)
        topo = _topology_of_axes(cb, cfg)
        topo.update(D_lens=d_lens, n_heads=n_heads,
                    external_mse=row[2], external_cos=row[3])
        rungs.append(topo)
        del model, cb
        if device == "cuda":
            torch.cuda.empty_cache()

    dt = time.time() - t0
    b2 = [(r["D_lens"], r["beta2_per_axis"]) for r in rungs
          if r.get("beta2_per_axis") is not None]
    verdict = {
        "ripser_available": any(r.get("ripser_available") for r in rungs),
        "any_finite_void": any((r.get("beta2") or 0) > 0 for r in rungs),
        "beta2_per_axis_by_dlens": dict(b2),
        "beta2_trend_nondec": ([v for _, v in b2] == sorted(v for _, v in b2))
        if len(b2) > 1 else None,
    }
    report = {"config": asdict(cfg), "verdict": verdict, "rungs": rungs,
              "elapsed_sec": dt}
    fname = out_dir / f"void_scaling_{cfg.aleph_hf_version}_{cfg.stem}_{cfg.lens_sign}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("VOID-SCALING VERDICT")
    print("=" * 72)
    print(f"  ripser available: {verdict['ripser_available']}   "
          f"any finite void: {verdict['any_finite_void']}   "
          f"beta2/axis non-decreasing: {verdict['beta2_trend_nondec']}")
    print(f"\n  {'D_lens':>6} {'n_axes':>6} {'beta2':>6} {'b2/axis':>8} "
          f"{'percol°':>8} {'loc_dim':>7} {'dev':>8} {'statute':>14} {'ext_cos':>8}")
    for r in rungs:
        b2a = r.get("beta2_per_axis")
        perc = r.get("percolation_thresh_deg")
        print(f"  {r['D_lens']:>6} {r.get('n_axes', 0):>6} "
              f"{str(r.get('beta2')):>6} "
              f"{(f'{b2a:.3f}' if b2a is not None else 'n/a'):>8} "
              f"{(f'{perc:.1f}' if perc is not None else 'n/a'):>8} "
              f"{r.get('local_dim_pr_p50', float('nan')):>7.2f} "
              f"{r.get('deviation', float('nan')):>+8.4f} "
              f"{str(r.get('statute')):>14} {r.get('external_cos', float('nan')):>8.4f}")
    print(f"\n  elapsed: {dt:.1f}s   report: {fname}")
    return report


# ════════════════════════════════════════════════════════════════════════
#  Colab / Jupyter arg safety
# ════════════════════════════════════════════════════════════════════════

def _is_jupyter_kernel() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _filter_jupyter_args(argv):
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "-f":
            skip = True
            continue
        if arg.startswith("-f=") or arg.endswith(".json"):
            continue
        out.append(arg)
    return out


def run(**kwargs):
    """Notebook entry — no CLI parsing."""
    ladder = kwargs.pop("ladder", None)
    cfg = SweepConfig(**{k: v for k, v in kwargs.items()
                         if k in SweepConfig.__dataclass_fields__})
    if ladder is not None:
        cfg.ladder = list(ladder)
    return sweep(cfg)


def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv[1:]
    if _is_jupyter_kernel():
        argv = _filter_jupyter_args(argv)

    p = argparse.ArgumentParser()
    p.add_argument("--aleph-hf-version", default=DEFAULT_ALEPH_VERSION)
    p.add_argument("--aleph-repo", default="AbstractPhil/geolip-aleph-void")
    p.add_argument("--dataset", default="byte_trigram")
    p.add_argument("--ladder", type=int, nargs="+", default=None)
    p.add_argument("--stem", choices=["m_hat", "m"], default="m_hat")
    p.add_argument("--lens-sign", choices=["signed", "canon"], default="signed")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--ds-size", type=int, default=200_000)
    p.add_argument("--val-size", type=int, default=4_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--topo-batches", type=int, default=4)
    p.add_argument("--ripser-thresh-deg", type=float, default=20.0)
    p.add_argument("--out-dir", default="./exp_006_results")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--device", default=None)
    args, _unknown = p.parse_known_args(argv)

    cfg = SweepConfig(
        aleph_hf_version=args.aleph_hf_version, aleph_repo=args.aleph_repo,
        dataset=args.dataset, stem=args.stem, lens_sign=args.lens_sign,
        epochs=args.epochs, ds_size=args.ds_size, val_size=args.val_size,
        batch_size=args.batch_size, topo_batches=args.topo_batches,
        ripser_thresh_deg=args.ripser_thresh_deg, out_dir=args.out_dir,
        upload=not args.no_upload, device=args.device,
    )
    if args.ladder is not None:
        cfg.ladder = args.ladder
    return sweep(cfg)


if __name__ == "__main__":
    main()

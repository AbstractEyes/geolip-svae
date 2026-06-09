"""exp_004_aleph_multiscale_lens.py — SELF-CONTAINED multiscale-lens fidelity sweep.

Question (the multiscale aleph-void series, exp 004 of 3): does lifting a FROZEN
geolip-aleph-void battery through the isometric lens to a larger D_lens let a
small spectral shell reconstruct BETTER than the frozen aleph alone, and does
external-recon fidelity rise with D_lens until it saturates?

Design — REUSE core, do not reimplement:
  • The shell, lens, spectral stack and trainer all live in geolip_svae:
        geolip_svae.train_aleph.train_aleph_transformer(...)   -> (model, history)
        geolip_svae.aleph_model.AlephTransformer / SingleLens
    This file only ORCHESTRATES a D_lens ladder and tabulates fidelity.
  • The frozen aleph is hosted, hard-address, byte-trigram. Discover available
    versions with geolip_svae.inference.list_versions(); default to the
    documented 'aleph_byte_trigram_tied_hard_K64'. Do NOT hard-code a version
    name you have not confirmed exists.
  • n_heads MUST divide D_lens (SpectralAlphaAttention asserts it). A fixed
    n_heads=8 crashes the D_lens=4 rung — _valid_n_heads() picks the largest
    divisor <= 8 per rung.

No new latent math: only the documented isometric lens + bounded-alpha stack +
the frozen aleph's closed-form address. No core edits.

Colab:
    from svae_proto.exp_004_aleph_multiscale_lens.run import run
    report = run(ladder=[4, 16, 64, 256], epochs=4,
                 aleph_hf_version='aleph_byte_trigram_tied_hard_K64')
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════════

DEFAULT_LADDER: List[int] = [4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_ALEPH_VERSION = "aleph_byte_trigram_tied_hard_K64"


@dataclass
class SweepConfig:
    # frozen battery
    aleph_hf_version: str = DEFAULT_ALEPH_VERSION
    aleph_repo: str = "AbstractPhil/geolip-aleph-void"
    dataset: str = "byte_trigram"
    # the multiscale knob
    ladder: List[int] = field(default_factory=lambda: list(DEFAULT_LADDER))
    # shell knobs (documented defaults)
    stem: str = "m_hat"           # 'm_hat' (addressed) | 'm' (raw control)
    lens_sign: str = "signed"     # 'signed' | 'canon'
    n_layers: int = 6
    shell_hidden: int = 512
    # training budget (kept small for Colab; the shell converges fast)
    epochs: int = 4
    ds_size: int = 200_000
    val_size: int = 4_000
    batch_size: int = 2048
    # eval set used for the internal-recon floor (shared across rungs)
    eval_batches: int = 10
    # io
    out_dir: str = "./experiments/exp_004_results"
    upload: bool = True
    seed: int = 0
    device: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

def _valid_n_heads(d_lens: int, cap: int = 8) -> int:
    """Largest divisor of d_lens that is <= cap. SpectralAlphaAttention asserts
    d_lens % n_heads == 0, so a fixed n_heads=8 crashes the d_lens=4 rung."""
    for h in range(min(cap, d_lens), 0, -1):
        if d_lens % h == 0:
            return h
    return 1


def _resolve_device(dev: Optional[str]) -> str:
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_eval_loader(cfg: SweepConfig, aleph, device: str):
    """A small fixed byte-trigram test loader, laid out with the SAME patch size
    the frozen aleph was trained on (else the encoder sees a scrambled grid)."""
    from geolip_svae.train_aleph import PRESETS
    from geolip_svae.dataset_presets import get_dataset_bundle

    ev = dict(PRESETS[cfg.dataset])
    ev.update(patch_size=aleph.patch_size, channels=aleph.channels,
              batch_size=cfg.batch_size,
              ds_size=max(cfg.batch_size, cfg.batch_size * cfg.eval_batches),
              val_size=max(cfg.batch_size, cfg.batch_size * cfg.eval_batches))
    bundle = get_dataset_bundle(ev, channels=aleph.channels)
    return bundle.test_loader


@torch.no_grad()
def _internal_mse(aleph, loader, device: str, max_batches: int) -> float:
    """The frozen aleph's own recon MSE — the D_base floor the shell must beat."""
    aleph.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        tot += F.mse_loss(aleph(images)["recon"], images).item()
        n += 1
    return tot / max(n, 1)


def _lens_isometry_err(model) -> float:
    """||EᵀE − I_{D_base}||∞ for the fixed lens buffer. The lift must be exact."""
    E = model.lens.E                       # (D_lens, D_base), orthonormal columns
    g = E.T @ E
    I = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
    return float((g - I).abs().max().item())


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
    print(f"exp_004 multiscale-lens fidelity — {cfg.aleph_repo}/{cfg.aleph_hf_version}")
    print(f"  dataset={cfg.dataset}  stem={cfg.stem}  lens_sign={cfg.lens_sign}  "
          f"device={device}")
    print(f"  ladder={cfg.ladder}")
    print("=" * 72)

    # frozen aleph (once): geometry + the constant internal-recon floor
    aleph0, _ = load_model(hf_version=cfg.aleph_hf_version,
                           repo_id=cfg.aleph_repo, device=device)
    for p in aleph0.parameters():
        p.requires_grad_(False)
    eval_loader = _build_eval_loader(cfg, aleph0, device)
    internal_mse = _internal_mse(aleph0, eval_loader, device, cfg.eval_batches)
    print(f"  frozen aleph: V={aleph0.matrix_v} D_base={aleph0.D} "
          f"ps={aleph0.patch_size} addr={aleph0.address}  internal_mse={internal_mse:.3e}")

    rungs: List[Dict] = []
    t0 = time.time()
    for i, d_lens in enumerate(cfg.ladder):
        if d_lens < aleph0.D:
            print(f"  [skip] D_lens={d_lens} < D_base={aleph0.D} (lens needs lift)")
            continue
        n_heads = _valid_n_heads(d_lens)
        print(f"\n── rung {i+1}/{len(cfg.ladder)}: D_lens={d_lens} "
              f"n_heads={n_heads}x{cfg.n_layers} ──")
        model, history = train_aleph_transformer(
            cfg.aleph_hf_version,
            D_lens=d_lens, dataset=cfg.dataset,
            stem=cfg.stem, lens_sign=cfg.lens_sign,
            n_heads=n_heads, n_layers=cfg.n_layers, shell_hidden=cfg.shell_hidden,
            device=device, upload=cfg.upload,
            aleph_repo=cfg.aleph_repo,
            hf_version=f"exp_004_d{d_lens}_{cfg.stem}_{cfg.lens_sign}",
            save_dir=str(out_dir / f"d{d_lens}_ckpt"),
            tb_dir=str(out_dir / "tb"),
            cfg_overrides=dict(epochs=cfg.epochs, ds_size=cfg.ds_size,
                               val_size=cfg.val_size, batch_size=cfg.batch_size),
        )
        # history rows: (step, train_loss, ext_mse, ext_cos, omega_cv, addr_margin, mean_alpha)
        row = history[-1] if history else (0, 0, float("nan"), float("nan"),
                                           float("nan"), float("nan"), float("nan"))
        rungs.append({
            "D_lens": d_lens, "n_heads": n_heads,
            "external_mse": row[2], "external_cos": row[3],
            "omega_cv": row[4], "address_margin": row[5], "mean_alpha": row[6],
            "internal_mse": internal_mse,
            "beats_internal": (row[2] < internal_mse),
            "lens_isometry_err": _lens_isometry_err(model),
            "shell_params": model.num_params(),
        })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    dt = time.time() - t0
    best = min((r for r in rungs if r["external_cos"] == r["external_cos"]),
               key=lambda r: r["external_mse"], default=None)
    verdict = {
        "internal_mse": internal_mse,
        "any_beats_internal": any(r["beats_internal"] for r in rungs),
        "best_rung": (best["D_lens"] if best else None),
        "best_external_mse": (best["external_mse"] if best else None),
        "best_external_cos": (best["external_cos"] if best else None),
        "all_lens_exact": all(r["lens_isometry_err"] < 1e-4 for r in rungs),
        "all_alpha_in_envelope": all(0.0 <= r["mean_alpha"] <= 0.2 for r in rungs),
    }
    report = {"config": asdict(cfg), "verdict": verdict, "rungs": rungs,
              "elapsed_sec": dt}
    fname = out_dir / f"multiscale_lens_{cfg.aleph_hf_version}_{cfg.stem}_{cfg.lens_sign}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("MULTISCALE-LENS FIDELITY VERDICT")
    print("=" * 72)
    print(f"  internal_mse (D_base floor): {internal_mse:.3e}")
    print(f"  any rung beats internal: {verdict['any_beats_internal']}  "
          f"best D_lens={verdict['best_rung']} "
          f"(ext_mse={verdict['best_external_mse']}, "
          f"ext_cos={verdict['best_external_cos']})")
    print(f"  lens exact (all): {verdict['all_lens_exact']}  "
          f"alpha in [0,0.2] (all): {verdict['all_alpha_in_envelope']}")
    print(f"\n  {'D_lens':>6} {'heads':>5} {'ext_mse':>10} {'ext_cos':>8} "
          f"{'omega_cv':>8} {'alpha':>6} {'beat':>5} {'iso_err':>9} {'params':>9}")
    for r in rungs:
        print(f"  {r['D_lens']:>6} {r['n_heads']:>5} {r['external_mse']:>10.3e} "
              f"{r['external_cos']:>8.4f} {r['omega_cv']:>8.3f} {r['mean_alpha']:>6.3f} "
              f"{str(r['beats_internal']):>5} {r['lens_isometry_err']:>9.1e} "
              f"{r['shell_params']:>9,}")
    print(f"\n  elapsed: {dt:.1f}s   report: {fname}")
    return report


# ════════════════════════════════════════════════════════════════════════
#  Colab / Jupyter arg safety (verbatim from exp_002/exp_003)
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
    """Notebook entry — no CLI parsing.
        from svae_proto.exp_004_aleph_multiscale_lens.run import run
        run(ladder=[4, 16, 64, 256], epochs=4)
    """
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
    p.add_argument("--ladder", type=int, nargs="+", default=None,
                   help="D_lens rungs, e.g. --ladder 4 16 64 256")
    p.add_argument("--stem", choices=["m_hat", "m"], default="m_hat")
    p.add_argument("--lens-sign", choices=["signed", "canon"], default="signed")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--ds-size", type=int, default=200_000)
    p.add_argument("--val-size", type=int, default=4_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", default="./exp_004_results")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--list-versions", action="store_true",
                   help="print available aleph versions and exit")
    args, _unknown = p.parse_known_args(argv)

    if args.list_versions:
        from geolip_svae.inference.loading import list_versions
        list_versions()
        return None

    cfg = SweepConfig(
        aleph_hf_version=args.aleph_hf_version, aleph_repo=args.aleph_repo,
        dataset=args.dataset, stem=args.stem, lens_sign=args.lens_sign,
        epochs=args.epochs, ds_size=args.ds_size, val_size=args.val_size,
        batch_size=args.batch_size, out_dir=args.out_dir,
        upload=not args.no_upload, device=args.device,
    )
    if args.ladder is not None:
        cfg.ladder = args.ladder
    return sweep(cfg)


if __name__ == "__main__":
    main()

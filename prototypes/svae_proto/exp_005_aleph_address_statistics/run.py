"""exp_005_aleph_address_statistics.py — SELF-CONTAINED address/recovery sweep.

Question (multiscale aleph-void series, exp 005 of 3): does the aleph ADDRESS
information survive — and strengthen — through the isometric multiscale lift?
Measured as byte-faithful round-trip recovery through the shell's external_recon,
swept over D_lens and the two documented ablations (stem, lens_sign).

Design — REUSE core, do not reimplement:
  • Shell training: geolip_svae.train_aleph.train_aleph_transformer(...).
  • Byte round-trip primitives: ByteTrigramDataset.bytes_to_image / image_to_bytes.
  • Frozen-aleph baseline recovery: geolip_svae.inference.text.text_recovery_metrics
    (it reads the aleph's OWN recon via engine.reconstruct — the D_base floor).
  • Address health: geolip_svae.train_aleph._address_stats (soft/hard perplexity).

NOVEL piece (why core can't be used as-is): text_recovery_metrics round-trips the
FROZEN aleph's out['recon']; to measure recovery THROUGH the lift we must read the
SHELL's out['external_recon']. _shell_byte_recovery() does exactly that, reusing
the same byte primitives.

No new latent math; no core edits. Colab:
    from svae_proto.exp_005_aleph_address_statistics.run import run
    report = run(ladder=[16, 64, 256], ablate_d_lens=64, epochs=4)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


DEFAULT_LADDER: List[int] = [16, 64, 256]
DEFAULT_ALEPH_VERSION = "aleph_byte_trigram_tied_hard_K64"

# Fixed, deterministic evaluation passage (varied ASCII). Recovery is relative
# across rungs, so the exact text only needs to be stable run-to-run.
DEFAULT_TEXT = (
    "The aleph address snaps each spherical row to a learned projective axis; "
    "the void-rich codebook then carries reconstruction through the lift. "
    "Quick brown foxes jump: 0123456789, (parens) & symbols #@%! survive too."
)


@dataclass
class SweepConfig:
    aleph_hf_version: str = DEFAULT_ALEPH_VERSION
    aleph_repo: str = "AbstractPhil/geolip-aleph-void"
    dataset: str = "byte_trigram"
    ladder: List[int] = field(default_factory=lambda: list(DEFAULT_LADDER))
    ablate_d_lens: int = 64            # single scale for the stem/lens_sign 2x2
    n_layers: int = 6
    shell_hidden: int = 512
    img_size: int = 64
    epochs: int = 4
    ds_size: int = 200_000
    val_size: int = 4_000
    batch_size: int = 2048
    text: str = DEFAULT_TEXT
    out_dir: str = "./experiments/exp_005_results"
    upload: bool = False
    seed: int = 0
    device: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════
#  Helpers (self-contained copies — each experiment is independently deletable)
# ════════════════════════════════════════════════════════════════════════

def _valid_n_heads(d_lens: int, cap: int = 8) -> int:
    for h in range(min(cap, d_lens), 0, -1):
        if d_lens % h == 0:
            return h
    return 1


def _resolve_device(dev: Optional[str]) -> str:
    return dev or ("cuda" if torch.cuda.is_available() else "cpu")


def _prep_image(text: str, img_size: int, ps: int, channels: int
                ) -> Tuple[torch.Tensor, np.ndarray, int]:
    """Encode text → byte-trigram image (space-padded). Returns (img[1,C,H,W],
    padded_chunk uint8, n_real_bytes)."""
    from geolip_svae.dataset_presets import ByteTrigramDataset
    target = img_size * img_size * channels
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    n_real = int(min(len(raw), target))
    chunk = np.full(target, 32, dtype=np.uint8)        # pad with space (32)
    chunk[:n_real] = raw[:n_real]
    img = torch.from_numpy(
        ByteTrigramDataset.bytes_to_image(chunk, img_size, ps, channels))
    return img.unsqueeze(0), chunk, n_real


@torch.no_grad()
def _shell_byte_recovery(model, text: str, img_size: int, ps: int,
                         channels: int, device: str) -> Dict:
    """NOVEL: round-trip bytes through the SHELL's external_recon (not the aleph's
    out['recon']). Mirrors core text_recovery_metrics but reads the lifted recon."""
    from geolip_svae.dataset_presets import ByteTrigramDataset
    img, chunk, n_real = _prep_image(text, img_size, ps, channels)
    model.eval()
    out = model(img.to(device))
    recon = out["external_recon"].cpu()
    recon_bytes = ByteTrigramDataset.image_to_bytes(recon, ps, channels)
    flat = recon_bytes.reshape(-1).numpy().astype(np.uint8)
    orig = chunk.astype(np.uint8)
    if n_real == 0:
        return dict(real_byte_acc=float("nan"), real_byte_l1=float("nan"),
                    recon_text="")
    eq = (flat[:n_real] == orig[:n_real])
    return dict(
        real_byte_acc=float(eq.mean()),
        real_byte_l1=float(np.abs(flat[:n_real].astype(np.int32)
                                  - orig[:n_real].astype(np.int32)).mean()),
        recon_text=bytes(flat[:n_real]).decode("utf-8", errors="replace")[:120],
    )


@torch.no_grad()
def _aleph_baselines(aleph, text: str, img_size: int, ps: int, channels: int,
                     device: str) -> Dict:
    """Fixed D_base baselines: the frozen aleph's own recovery (core
    text_recovery_metrics) + its address health (soft/hard perplexity)."""
    from geolip_svae.inference import InferenceEngine
    from geolip_svae.inference.text import text_recovery_metrics
    out: Dict = {}
    try:
        engine = InferenceEngine(aleph)
        rec = text_recovery_metrics(engine, text, img_size=img_size,
                                    patch_size=ps, channels=channels)
        out["aleph_real_byte_acc"] = rec.get("real_byte_acc")
        out["aleph_recon_mse"] = rec.get("recon_mse")
    except Exception as e:
        out["aleph_recovery_error"] = f"{type(e).__name__}: {e}"
    # perplexity (logits emitted in eval for address != 'none')
    try:
        from geolip_svae.train_aleph import _address_stats
        img, _, _ = _prep_image(text, img_size, ps, channels)
        aleph.eval()
        a_out = aleph(img.to(device))
        logits = a_out["svd"].get("aleph_logits")
        if logits is not None:
            soft_ppl, margin, hard_ppl, _ = _address_stats(
                logits, getattr(aleph, "address_tau", 0.1))
            out.update(soft_perplexity=soft_ppl, hard_perplexity=hard_ppl,
                       address_margin=margin,
                       n_oriented_axes=int(2 * aleph.n_axes))
        else:
            out["perplexity_note"] = "no aleph_logits (address='none'?)"
    except Exception as e:
        out["perplexity_error"] = f"{type(e).__name__}: {e}"
    return out


def _train_one(cfg: SweepConfig, d_lens: int, stem: str, lens_sign: str,
               device: str, out_dir: Path):
    from geolip_svae.train_aleph import train_aleph_transformer
    n_heads = _valid_n_heads(d_lens)
    model, history = train_aleph_transformer(
        cfg.aleph_hf_version,
        D_lens=d_lens, dataset=cfg.dataset, stem=stem, lens_sign=lens_sign,
        n_heads=n_heads, n_layers=cfg.n_layers, shell_hidden=cfg.shell_hidden,
        device=device, upload=cfg.upload, aleph_repo=cfg.aleph_repo,
        hf_version=f"exp_005_d{d_lens}_{stem}_{lens_sign}",
        save_dir=str(out_dir / f"d{d_lens}_{stem}_{lens_sign}_ckpt"),
        tb_dir=str(out_dir / "tb"),
        cfg_overrides=dict(epochs=cfg.epochs, ds_size=cfg.ds_size,
                           val_size=cfg.val_size, batch_size=cfg.batch_size),
    )
    row = history[-1] if history else (0, 0, float("nan"), float("nan"),
                                       float("nan"), float("nan"), float("nan"))
    rec = _shell_byte_recovery(model, cfg.text, cfg.img_size,
                               model.ps, model.channels, device)
    result = {
        "D_lens": d_lens, "n_heads": n_heads, "stem": stem, "lens_sign": lens_sign,
        "external_mse": row[2], "external_cos": row[3],
        "shell_real_byte_acc": rec["real_byte_acc"],
        "shell_real_byte_l1": rec["real_byte_l1"],
        "shell_recon_text": rec["recon_text"],
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# ════════════════════════════════════════════════════════════════════════
#  Sweep
# ════════════════════════════════════════════════════════════════════════

def sweep(cfg: SweepConfig) -> Dict:
    from geolip_svae.inference.loading import load_model

    device = _resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"exp_005 address statistics — {cfg.aleph_repo}/{cfg.aleph_hf_version}")
    print(f"  ladder={cfg.ladder}  ablate@D_lens={cfg.ablate_d_lens}  device={device}")
    print("=" * 72)

    aleph0, _ = load_model(hf_version=cfg.aleph_hf_version,
                           repo_id=cfg.aleph_repo, device=device)
    for p in aleph0.parameters():
        p.requires_grad_(False)
    baselines = _aleph_baselines(aleph0, cfg.text, cfg.img_size,
                                 aleph0.patch_size, aleph0.channels, device)
    print(f"  frozen aleph baseline: {baselines}")

    t0 = time.time()
    # primary arm (m_hat, signed) over the full ladder
    primary: List[Dict] = []
    for i, d_lens in enumerate(cfg.ladder):
        if d_lens < aleph0.D:
            print(f"  [skip] D_lens={d_lens} < D_base={aleph0.D}")
            continue
        print(f"\n── primary rung {i+1}/{len(cfg.ladder)}: D_lens={d_lens} (m_hat, signed) ──")
        primary.append(_train_one(cfg, d_lens, "m_hat", "signed", device, out_dir))

    # ablation 2x2 at a single scale (skip the (m_hat,signed) corner already in primary)
    ablation: List[Dict] = []
    for stem in ("m_hat", "m"):
        for lens_sign in ("signed", "canon"):
            if stem == "m_hat" and lens_sign == "signed" and \
                    any(r["D_lens"] == cfg.ablate_d_lens for r in primary):
                # reuse the primary rung at this scale
                match = next(r for r in primary if r["D_lens"] == cfg.ablate_d_lens)
                ablation.append(dict(match, reused_from_primary=True))
                continue
            print(f"\n── ablation: D_lens={cfg.ablate_d_lens} ({stem}, {lens_sign}) ──")
            ablation.append(_train_one(cfg, cfg.ablate_d_lens, stem, lens_sign,
                                       device, out_dir))

    dt = time.time() - t0

    # verdict: documented ablation directions + recovery trend
    def _acc(stem, sign):
        for r in ablation:
            if r["stem"] == stem and r["lens_sign"] == sign:
                return r["shell_real_byte_acc"]
        return float("nan")
    a_ms, a_mc = _acc("m_hat", "signed"), _acc("m_hat", "canon")
    a_rs = _acc("m", "signed")
    accs = [r["shell_real_byte_acc"] for r in primary
            if r["shell_real_byte_acc"] == r["shell_real_byte_acc"]]
    verdict = {
        "signed_beats_canon": (a_ms >= a_mc) if (a_ms == a_ms and a_mc == a_mc) else None,
        "m_hat_beats_m": (a_ms >= a_rs) if (a_ms == a_ms and a_rs == a_rs) else None,
        "primary_recovery_monotone_nondec": (accs == sorted(accs)) if len(accs) > 1 else None,
        "best_primary_acc": (max(accs) if accs else None),
        "aleph_baseline_acc": baselines.get("aleph_real_byte_acc"),
    }
    report = {"config": asdict(cfg), "baselines": baselines, "verdict": verdict,
              "primary": primary, "ablation": ablation, "elapsed_sec": dt}
    fname = out_dir / f"address_statistics_{cfg.aleph_hf_version}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("ADDRESS-STATISTICS VERDICT")
    print("=" * 72)
    print(f"  signed>=canon: {verdict['signed_beats_canon']}   "
          f"m_hat>=m: {verdict['m_hat_beats_m']}   "
          f"primary recovery non-decreasing: {verdict['primary_recovery_monotone_nondec']}")
    print(f"  aleph baseline byte_acc={verdict['aleph_baseline_acc']}  "
          f"best shell byte_acc={verdict['best_primary_acc']}")
    print(f"\n  primary arm (m_hat, signed):")
    print(f"  {'D_lens':>6} {'ext_mse':>10} {'ext_cos':>8} {'byte_acc':>9} {'byte_l1':>8}")
    for r in primary:
        print(f"  {r['D_lens']:>6} {r['external_mse']:>10.3e} {r['external_cos']:>8.4f} "
              f"{r['shell_real_byte_acc']:>9.4f} {r['shell_real_byte_l1']:>8.3f}")
    print(f"\n  ablation @ D_lens={cfg.ablate_d_lens}:")
    print(f"  {'stem':>6} {'lens_sign':>9} {'byte_acc':>9} {'ext_cos':>8}")
    for r in ablation:
        print(f"  {r['stem']:>6} {r['lens_sign']:>9} {r['shell_real_byte_acc']:>9.4f} "
              f"{r['external_cos']:>8.4f}")
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
    p.add_argument("--ablate-d-lens", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--ds-size", type=int, default=200_000)
    p.add_argument("--val-size", type=int, default=4_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", default="./exp_005_results")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--device", default=None)
    args, _unknown = p.parse_known_args(argv)

    cfg = SweepConfig(
        aleph_hf_version=args.aleph_hf_version, aleph_repo=args.aleph_repo,
        dataset=args.dataset, ablate_d_lens=args.ablate_d_lens,
        epochs=args.epochs, ds_size=args.ds_size, val_size=args.val_size,
        batch_size=args.batch_size, out_dir=args.out_dir,
        upload=not args.no_upload, device=args.device,
    )
    if args.ladder is not None:
        cfg.ladder = args.ladder
    return sweep(cfg)


if __name__ == "__main__":
    main()

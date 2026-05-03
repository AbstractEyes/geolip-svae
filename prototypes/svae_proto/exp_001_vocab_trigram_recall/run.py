"""
svae_proto.exp_001_vocab_trigram_recall.run
========================================
Thin launcher for experiment 001.

What it does:
    1. Imports VocabTrigramDataset's factory and registers it under the
       name 'vocab_trigram' in the geolip_svae DATASET_FACTORIES registry
       AT RUNTIME ONLY (no source files modified).
    2. Picks a cfg variant from cfg.py (override via --variant).
    3. Calls geolip_svae.train.train(cfg).
    4. Runs the token-level recovery eval (eval.run_vocab_eval) against
       the trained model and writes a JSON report alongside the
       checkpoints.

Usage
-----
    python -m svae_proto.exp_001_vocab_trigram_recall.run --variant proto_64
    python -m svae_proto.exp_001_vocab_trigram_recall.run --variant freckles_64
    python -m svae_proto.exp_001_vocab_trigram_recall.run --variant fresnel_128

Override epochs / save dir / etc. via standard cfg keys passed through
``--cfg-override key=value``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


_VARIANTS = {
    'proto_64':     'CFG_PROTO_64',
    'freckles_64':  'CFG_FRECKLES_64',
    'fresnel_128':  'CFG_FRESNEL_128',
}


def _parse_overrides(overrides):
    """Parse `--cfg-override key=value` pairs. Best-effort literal eval."""
    import ast
    out: Dict[str, Any] = {}
    for kv in overrides or []:
        if '=' not in kv:
            raise SystemExit(f"--cfg-override expects key=value; got {kv!r}")
        k, v = kv.split('=', 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v   # leave as plain string
    return out


def _register_dataset() -> None:
    """Register VocabTrigramDataset under 'vocab_trigram' for this process."""
    from geolip_svae.dataset_presets import DATASET_FACTORIES
    from .dataset import vocab_trigram_factory
    DATASET_FACTORIES['vocab_trigram'] = vocab_trigram_factory
    print(f"  [proto001] registered dataset 'vocab_trigram' "
          f"({len(DATASET_FACTORIES)} total factories now)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Experiment 001 — vocabulary trigram recall')
    parser.add_argument('--variant', choices=list(_VARIANTS),
                        default='proto_64',
                        help='Which cfg variant from cfg.py to run.')
    parser.add_argument('--cfg-override', action='append', default=[],
                        help='Override cfg key. Repeat. Example: '
                             '--cfg-override epochs=5 --cfg-override batch_size=32')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training; only run the post-train eval '
                             '(model loaded from save_dir/best.pt, falling '
                             'back to the latest epoch_*.pt if best is missing).')
    parser.add_argument('--eval-samples', type=int, default=512,
                        help='Number of test-split samples for the token-level eval.')
    parser.add_argument('--hf-token', default=None,
                        help='HuggingFace token. Sets HF_TOKEN env var before '
                             'the trainer imports huggingface_hub. On Colab, '
                             'pass `userdata.get("HF_TOKEN")` from a cell '
                             'before `!python -m ...` (see README).')
    args = parser.parse_args(argv)

    # Set HF_TOKEN BEFORE importing the trainer — its module-level auth
    # block runs at first import, and HfApi() picks up HF_TOKEN from env.
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
        print(f"  [proto001] HF_TOKEN set from --hf-token "
              f"({len(args.hf_token)} chars)")

    # 1. Register the dataset (transient, in-process only)
    _register_dataset()

    # 2. Pick + override cfg
    from . import cfg as cfg_mod
    cfg: Dict[str, Any] = dict(getattr(cfg_mod, _VARIANTS[args.variant]))
    cfg.update(_parse_overrides(args.cfg_override))
    print(f"  [proto001] variant={args.variant}, "
          f"epochs={cfg['epochs']}, batch={cfg['batch_size']}, "
          f"img_size={cfg['img_size']}, V={cfg['V']}, D={cfg['D']}")

    # 3. Train (unless skipping)
    if not args.skip_train:
        from geolip_svae.train import train
        train(cfg)

    # 4. Token-level recovery eval on the test split
    print(f"\n  [proto001] running token-level recovery eval...")
    import torch
    from geolip_svae.model import PatchSVAE
    from .dataset import VocabTrigramDataset
    from .eval import run_vocab_eval

    # Trainer writes checkpoints DIRECTLY into save_dir (not save_dir/checkpoints/).
    # See geolip_svae/train.py:821 (best.pt) and :825 (epoch_NNNN.pt).
    save_dir = cfg.get('save_dir', '/content/checkpoints')
    ckpt_path = os.path.join(save_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        # Fallback: latest epoch_*.pt in save_dir
        candidates = sorted(
            (f for f in os.listdir(save_dir)
             if f.startswith('epoch_') and f.endswith('.pt')),
            reverse=True,
        ) if os.path.isdir(save_dir) else []
        if not candidates:
            print(f"  [proto001] no checkpoint at {ckpt_path} or epoch_*.pt "
                  f"under {save_dir}, skipping eval.")
            return
        ckpt_path = os.path.join(save_dir, candidates[0])
    print(f"  [proto001] loading {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_cfg = ckpt.get('config', cfg)

    # Build a model matching the saved config. Use the cfg-driven kwargs
    # so this works for any of the three variants.
    model = PatchSVAE(
        V=model_cfg.get('V', cfg['V']),
        D=model_cfg.get('D', cfg['D']),
        ps=model_cfg.get('patch_size', cfg['patch_size']),
        hidden=model_cfg.get('hidden', cfg['hidden']),
        depth=model_cfg.get('depth', cfg['depth']),
        n_cross=model_cfg.get('n_cross_layers', cfg['n_cross']),
        channels=model_cfg.get('channels', cfg.get('channels', 3)),
        linear_readout=model_cfg.get('linear_readout', cfg.get('linear_readout', False)),
        svd_mode=model_cfg.get('svd_mode', cfg.get('svd_mode', 'default')),
        svd_method=model_cfg.get('svd_method', cfg.get('svd_method', 'auto')),
        svd_compute_dtype=model_cfg.get('svd_compute_dtype',
                                          cfg.get('svd_compute_dtype', 'fp64')),
        match_params=model_cfg.get('match_params', cfg.get('match_params', True)),
        smooth_mid=model_cfg.get('smooth_mid', cfg.get('smooth_mid')),
        n_heads=model_cfg.get('n_heads', cfg.get('n_heads')),
    )
    model.load_state_dict(ckpt.get('model_state_dict') or ckpt['model_state'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    test_ds = VocabTrigramDataset(
        corpus=cfg.get('vt_corpus', 'wikitext-2-raw-v1'),
        tokenizer=cfg.get('vt_tokenizer', 'google-t5/t5-base'),
        img_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
        channels=cfg.get('channels', 3),
        n_samples=args.eval_samples,
        seed=cfg.get('vt_seed', 0) + 9999,
        max_corpus_chars=cfg.get('vt_max_corpus_chars', 4_000_000),
        split=cfg.get('vt_test_split', 'test'),
    )

    report = run_vocab_eval(
        model, test_ds, device=device,
        n_samples=args.eval_samples,
        batch_size=min(32, cfg['batch_size']),
        notes=f"variant={args.variant}, ckpt={os.path.basename(ckpt_path)}",
    )
    print()
    print(report.summary())

    out_path = os.path.join(save_dir, f'vocab_eval_{args.variant}.json')
    report.save(out_path)
    print(f"\n  [proto001] report written to {out_path}")


if __name__ == '__main__':
    main()

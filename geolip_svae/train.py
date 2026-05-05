"""
SVAE Unified Trainer
=====================
Single entry point for every PatchSVAE variant in the package.

CLI
---
    python -m geolip_svae.train --preset NAME
    python -m geolip_svae.train --list-presets
    python -m geolip_svae.train --preset NAME --epochs 50 --no-upload

Common preset families (full catalog: ``--list-presets``):

    Fresnel    (images, V=256 D=16)         python -m geolip_svae.train --preset fresnel_base
    Johanna    (noise,  V=256 D=16)         python -m geolip_svae.train --preset johanna_base
    Alexandria (text,   V=256 D=16)         python -m geolip_svae.train --preset alexandria_small
    Freckles   (noise,  V=48  D=4)          python -m geolip_svae.train --preset freckles_64
    Fresnel-64 (images, V=48  D=4)          python -m geolip_svae.train --preset fresnel_64
    H2-64      (sphere, V=32  D=4)          python -m geolip_svae.train --preset h2_64_single
    BinTree    (substrate, h2-64 arch)      python -m geolip_svae.train --preset bintree_proto
    SP-bits    (substrate, h2-64 arch)      python -m geolip_svae.train --preset sentencepiece_proto
    ByteTri    (substrate, h2-64 arch)      python -m geolip_svae.train --preset byte_trigram_proto

For long-running continuation of any trained model on streaming random crops
(the "sublens perspective" mode that produced v50_fresnel_64's 140M+ images),
see ``geolip_svae.train_streaming``.

Module layout
-------------
The trainer is split across three sibling modules. Each owns a single
responsibility so the loop stays narrow:

    train.py             — the training loop, CLI, optimizer/scheduler,
                           pretrained loading, HF upload helpers, codebook
                           build hook. Reads PRESETS / dataset factories.
    train_presets.py     — PRESETS registry + a fully-documented TEMPLATE
                           dict listing every cfg key the trainer accepts
                           with its default and accepted values.
    dataset_presets.py   — DATASET_FACTORIES registry, all dataset classes,
                           recovery metrics, eval helper. Exposes
                           get_dataset_bundle(cfg, channels) → DatasetBundle.

Key model-side modules consumed:

    geolip_svae.model              — PatchSVAE, gram_eigh_svd dispatcher,
                                     ACTIVATIONS / ACTIVATION_MODULES /
                                     ACTIVATION_SITES, SVD_METHODS.
    geolip_svae.inference          — engine, codebook, calibration,
                                     train_codebook (post-train hook).

Configuring a run
-----------------
A cfg dict is just a Python ``dict``. Required keys (V, D, patch_size,
hidden, depth, n_cross, dataset, img_size, batch_size, lr, epochs,
target_cv, hf_version) raise KeyError if missing; everything else is
optional and falls through to the documented default. See
``geolip_svae.train_presets.TEMPLATE`` for the canonical, fully-specified
cfg dict — copy it to start a new preset.

Architecture features that flow through cfg
-------------------------------------------
  * F/G/H/L group ablation toggles (activation, activations, row_norm,
    svd_mode, linear_readout, match_params, init_scheme).
  * SVD dispatcher (svd_method, svd_compute_dtype) — auto-routes through
    geolip-core's batched_svd, which fires the fused Triton N=4 kernel
    at D=4 on CUDA. Drops the historical need for the linear_readout
    workaround at small D.
  * Per-site activations — five named slots (enc_in, enc_block_inner,
    dec_in, dec_block_inner, boundary_smooth) each pickable from a
    21-entry registry. Defaults preserve pre-refactor GELU behavior
    bit-for-bit.
  * Channels — non-3-channel datasets (1-channel, 5-channel) by setting
    cfg['channels'].

Reporting
---------
  * johanna_F diagnostics: epoch_max_grad, per-layer alpha mean/std,
    cv_in_band boolean, full per-step ``history`` dumped to final_report.json.
  * Mid-epoch reporting cadence via ``report_every`` (in batches).
  * allowed_types filter for noise datasets — Gaussian-only foundation,
    custom subset, or all 16.
  * Recovery metrics for substrate datasets (binary tree bits,
    sentencepiece bits/tokens, byte-trigram).

Codebook hook
-------------
At end-of-train, ``create_codebook(model, cfg, ...)`` is invoked
(``cfg['build_codebook']=False`` to opt out). This extracts the final
projective-axis Codebook artifact and runs the kNN-graph / local-PCA /
optional ripser persistent-homology probes. Output lands under
``save_dir/codebooks/`` and uploads to HF when enabled.
"""

import os
import math
import json
import time
import argparse
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from geolip_svae.model import PatchSVAE, cv_of

# ── HuggingFace auth ─────────────────────────────────────────────────
#
# Three sources, in priority order:
#   1. HF_TOKEN already set in os.environ (e.g. by run.py --hf-token, or
#      by the user setting it explicitly in the cell before invoking).
#   2. google.colab.userdata.get('HF_TOKEN') — the Colab-secret path.
#      Only fires when running in Colab AND the secret has been granted
#      access to the current notebook.
#   3. None — auth disabled, HF upload skipped, public reads still work.

if not os.environ.get('HF_TOKEN'):
    try:
        from google.colab import userdata
        _tok = userdata.get('HF_TOKEN')
        if _tok:                                 # None when not authorized
            os.environ['HF_TOKEN'] = _tok
    except Exception:
        pass

if os.environ.get('HF_TOKEN'):
    try:
        from huggingface_hub import login
        login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    except Exception as _e:
        print(f"  [hf-auth] login skipped: {type(_e).__name__}: {_e}")


# ═══════════════════════════════════════════════════════════════════
# PRESETS — owned by geolip_svae.train_presets
# ═══════════════════════════════════════════════════════════════════
# The PRESETS catalog and the TEMPLATE dict (a fully-documented cfg
# skeleton) live in ``train_presets.py``. That module is intentionally
# torch-free, so non-trainer code can import the catalog without paying
# for torch / torchvision / huggingface_hub. Authoring a new preset
# does NOT require editing this file — add an entry to PRESETS in
# train_presets.py and the trainer picks it up automatically.
#
# Re-exported here so legacy ``from geolip_svae.train import PRESETS``
# call sites keep working.
from geolip_svae.train_presets import PRESETS, TEMPLATE  # noqa: F401


# ═══════════════════════════════════════════════════════════════════
# DATASETS — owned by geolip_svae.dataset_presets
# ═══════════════════════════════════════════════════════════════════
# All dataset classes, noise machinery, recovery metrics, the per-type
# eval helper, and the DATASET_FACTORIES registry live in
# ``dataset_presets.py``. The trainer's only contact with dataset
# code is::
#
#     bundle = get_dataset_bundle(cfg, channels=channels)
#
# which returns a DatasetBundle with train/test loaders + flag booleans
# (is_noise / is_text / is_image / is_tree / is_sentencepiece /
# is_byte_trigram) describing what kind of data is flowing through.
#
# Re-exports below cover legacy ``from geolip_svae.train import
# ByteTrigramDataset`` style call sites used by tests and notebooks.
from geolip_svae.dataset_presets import (
    NOISE_NAMES, TIERS,
    _pink_noise, _brown_noise, _generate_noise,
    CurriculumNoiseDataset, OmegaNoiseDataset,
    HFImageDataset, get_image_loaders,
    WikiTextAsImage,
    BinaryTreeDataset, decode_image_to_trees, bit_recovery_metrics,
    SentencePieceBitDataset, decode_image_to_tokens,
    token_bit_recovery_metrics,
    ByteTrigramDataset, byte_recovery_metrics,
    eval_per_type,
    get_dataset_bundle, DatasetBundle,
)



# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train(cfg: Dict[str, Any]):
    """Main training loop. cfg is a preset dict or custom config."""

    # ── Architecture kwargs (what v1 was missing) ──
    V              = cfg['V']
    D              = cfg['D']
    patch_size     = cfg['patch_size']
    hidden         = cfg['hidden']
    depth          = cfg['depth']
    n_cross        = cfg['n_cross']
    n_heads        = cfg.get('n_heads', None)
    smooth_mid     = cfg.get('smooth_mid', None)
    linear_readout = cfg.get('linear_readout', False)
    linear_readout_power = cfg.get('linear_readout_power', 2.0)
    svd_mode       = cfg.get('svd_mode', 'default')
    match_params   = cfg.get('match_params', True)
    channels       = cfg.get('channels', 3)

    # ── Architecture kwargs added with geolip-core 0.3.0 / activation
    #    registry refactor. All defaults preserve pre-refactor behavior;
    #    presets only need to specify these when overriding.
    solver            = cfg.get('solver', 'default')
    activation        = cfg.get('activation', 'gelu')
    activations       = cfg.get('activations', None)
    row_norm          = cfg.get('row_norm', 'sphere')
    svd_method        = cfg.get('svd_method', 'auto')
    svd_compute_dtype = cfg.get('svd_compute_dtype', 'fp64')
    init_scheme       = cfg.get('init_scheme', 'orthogonal')

    # ── Training ──
    dataset       = cfg['dataset']
    img_size      = cfg['img_size']
    batch_size    = cfg['batch_size']
    lr            = cfg['lr']
    epochs        = cfg['epochs']
    target_cv     = cfg['target_cv']
    hf_version    = cfg['hf_version']
    save_every    = cfg.get('save_every', 10)
    report_every  = cfg.get('report_every', 500)

    # ── Loss ──
    cv_weight     = cfg.get('cv_weight', 0.3)
    boost         = cfg.get('boost', 0.5)
    sigma         = cfg.get('sigma', 0.15)

    # ── CV band thresholds (for the cv_in_band boolean diagnostic) ──
    # Defaults to the V=256/D=16 noise-substrate band (0.13-0.30) per the
    # CM CV deep embedding analysis framework. Override for h2-class (V=32/
    # D=4) which lives in a different basin — natural band is roughly
    # 0.85-1.05 per measured runs (h2 single-Gaussian: 0.88-0.92, bintree:
    # 0.80-1.01). Set both to None to disable the band check.
    cv_band_lo    = cfg.get('cv_band_lo', 0.13)
    cv_band_hi    = cfg.get('cv_band_hi', 0.30)

    # ── Data filters / curriculum ──
    pretrained    = cfg.get('pretrained', None)
    curriculum    = cfg.get('curriculum', None)
    tier_schedule = cfg.get('tier_schedule', None)
    allowed_types = cfg.get('allowed_types', None)

    # ── Tree config ──
    tree_depth    = cfg.get('tree_depth', 4)

    # ── Output paths ──
    save_dir      = cfg.get('save_dir', '/content/checkpoints')
    hf_repo       = cfg.get('hf_repo', 'AbstractPhil/geolip-SVAE')
    tb_dir        = cfg.get('tb_dir', '/content/runs')
    upload        = cfg.get('upload', True)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── TensorBoard ──
    from torch.utils.tensorboard import SummaryWriter
    run_name = f"{hf_version}_{img_size}x{img_size}_h{hidden}_d{depth}_lr{lr}"
    tb_path = os.path.join(tb_dir, run_name)
    writer = SummaryWriter(tb_path)
    print(f"  TensorBoard: {tb_path}")

    # ── HuggingFace ──
    hf_enabled = False
    api = None
    if upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.whoami()
            hf_enabled = True
            hf_prefix = f"{hf_version}/checkpoints"
            print(f"  HuggingFace: {hf_repo}/{hf_prefix}")
        except Exception as e:
            print(f"  HuggingFace: disabled ({e})")

    def upload_to_hf(local_path, remote_name, prefix=None):
        if not hf_enabled:
            return
        prefix = prefix if prefix is not None else hf_prefix
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{prefix}/{remote_name}",
                repo_id=hf_repo, repo_type="model")
            print(f"  ☁️  Uploaded: {hf_repo}/{prefix}/{remote_name}")
        except Exception as e:
            print(f"  ⚠️  HF upload: {e}")

    # ── Model ──
    model_kwargs = dict(
        V=V, D=D, ps=patch_size, hidden=hidden, depth=depth, n_cross=n_cross,
        channels=channels,
        solver=solver,
        activation=activation,
        activations=activations,
        row_norm=row_norm,
        svd_mode=svd_mode,
        svd_method=svd_method,
        svd_compute_dtype=svd_compute_dtype,
        linear_readout=linear_readout,
        match_params=match_params,
        init_scheme=init_scheme,
        linear_readout_power=linear_readout_power,
    )
    if n_heads is not None:
        model_kwargs['n_heads'] = n_heads
    if smooth_mid is not None:
        model_kwargs['smooth_mid'] = smooth_mid
    model = PatchSVAE(**model_kwargs).to(device)

    # ── Pretrained weights ──
    if pretrained:
        from huggingface_hub import hf_hub_download
        print(f"\n  Loading pretrained: {pretrained}")
        try:
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=pretrained,
                                         repo_type='model')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            print(f"  Loaded ep{ckpt['epoch']}, MSE={ckpt['test_mse']:.6f}")
        except Exception as e:
            print(f"  ⚠️  Pretrained load failed: {e} — training from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Data ──
    bundle = get_dataset_bundle(cfg, channels=channels)
    train_loader     = bundle.train_loader
    test_loader      = bundle.test_loader
    is_noise         = bundle.is_noise
    is_text          = bundle.is_text
    is_image         = bundle.is_image
    is_tree          = bundle.is_tree
    is_sentencepiece = bundle.is_sentencepiece
    is_byte_trigram  = bundle.is_byte_trigram

    # ── Print config ──
    n_patches = (img_size // patch_size) ** 2
    arch_tags = []
    if linear_readout:
        arch_tags.append('linear_readout')
    if svd_mode != 'default':
        arch_tags.append(f"svd={svd_mode}")
    arch_str = f" [{'+'.join(arch_tags)}]" if arch_tags else ""

    print(f"\n  SVAE TRAINER (v2) — {hf_version}{arch_str}")
    print(f"  {img_size}×{img_size}, {n_patches} patches, V={V}, D={D}, "
          f"{total_params:,} params")
    print(f"  Dataset: {dataset}, batch={batch_size}, lr={lr}, epochs={epochs}")
    print(f"  Target CV: {target_cv}, soft hand: boost={1+boost:.1f}x, "
          f"penalty={cv_weight}, band=[{cv_band_lo:.2f}, {cv_band_hi:.2f}]")
    if allowed_types is not None:
        print(f"  Allowed types: {allowed_types}")
    if curriculum:
        print(f"  Curriculum: {curriculum}")
    if tier_schedule:
        print(f"  Tier schedule: {tier_schedule}")
    if is_tree:
        _tds = train_loader.dataset
        print(f"  Tree depth: {tree_depth}, "
              f"n_nodes: {_tds.n_nodes}, n_pad: {_tds.n_pad}")
    if is_sentencepiece:
        _tds = train_loader.dataset
        print(f"  SP tokenizer: {cfg.get('sp_tokenizer', 'google-t5/t5-base')}, "
              f"corpus: {cfg.get('sp_corpus', 'wikitext-2-raw-v1')}, "
              f"n_bits: {cfg.get('sp_n_bits', 16)}, "
              f"vocab: {_tds.vocab_size}, tokens/img: {_tds.n_patches}")

    # ── Device + SVD pathway introspection ───────────────────────────
    # Reports the actual hardware in use, the geolip-core backend state,
    # and which SVD path the current cfg will engage at every forward.
    # Useful when verifying that an --preset really hits the Triton kernel
    # or when triaging why an h2-class run is slower than expected.
    print("  " + "─" * 96)
    if device.type == 'cuda':
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        props = torch.cuda.get_device_properties(idx)
        gib = props.total_memory / (1024 ** 3)
        print(f"  Device:       cuda:{idx} — {props.name} (sm_{cap[0]}{cap[1]}, "
              f"{props.multi_processor_count} SMs, {gib:.1f} GiB)")
        print(f"  Torch:        {torch.__version__}  "
              f"cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
    else:
        print(f"  Device:       cpu")
        print(f"  Torch:        {torch.__version__}  (CUDA unavailable)")

    # geolip-core backend state — covers triton + FL eigh availability
    try:
        from geolip_core.linalg._backend import backend as _gc_backend
        triton_str = (f"v{_gc_backend.triton_version}" if _gc_backend.has_triton
                      else "not installed")
        print(f"  geolip-core:  triton={triton_str}  "
              f"use_triton={_gc_backend.use_triton}  "
              f"use_fl_eigh={_gc_backend.use_fl_eigh}")
    except Exception as _e:
        _gc_backend = None
        print(f"  geolip-core:  backend introspection failed ({_e})")

    # Predict the SVD pathway the model will actually invoke.
    # Mirrors the dispatch logic in:
    #   - PatchSVAE.encode_patches (linear_readout / svd_mode branches)
    #   - PatchSVAE._svd (solver='conduit' shortcut)
    #   - geolip_core.linalg.batched_svd auto-dispatch (method='auto')
    _on_cuda = (device.type == 'cuda')
    _use_triton = bool(_gc_backend and _gc_backend.use_triton and _on_cuda)
    _use_fl    = bool(_gc_backend and _gc_backend.use_fl_eigh and _on_cuda)

    if linear_readout:
        path = (f"linear-readout (sphere-solver) — SVD bypassed; "
                f"learned nn.Linear(V*D, V*D) replaces U·S·Vt"
                f" — power={linear_readout_power}")
    elif svd_mode == 'fp32':
        path = "encode_patches fp32 ablation — torch.linalg.eigh @ fp32 (no autocast)"
    elif svd_mode == 'fp64':
        path = "encode_patches fp64 ablation — torch.linalg.eigh @ fp64 (no autocast)"
    elif svd_mode == 'batch_shared':
        path = (f"batch-shared SVD — single SVD per batch via dispatcher "
                f"(method={svd_method!r})")
    elif solver == 'conduit':
        path = ("FLEighConduit @ fp64 — telemetry path (svd_method ignored, "
                "ConduitPacket captured per forward)")
    else:
        # solver='default' + svd_mode='default' → batched_svd dispatcher
        if svd_method == 'torch':
            path = f"torch.linalg.svd @ {svd_compute_dtype}"
        elif svd_method == 'fl' and _use_fl and svd_compute_dtype == 'fp32':
            path = "FL eigh + Gram @ fp32 (forced)"
        elif svd_method == 'gram_eigh':
            path = f"torch.linalg.eigh + Gram @ {svd_compute_dtype} (forced)"
        elif svd_method == 'triton' and _use_triton and 2 <= D <= 6:
            path = (f"fused Triton N={D} kernel @ {svd_compute_dtype} (forced) "
                    f"— BLOCK_M=128, JACOBI_ITERS={'12' if D==6 else '6'}")
        elif svd_method == 'auto':
            # Auto-dispatch logic from geolip_core.linalg.batched_svd
            if 2 <= D <= 6 and _use_triton:
                path = (f"fused Triton N={D} kernel @ {svd_compute_dtype} "
                        f"— BLOCK_M=128, JACOBI_ITERS={'12' if D==6 else '6'}")
            elif D <= 12 and _use_fl and svd_compute_dtype == 'fp32':
                path = f"FL eigh + Gram @ fp32 (auto)"
            elif _on_cuda:
                path = f"torch.linalg.eigh + Gram @ {svd_compute_dtype} (auto, cuda)"
            else:
                path = f"torch.linalg.svd @ {svd_compute_dtype} (auto, cpu fallback)"
        else:
            # Method/compat mismatch — dispatcher will fall through; flag it.
            path = (f"forced method={svd_method!r} but conditions unmet "
                    f"— dispatcher will fall through to torch fallback")

    print(f"  SVD path:     {path}")
    print(f"  SVD config:   solver={solver!r}, svd_mode={svd_mode!r}, "
          f"svd_method={svd_method!r}, compute_dtype={svd_compute_dtype!r}")
    # Activations: only show non-default sites to keep output narrow.
    _act_changes = [
        f"{site}={name!r}" for site, name in model.activations.items()
        if name != 'gelu'
    ]
    if _act_changes:
        print(f"  Activations:  {', '.join(_act_changes)} (others: gelu)")
    else:
        print(f"  Activations:  all sites = 'gelu' (default)")
    print(f"  Row-norm:     {row_norm!r}   Init scheme: {init_scheme!r}")
    print("=" * 100)

    # ── Helpers ──
    best_recon = float('inf')
    history: List[Dict[str, Any]] = []

    def save_checkpoint(path, epoch_, test_mse_, extra=None, do_upload=True):
        ckpt_out = {
            'epoch': epoch_, 'test_mse': test_mse_,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'config': {
                'V': V, 'D': D, 'patch_size': patch_size,
                'hidden': hidden, 'depth': depth, 'n_cross_layers': n_cross,
                'n_heads': n_heads, 'smooth_mid': smooth_mid,
                'channels': channels,
                'solver': solver,
                'activation': activation,
                'activations': activations,
                'row_norm': row_norm,
                'svd_mode': svd_mode,
                'svd_method': svd_method,
                'svd_compute_dtype': svd_compute_dtype,
                'linear_readout': linear_readout,
                'linear_readout_power': linear_readout_power,
                'match_params': match_params,
                'init_scheme': init_scheme,
                'target_cv': target_cv, 'dataset': dataset,
                'img_size': img_size, 'lr': lr,
                # Pass through SP-specific kwargs so a checkpoint can be
                # rehydrated for evaluation without the original preset.
                'sp_tokenizer': cfg.get('sp_tokenizer'),
                'sp_corpus': cfg.get('sp_corpus'),
                'sp_n_bits': cfg.get('sp_n_bits'),
                'tree_depth': cfg.get('tree_depth'),
            },
        }
        if extra:
            ckpt_out.update(extra)
        torch.save(ckpt_out, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  💾 Saved: {path} ({size_mb:.1f}MB, ep{epoch_}, "
              f"MSE={test_mse_:.6f})")
        if do_upload:
            upload_to_hf(path, os.path.basename(path))

    def per_layer_alphas():
        """Return (alpha_mean, alpha_std) averaged across cross-attn layers."""
        if n_cross <= 0 or len(model.cross_attn) == 0:
            return 0.0, 0.0
        alphas = [layer.alpha.detach() for layer in model.cross_attn]
        a_mean = torch.stack([a.mean() for a in alphas]).mean().item()
        a_std = torch.stack([a.std() for a in alphas]).mean().item()
        return a_mean, a_std

    # ── Patience promotion state (for curriculum) ──
    tier_best_mse = float('inf')
    stale_epochs = 0

    # ── Training ──
    last_cv = target_cv
    last_prox = 1.0
    global_batch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, n_seen = 0.0, 0.0, 0
        epoch_max_grad = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}",
                    bar_format='{l_bar}{bar:20}{r_bar}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(images)
            recon_loss = F.mse_loss(out['recon'], images)

            # Soft-hand proximity (measure CV every 50 batches)
            with torch.no_grad():
                if batch_idx % 50 == 0:
                    current_cv = cv_of(out['svd']['M'][0, 0])
                    if current_cv > 0:
                        last_cv = current_cv
                    delta = last_cv - target_cv
                    last_prox = math.exp(-delta ** 2 / (2 * sigma ** 2))

            recon_w = 1.0 + boost * last_prox
            cv_pen = cv_weight * (1.0 - last_prox)
            loss = recon_w * recon_loss + cv_pen * (last_cv - target_cv) ** 2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.cross_attn.parameters(), max_norm=0.5
            )

            # Track total grad norm for stability
            total_grad = sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            epoch_max_grad = max(epoch_max_grad, total_grad)

            opt.step()

            total_loss += loss.item() * len(images)
            total_recon += recon_loss.item() * len(images)
            n_seen += len(images)
            global_batch += 1
            pbar.set_postfix_str(f"mse={recon_loss.item():.4f} cv={last_cv:.3f}")

            # Mid-epoch report
            if global_batch % report_every == 0:
                model.eval()
                with torch.no_grad():
                    test_imgs, _ = next(iter(test_loader))
                    test_imgs = test_imgs.to(device)
                    t_out = model(test_imgs)
                    test_mse = F.mse_loss(t_out['recon'], test_imgs).item()

                    S_batch = t_out['svd']['S']
                    S_orig = t_out['svd']['S_orig']
                    S_mean = S_batch.mean(dim=(0, 1))
                    S0 = S_mean[0].item()
                    SD = S_mean[-1].item()
                    ratio = S0 / (SD + 1e-8)
                    erank = model.effective_rank(
                        S_batch.reshape(-1, D)
                    ).mean().item()
                    s_delta = (S_batch - S_orig).abs().mean().item()
                    a_mean, a_std = per_layer_alphas()
                    cv_in_band = cv_band_lo <= last_cv <= cv_band_hi

                # TB scalars
                writer.add_scalar('train/loss', total_loss / n_seen, global_batch)
                writer.add_scalar('train/recon', total_recon / n_seen, global_batch)
                writer.add_scalar('test/mse', test_mse, global_batch)
                writer.add_scalar('geo/S0', S0, global_batch)
                writer.add_scalar('geo/SD', SD, global_batch)
                writer.add_scalar('geo/ratio', ratio, global_batch)
                writer.add_scalar('geo/erank', erank, global_batch)
                writer.add_scalar('geo/row_cv', last_cv, global_batch)
                writer.add_scalar('geo/cv_in_band', float(cv_in_band), global_batch)
                writer.add_scalar('geo/s_delta', s_delta, global_batch)
                writer.add_scalar('cross_attn/alpha_mean', a_mean, global_batch)
                writer.add_scalar('cross_attn/alpha_std', a_std, global_batch)
                writer.add_scalar('stab/prox', last_prox, global_batch)
                writer.add_scalar('stab/recon_w', recon_w, global_batch)
                writer.add_scalar('stab/epoch_max_grad', epoch_max_grad, global_batch)
                writer.add_scalar('stab/lr', opt.param_groups[0]['lr'], global_batch)

                history.append({
                    'epoch': epoch, 'global_batch': global_batch,
                    'train_recon': total_recon / n_seen,
                    'test_mse': test_mse,
                    'S0': S0, 'SD': SD, 'ratio': ratio, 'erank': erank,
                    'row_cv': last_cv, 'cv_in_band': cv_in_band,
                    's_delta': s_delta,
                    'alpha_mean': a_mean, 'alpha_std': a_std,
                    'epoch_max_grad': epoch_max_grad,
                })
                model.train()

        pbar.close()
        sched.step()
        epoch_time = time.time() - t0

        # ── Full epoch eval ──
        model.eval()
        test_mse_total, test_n = 0.0, 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                test_mse_total += F.mse_loss(out['recon'], imgs).item() * len(imgs)
                test_n += len(imgs)
        test_mse = test_mse_total / test_n

        # Geometry snapshot
        with torch.no_grad():
            sample, _ = next(iter(test_loader))
            sample = sample[:min(64, len(sample))].to(device)
            out = model(sample)
            S_mean = out['svd']['S'].mean(dim=(0, 1))
            S_orig = out['svd']['S_orig'].mean(dim=(0, 1))
            ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
            erank = model.effective_rank(out['svd']['S'].reshape(-1, D)).mean().item()
            s_delta = (S_mean - S_orig).abs().mean().item()
            a_mean, a_std = per_layer_alphas()
            cv_in_band = cv_band_lo <= last_cv <= cv_band_hi

        # Per-type MSE for noise variants
        type_str = ""
        if is_noise:
            active = list(range(16))
            ds_obj = train_loader.dataset
            if hasattr(ds_obj, 'active_types'):
                active = ds_obj.active_types
            type_mse = eval_per_type(
                model, active, img_size, device,
                n_per_type=32, channels=channels,
            )
            type_str = " ".join(f"{NOISE_NAMES[t][:4]}={v:.3f}"
                                  for t, v in sorted(type_mse.items()))

        # Byte accuracy for text
        byte_str = ""
        if is_text:
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:32].to(device)
                sample_out = model(sample_imgs)
                orig_b = ((sample_imgs.cpu().flatten(1) + 1.0) * 127.5)\
                    .round().clamp(0, 255).long()
                recon_b = ((sample_out['recon'].cpu().flatten(1) + 1.0) * 127.5)\
                    .round().clamp(0, 255).long()
                byte_acc = (orig_b == recon_b).float().mean().item()
            byte_str = f"bytes={byte_acc * 100:.1f}%"

        # Bit-recovery for binary tree
        tree_str = ""
        tree_metrics = None
        if is_tree:
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:64].to(device)
                sample_out = model(sample_imgs)
                orig_trees = decode_image_to_trees(sample_imgs, tree_depth)
                recon_trees = decode_image_to_trees(sample_out['recon'], tree_depth)
                tree_metrics = bit_recovery_metrics(orig_trees, recon_trees,
                                                    tree_depth)
            tree_str = (f"bits={tree_metrics['per_bit_acc']*100:.1f}% "
                        f"trees={tree_metrics['tree_exact_rate']*100:.1f}%")
            for lvl, acc in tree_metrics['per_level_acc'].items():
                writer.add_scalar(f'tree/level_{lvl}_acc', acc, epoch)
            writer.add_scalar('tree/per_bit_acc',
                              tree_metrics['per_bit_acc'], epoch)
            writer.add_scalar('tree/exact_rate',
                              tree_metrics['tree_exact_rate'], epoch)

        # Bit/token recovery for SentencePiece bits
        sp_str = ""
        sp_metrics = None
        if is_sentencepiece:
            sp_n_bits_eval = cfg.get('sp_n_bits', 16)
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:64].to(device)
                sample_out = model(sample_imgs)
                orig_bits = decode_image_to_tokens(sample_imgs, sp_n_bits_eval)
                recon_bits = decode_image_to_tokens(sample_out['recon'],
                                                    sp_n_bits_eval)
                sp_metrics = token_bit_recovery_metrics(orig_bits, recon_bits)
            sp_str = (f"bits={sp_metrics['per_bit_acc']*100:.1f}% "
                      f"toks={sp_metrics['token_exact_rate']*100:.1f}%")
            writer.add_scalar('sp/per_bit_acc',
                              sp_metrics['per_bit_acc'], epoch)
            writer.add_scalar('sp/token_exact_rate',
                              sp_metrics['token_exact_rate'], epoch)
            for bit_idx, acc in enumerate(sp_metrics['per_bit_position_acc']):
                writer.add_scalar(f'sp/bit_pos_{bit_idx}_acc', acc, epoch)
            for seq_idx, acc in enumerate(sp_metrics['per_seq_position_acc']):
                writer.add_scalar(f'sp/seq_pos_{seq_idx}_acc', acc, epoch)

        # Byte/trigram recovery for byte-trigram dataset
        bt_str = ""
        bt_metrics = None
        if is_byte_trigram:
            with torch.no_grad():
                # Sample size capped to fit memory at 256x256; n_cells per
                # image is large (4096 patches × 16 cells × 3 bytes at 256×256)
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:8].to(device)
                sample_out = model(sample_imgs)
                orig_bytes = ByteTrigramDataset.image_to_bytes(
                    sample_imgs.cpu(), patch_size, channels)
                recon_bytes = ByteTrigramDataset.image_to_bytes(
                    sample_out['recon'].cpu(), patch_size, channels)
                bt_metrics = byte_recovery_metrics(orig_bytes, recon_bytes)
            bt_str = (f"bytes={bt_metrics['per_byte_acc']*100:.1f}% "
                      f"trig={bt_metrics['trigram_exact_rate']*100:.1f}% "
                      f"L1={bt_metrics['per_byte_l1']:.2f}")
            writer.add_scalar('bt/per_byte_acc',
                              bt_metrics['per_byte_acc'], epoch)
            writer.add_scalar('bt/per_byte_l1',
                              bt_metrics['per_byte_l1'], epoch)
            writer.add_scalar('bt/trigram_exact_rate',
                              bt_metrics['trigram_exact_rate'], epoch)
            for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
                writer.add_scalar(f'bt/channel_{ch_name}_acc',
                                  bt_metrics['per_channel_acc'][ch_idx], epoch)

        print(f" {epoch:3d} | {total_loss/n_seen:.4f} {total_recon/n_seen:.4f} "
              f"{epoch_time:.0f}s | test={test_mse:.6f} | "
              f"S0={S_mean[0]:.3f} SD={S_mean[-1]:.3f} r={ratio:.2f} er={erank:.2f}"
              f" | cv={last_cv:.3f} band={'Y' if cv_in_band else 'N'} "
              f"Sd={s_delta:.5f} a={a_mean:.3f} g={epoch_max_grad:.1f} "
              f"{byte_str} {tree_str} {sp_str} {bt_str} {type_str}")

        # Per-epoch TB
        writer.add_scalar('epoch/test_mse', test_mse, epoch)
        writer.add_scalar('epoch/train_recon', total_recon / n_seen, epoch)
        writer.add_scalar('epoch/cv', last_cv, epoch)
        writer.add_scalar('epoch/cv_in_band', float(cv_in_band), epoch)
        writer.add_scalar('epoch/S0', S_mean[0].item(), epoch)
        writer.add_scalar('epoch/erank', erank, epoch)
        writer.add_scalar('epoch/max_grad', epoch_max_grad, epoch)
        writer.add_scalar('epoch/time_s', epoch_time, epoch)
        writer.add_scalar('epoch/alpha_mean', a_mean, epoch)
        writer.add_scalar('epoch/alpha_std', a_std, epoch)

        # End-of-epoch history record
        history.append({
            'epoch': epoch, 'global_batch': global_batch,
            'epoch_test_mse': test_mse,
            'train_recon': total_recon / n_seen,
            'S0': S_mean[0].item(), 'SD': S_mean[-1].item(),
            'ratio': ratio, 'erank': erank,
            'row_cv': last_cv, 'cv_in_band': cv_in_band,
            's_delta': s_delta,
            'alpha_mean': a_mean, 'alpha_std': a_std,
            'epoch_max_grad': epoch_max_grad,
            'epoch_time_s': epoch_time,
            'tree_metrics': tree_metrics,
            'sp_metrics': sp_metrics,
            'bt_metrics': bt_metrics,
        })

        # ── Curriculum: scheduled tier unlocks ──
        if curriculum == 'scheduled' and tier_schedule and epoch in tier_schedule:
            next_tier = tier_schedule[epoch]
            train_loader.dataset.unlock_tier(next_tier)
            test_loader.dataset.unlock_tier(next_tier)
            new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
            print(f"\n  ★ TIER {next_tier} UNLOCKED (epoch {epoch}): "
                  f"+{', '.join(new_names)}")
            active_now = [NOISE_NAMES[t] for t in train_loader.dataset.active_types]
            print(f"    Active: {active_now}\n")
            save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                            epoch, test_mse, do_upload=True)

        # ── Curriculum: patience-based promotion ──
        if curriculum == 'patience' and hasattr(train_loader.dataset, 'unlock_tier'):
            improvement = (tier_best_mse - test_mse) / (tier_best_mse + 1e-8)
            if test_mse < tier_best_mse:
                tier_best_mse = test_mse
            if improvement < 0.01:
                stale_epochs += 1
            else:
                stale_epochs = 0
            if (stale_epochs >= 10
                    and train_loader.dataset.current_tier >= 0
                    and train_loader.dataset.current_tier < max(TIERS.keys())):
                next_tier = train_loader.dataset.current_tier + 1
                train_loader.dataset.unlock_tier(next_tier)
                test_loader.dataset.unlock_tier(next_tier)
                new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
                print(f"\n  ★ PROMOTED TO TIER {next_tier}: +{', '.join(new_names)}")
                active_now = [NOISE_NAMES[t] for t in train_loader.dataset.active_types]
                print(f"    Active: {active_now}\n")
                tier_best_mse = test_mse
                stale_epochs = 0
                save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                                epoch, test_mse, do_upload=True)

        # ── Checkpointing ──
        if test_mse < best_recon:
            best_recon = test_mse
            save_checkpoint(os.path.join(save_dir, 'best.pt'),
                            epoch, test_mse, do_upload=False)

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(os.path.join(save_dir, f'epoch_{epoch:04d}.pt'),
                            epoch, test_mse)
            best_path = os.path.join(save_dir, 'best.pt')
            if os.path.exists(best_path):
                upload_to_hf(best_path, 'best.pt')
            writer.flush()
            if hf_enabled:
                try:
                    api.upload_folder(folder_path=tb_path,
                                      path_in_repo=f"{hf_version}/tensorboard/{run_name}",
                                      repo_id=hf_repo, repo_type="model")
                    print(f"  ☁️  TB synced")
                except Exception:
                    pass

    writer.close()

    # ── Final report ──
    final_report = {
        'run_name': run_name,
        'config': {
            'V': V, 'D': D, 'patch_size': patch_size,
            'hidden': hidden, 'depth': depth, 'n_cross_layers': n_cross,
            'n_heads': n_heads, 'smooth_mid': smooth_mid,
            'channels': channels,
            'linear_readout': linear_readout, 'svd_mode': svd_mode,
            'match_params': match_params,
            'dataset': dataset, 'img_size': img_size, 'batch_size': batch_size,
            'lr': lr, 'epochs': epochs, 'target_cv': target_cv,
            'allowed_types': allowed_types, 'curriculum': curriculum,
            'tier_schedule': tier_schedule, 'tree_depth': tree_depth,
        },
        'n_params': total_params,
        'n_patches': n_patches,
        'best_test_mse': best_recon,
        'history': history,
    }
    report_path = os.path.join(save_dir, 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"\n  Final report: {report_path}")
    if hf_enabled:
        upload_to_hf(report_path, 'final_report.json',
                      prefix=hf_version)

    print(f"\n  TRAINING COMPLETE — {hf_version}")
    print(f"  Best MSE: {best_recon:.6f}")
    print(f"  Checkpoints: {save_dir}/")

    # ── Build codebook (default on; opt out via cfg['build_codebook']=False) ──
    if cfg.get('build_codebook', True):
        try:
            from geolip_svae.inference.train_codebook import create_codebook
            print(f"\n  Building codebook for {hf_version}…")
            create_codebook(
                model, cfg,
                model_id=hf_version,
                out_dir=os.path.join(save_dir, 'codebooks'),
                upload_to_hf=hf_enabled,
                hf_repo=hf_repo,
                hf_version=hf_version,
                run_topology=cfg.get('build_topology', True),
            )
        except Exception as e:
            print(f"  ⚠️  codebook build failed: {type(e).__name__}: {e}")

    return final_report


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _is_jupyter_kernel() -> bool:
    """Detect if we're running inside an IPython/Jupyter kernel.

    Jupyter passes `-f /path/to/kernel.json` to the launcher's argv, which
    breaks naive argparse. When detected, we strip these kernel args before
    parsing so a `from geolip_svae.train import *; train(PRESETS['x'])`
    workflow works alongside the CLI.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def _filter_jupyter_args(argv):
    """Strip the `-f /path/to/kernel.json` pair if present."""
    out = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == '-f':
            skip_next = True
            continue
        if arg.startswith('-f='):
            continue
        out.append(arg)
    return out


def main(argv=None):
    """Entry point. argv defaults to sys.argv[1:] but can be passed for
    programmatic use. Tolerates Jupyter kernel args via parse_known_args."""
    import sys as _sys
    if argv is None:
        argv = _sys.argv[1:]
    if _is_jupyter_kernel():
        argv = _filter_jupyter_args(argv)

    parser = argparse.ArgumentParser(description='SVAE Unified Trainer (v2)')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Named preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from preset')
    parser.add_argument('--no-upload', action='store_true',
                        help='Disable HF upload')
    # Tolerate unknown args (e.g. Jupyter's stray flags). We don't error on them.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Quiet warning — not an error since we expect Jupyter pollution
        print(f"  [main] Ignoring unknown args: {unknown}")

    if args.list_presets:
        for name, cfg in PRESETS.items():
            ds = cfg['dataset']
            sz = cfg['img_size']
            ep = cfg['epochs']
            arch_tags = []
            if cfg.get('linear_readout'):
                arch_tags.append('lin_readout')
            if cfg.get('svd_mode', 'default') != 'default':
                arch_tags.append(f"svd={cfg['svd_mode']}")
            arch = f" [{'+'.join(arch_tags)}]" if arch_tags else ""
            pre = cfg.get('pretrained', 'scratch')
            print(f"  {name:<22s} {ds:<20s} {sz}×{sz}  {ep:>3d} ep"
                  f"  V={cfg['V']:<3d} D={cfg['D']:<3d}{arch}  from={pre}")
        return

    if not args.preset:
        parser.print_help()
        print("\nPresets:")
        for name in PRESETS:
            print(f"  {name}")
        return

    cfg = dict(PRESETS[args.preset])
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.no_upload:
        cfg['upload'] = False

    torch.set_float32_matmul_precision('high')
    train(cfg)


if __name__ == "__main__":
    main(['--preset', 't1_ps4_d4_v32_h128_svd'])
    #main(['--preset', 'h2_64_1channel'])

"""
geolip_svae.train_presets
==========================
Named training-preset registry for the SVAE Unified Trainer.

Every named cfg dict the trainer can consume lives here
(``python -m geolip_svae.train --preset NAME``). The module is deliberately
torch-free — only ``typing`` is imported — so reading the catalog
programmatically (preset listings, codebook tooling, downstream notebooks)
does not pay the trainer's heavy import cost. Adding a preset never
requires editing ``train.py``.

Public surface
--------------
    PRESETS : dict[str, dict[str, Any]]
        Every named preset. Keyed by preset name (the value passed to
        ``--preset NAME`` on the CLI).

    TEMPLATE : dict[str, Any]
        Fully-specified cfg dict listing every key the trainer recognizes,
        each set to its default. Copy this when authoring a new preset and
        delete the keys you accept defaults for. ``TEMPLATE`` is itself
        a valid preset that runs Fresnel-tiny on TinyImageNet at 64×64.

Back-compat
-----------
``geolip_svae.train`` re-imports ``PRESETS`` so existing callers using
``from geolip_svae.train import PRESETS`` continue to work.

Authoring a new preset
----------------------
1. Copy ``TEMPLATE`` into a new entry in ``PRESETS``.
2. Set the REQUIRED keys (V, D, patch_size, hidden, depth, n_cross,
   dataset, img_size, batch_size, lr, epochs, target_cv, hf_version).
3. Override the optional keys you care about; delete the rest.
4. Verify with ``python -m geolip_svae.train --list-presets``.

Cfg key categories
------------------
A preset dict groups keys into the following responsibility bands. Every
key consumed by the trainer or the dataset factory is listed in
``TEMPLATE`` below with its default + accepted values inline.

    Architecture        — passed to PatchSVAE constructor
    Ablation toggles    — F/G/H/L group ablations + new SVD dispatch knobs
    Training basics     — dataset, batch, lr, epochs
    Loss / band         — CV gate behavior
    Schedule / data     — pretrained, curriculum, tier_schedule, allowed_types
    Dataset-specific    — sp_*, bt_*, tree_depth, ds_size, val_size
    Output / IO         — save_dir, hf_repo, tb_dir, upload
    Codebook hook       — build_codebook, build_topology

Notes on important toggles
--------------------------
Sphere-solver path (h2-class):
    linear_readout=True, svd_mode='none', match_params=True
    — replaces SVD with a learned linear readout. Required to load any
    h2-64 battery checkpoint. Pre-Triton, this was the only practical
    path at D=4; with geolip-core 0.3.0 the fused N=4 Triton kernel makes
    real SVD viable, so new D=4 presets can drop these flags.

Triton fused SVD (D ∈ {2,3,4,5,6}):
    svd_method='auto' (default) routes through the dispatcher. For an
    explicit force, set svd_method='triton'. fp64 internal precision is
    the default and recommended for stable V orthogonality.

Per-site activations:
    activations={'enc_in': ..., 'enc_block_inner': ..., 'dec_in': ...,
                 'dec_block_inner': ..., 'boundary_smooth': ...}
    Each site picks any name from ``geolip_svae.model.ACTIVATIONS`` (21
    parameterless variants). Defaults preserve pre-refactor GELU behavior.
    The legacy ``activation='gelu'`` shortcut still works — it now means
    ``activations={'enc_in': activation}``.
"""
import math
from typing import Any, Dict


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE — every cfg key with default + accepted values
# ═══════════════════════════════════════════════════════════════════
#
# This is a fully-specified, runnable cfg dict. It mirrors Fresnel-tiny
# (TinyImageNet 64×64, 300 ep) so it works as a preset on its own. Copy
# it when adding a new preset and override the keys you need; the
# trainer treats every key here as optional EXCEPT those marked REQUIRED.
#
# The trainer's cfg-reader lives in ``geolip_svae.train.train()``;
# dataset-specific keys are consumed by the factories in
# ``geolip_svae.dataset_presets`` (see ``DATASET_FACTORIES``).

TEMPLATE: Dict[str, Any] = dict(

    # ── Architecture (passed to PatchSVAE) ───────────────────────────
    V              = 256,           # REQUIRED. Rows of encoded matrix M.
    D              = 16,            # REQUIRED. Cols of M / spectral dim. SVD runs at this N.
    patch_size     = 16,            # REQUIRED. Square patch edge in pixels.
    hidden         = 768,           # REQUIRED. MLP hidden dim (encoder + decoder).
    depth          = 4,             # REQUIRED. Number of residual blocks each side.
    n_cross        = 2,             # REQUIRED. Spectral cross-attention layer count.
    n_heads        = None,          # Heads per cross-attn layer; None → 2 if D≤8 else min(4,D).
    smooth_mid     = None,          # BoundarySmooth mid channels; None → 16 if ps≥16 else 8.
    channels       = 3,             # Image channel count; sets patch_dim = C*ps*ps.

    # ── Ablation toggles (F / G / H / L groups + SVD dispatch) ───────
    solver            = 'default',  # 'default' | 'conduit' (FLEighConduit telemetry).
    activation        = 'gelu',     # Legacy F-group: shortcut for activations['enc_in'].
                                    # Any key from geolip_svae.model.ACTIVATIONS.
    activations       = None,       # Per-site dict; None preserves defaults. Sites:
                                    #   'enc_in', 'enc_block_inner',
                                    #   'dec_in', 'dec_block_inner',
                                    #   'boundary_smooth'
                                    # Each value: any key from ACTIVATIONS (21 entries).
    row_norm          = 'sphere',   # G-group: 'sphere' | 'layernorm' | 'scale' | 'none'.
                                    # Sphere-norm (rows of M on S^(D-1)) is load-bearing.
    svd_mode          = 'default',  # H-group: 'default' | 'fp32' | 'fp64' | 'batch_shared' | 'none'.
                                    # 'none' + linear_readout=True is the sphere-solver path.
    svd_method        = 'auto',     # New (geolip-core 0.3.0). 'auto' | 'fl' | 'gram_eigh' |
                                    # 'triton' | 'torch'. Routes the batched_svd dispatcher
                                    # when svd_mode='default'. 'auto' picks the fused Triton
                                    # kernel for D∈{2..6} on CUDA when triton is installed.
    svd_compute_dtype = 'fp64',     # 'fp64' (recommended for V orthogonality) | 'fp32'.
    linear_readout    = False,      # H-group: replace SVD with learned readout.
    match_params      = True,       # When linear_readout=True: True → nn.Linear(V*D, V*D),
                                    # False → nn.Identity. False is param-saving but
                                    # geometrically degenerate.
    readout_radial_power = 2.0,     # When linear_readout=True: power for the radial component of the learned readout.
    init_scheme       = 'orthogonal',  # L-group: 'orthogonal' (default) | 'kaiming_normal' |
                                       # 'xavier_uniform' | 'normal_0_02'. enc_out always
                                       # gets re-orthogonalized regardless.

    # ── Training basics ──────────────────────────────────────────────
    dataset       = 'tiny_imagenet',   # REQUIRED. One of geolip_svae.dataset_presets.DATASET_FACTORIES keys:
                                       #   'tiny_imagenet' | 'imagenet_128' | 'imagenet_256'
                                       #   'curriculum_noise' | 'omega_noise' | 'scheduled_noise'
                                       #   'wikipedia' | 'binary_tree'
                                       #   'sentencepiece_bits' | 'byte_trigram'
    img_size      = 64,                # REQUIRED. Image edge in pixels (must be % patch_size == 0).
    batch_size    = 256,               # REQUIRED.
    lr            = 1e-4,              # REQUIRED. Adam (NOT AdamW — weight decay fights geometry).
    epochs        = 300,               # REQUIRED.
    target_cv     = 0.2915,            # REQUIRED. CV gate target for the soft-hand band.
    hf_version    = 'v_template',      # REQUIRED. HF prefix and run identifier.
    save_every    = 10,                # Save a checkpoint every N epochs.
    report_every  = 500,               # Mid-epoch report cadence (in batches).

    # ── Loss / CV soft-hand ──────────────────────────────────────────
    cv_weight     = 0.3,               # Weight on the CV penalty term.
    boost         = 0.5,               # Multiplier applied when CV is below the band.
    sigma         = 0.15,              # CV softness scale.
    cv_band_lo    = 0.25,              # Low edge of the in-band region. Default for V=256/D=16
                                       # noise. h2-class (V=32/D=4) lives ~0.85-1.05; override.
    cv_band_hi    = 1.25,              # High edge of the in-band region.

    # ── Schedule / data filtering ────────────────────────────────────
    pretrained    = None,              # HF path under hf_repo, e.g.
                                       # 'v40_freckles_noise/checkpoints/best.pt'.
    curriculum    = None,              # None | 'patience' | 'scheduled'. Noise datasets only.
    tier_schedule = None,              # dict[int_epoch -> int_tier] for curriculum='scheduled'.
    allowed_types = None,              # list[int] of NOISE_NAMES indices for noise datasets.
                                       # None = all 16. [0] = Gaussian only. Etc.

    # ── Dataset-specific ─────────────────────────────────────────────
    ds_size       = None,              # Override sample count for HF datasets (image / wiki / bt).
    val_size      = None,              # Override validation sample count.
    tree_depth    = 4,                 # binary_tree dataset: BFS depth of the binary trees.
    sp_tokenizer  = 'google-t5/t5-base',    # sentencepiece_bits: HF model id with spiece.model.
    sp_corpus     = 'wikitext-2-raw-v1',    # sentencepiece_bits: HF datasets id.
    sp_n_bits     = 16,                     # sentencepiece_bits: bits per token.
    bt_corpus     = 'wikitext-2-raw-v1',    # byte_trigram: HF datasets id.
    bt_max_corpus_bytes = None,             # byte_trigram: cap corpus bytes (None = full).

    # ── Output paths / HF upload ─────────────────────────────────────
    save_dir      = '/content/checkpoints',         # Local checkpoint directory.
    hf_repo       = 'AbstractPhil/geolip-SVAE',     # HF model repo id.
    hf_token      = None,              # HuggingFace API token. Normally LEFT
                                       # NONE in committed cfg dicts (it's a
                                       # secret). Set via:
                                       #   - CLI:  --hf-token $TOKEN
                                       #   - env:  HF_TOKEN env var (auto-picked)
                                       #   - prog: cfg['hf_token'] = '...' before
                                       #           calling train(cfg)
                                       # When set in cfg, the trainer populates
                                       # os.environ['HF_TOKEN'] and calls
                                       # huggingface_hub.login() before any
                                       # HfApi() calls fire.
    tb_dir        = '/content/runs',                # TensorBoard log directory.
    upload        = True,              # Upload checkpoints + logs + report to HF.

    # ── Codebook hook (runs at end of train()) ───────────────────────
    build_codebook = True,             # Extract a Codebook artifact from the final model.
    build_topology = True,             # Run kNN/PCA/ripser topology probes (ripser optional).
)


# ═══════════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════════
#
# Every entry below is a partial cfg dict — keys not present fall through
# to the defaults documented in TEMPLATE above. All architecture kwargs
# are explicit; presets opt into the h2-class sphere-solver path via
#     linear_readout=True, svd_mode='none', match_params=True.

PRESETS: Dict[str, Dict[str, Any]] = {
    # ── Fresnel (images) ──
    'fresnel_tiny': dict(
        # Architecture
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        # Training
        dataset='tiny_imagenet', img_size=64, batch_size=256,
        lr=1e-4, epochs=300, target_cv=0.2915,
        hf_version='v19_fresnel_tiny', save_every=10,
    ),
    'fresnel_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_128', img_size=128, batch_size=128,
        lr=1e-4, epochs=50, target_cv=0.2915,
        hf_version='v12_imagenet128', save_every=1,
    ),
    'fresnel_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_256', img_size=256, batch_size=64,
        lr=1e-4, epochs=20, target_cv=0.2915,
        hf_version='v13_imagenet256', save_every=1,
    ),

    # ── Johanna (noise) ──
    'johanna_tiny': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='curriculum_noise', img_size=64, batch_size=512,
        lr=3e-4, epochs=300, target_cv=0.125,
        hf_version='v18_johanna_curriculum', save_every=25,
        curriculum='patience',
    ),
    'johanna_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='omega_noise', img_size=128, batch_size=128,
        lr=1e-4, epochs=200, target_cv=0.125,
        hf_version='v16_johanna_omega', save_every=10,
        pretrained='v14_noise/checkpoints/epoch_0200.pt',
    ),
    'johanna_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='scheduled_noise', img_size=256, batch_size=64,
        lr=1e-4, epochs=30, target_cv=0.2915,
        hf_version='v20_johanna_base', save_every=5,
        curriculum='scheduled', tier_schedule={5: 1, 8: 2, 10: 3, 12: 4},
    ),

    # ── Alexandria (text) ──
    'alexandria_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='wikipedia', img_size=128, batch_size=128,
        lr=1e-4, epochs=100, target_cv=0.2915,
        hf_version='v22_alexandria_small', save_every=10,
        pretrained='v16_johanna_omega/checkpoints/best.pt',
        ds_size=200000, val_size=5000,
    ),

    # ── Freckles (D=4 noise specialist, 2.55M params) ──
    # Resolution-invariant by construction: cross-attn weights dimensioned by
    # D=4, not N. Same weights work at any patch count. v41 inits from v40,
    # v42 inits from v41 — cumulative resolution-transfer chain.
    'freckles_64': dict(
        # Freckles architecture
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Training: 16-type omega noise at 64x64 (256 patches) for 100 ep
        dataset='omega_noise', img_size=64, batch_size=256,
        lr=1e-4, epochs=100, target_cv=0.125,  # historical value, predates 0.20-0.23 band
        hf_version='v40_freckles_noise', save_every=10,
        ds_size=500_000, val_size=10_000,
        report_every=500,
    ),
    'freckles_256': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Resolution transfer test: v40 weights, fine-tune at 256x256 (4096 patches).
        # 1 epoch is enough — spectrum stays within 0.4% of v40.
        dataset='omega_noise', img_size=256, batch_size=64,
        lr=1e-5, epochs=1, target_cv=0.125,
        hf_version='v41_freckles_256', save_every=1,
        pretrained='v40_freckles_noise/checkpoints/best.pt',
        ds_size=200_000, val_size=2_000,
        report_every=200,
    ),
    'freckles_512': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Continued resolution transfer: v41 weights, fine-tune at 512x512.
        dataset='omega_noise', img_size=512, batch_size=16,
        lr=1e-5, epochs=1, target_cv=0.125,
        hf_version='v42_freckles_512', save_every=1,
        pretrained='v41_freckles_256/checkpoints/best.pt',
        ds_size=80_000, val_size=1_000,
        report_every=100,
    ),

    # ── Fresnel-64 (D=4 ImageNet specialist, Freckles geometry) ──
    # Same architecture as Freckles, trained on ImageNet crops instead of noise.
    # 297M unique crops; spectrum locks at step 17,500. Identical attractor to
    # Freckles v40 → universal manifold across modalities.
    'fresnel_64': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        dataset='tiny_imagenet', img_size=64, batch_size=256,
        lr=1e-4, epochs=100, target_cv=0.125,
        hf_version='v50_fresnel_64', save_every=10,
        report_every=500,
    ),
    # Note: the v50_fresnel_64 model was *also* run through 140M+ random 64x64
    # crops of ImageNet-256 via train_streaming.py — see that module for the
    # continuation trainer. The "fresnel_64_256" name on HF refers to that
    # streaming continuation (sublens perspective, not a 256x256 finetune).

    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_single': dict(
        # H2_linear_matched architecture
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=20, target_cv=0.9, cv_weight=0.0,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.80, cv_band_hi=1.05,
        allowed_types=[0],
        hf_version='h2_64_repro_single', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    't1_ps4_d4_v32_h128_svd': dict(
        # H2_linear_matched architecture
        V=32, D=4, patch_size=4, hidden=128, depth=1, n_cross=1, n_heads=8,
        smooth_mid=16,
        svd_method='triton',
        #linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=10, target_cv=1.0, cv_weight=0.1,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.80, cv_band_hi=1.25,
        #allowed_types=[0],
        hf_version='t1_ps4_d4_v32_h128_svd', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_1channel': dict(
        # H2_linear_matched architecture
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16, channels=1,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=256,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=10, target_cv=0.9, cv_weight=0.0,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.80, cv_band_hi=1.25,
        # allowed_types=[0],
        hf_version='h2_64_1channel', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # Diagnostics cadence
        report_every=200,
    ),


    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_5channel': dict(
        # H2_linear_matched architecture
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16, channels=5,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=256,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=10, target_cv=0.9, cv_weight=0.0,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.80, cv_band_hi=1.25,
        #allowed_types=[0],
        hf_version='h2_64_5channel', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_5channel_v40_d4_ps4_h80': dict(
        # H2_linear_matched architecture
        V=40, D=4, patch_size=4, hidden=80, depth=1, n_cross=1, n_heads=4,
        smooth_mid=10, channels=5,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=256,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=10, target_cv=0.9, cv_weight=0.0,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.80, cv_band_hi=1.25,
        # allowed_types=[0],
        hf_version='h2_64_5channel_v40_d4_ps4_h80', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── H2-class dodecahedron (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_h64_v64_d16_ps16_single_full_noise_image64x64': dict(
        # H2_linear_matched architecture
        V=64,
        D=16,
        patch_size=16,
        hidden=64,
        depth=1,
        n_cross=1,
        n_heads=8,
        smooth_mid=16,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=100, target_cv=0.2, cv_weight=0.001,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.10, cv_band_hi=0.35,
        #allowed_types=[0],
        hf_version='h2_h64_v64_d16_ps16_single_full_noise', save_every=5,
        ds_size=1_280_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── H2-class dodecahedron (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_dodecahedron_v1': dict(
        # H2_linear_matched architecture
        V=20,
        D=3,
        patch_size=4,
        hidden=64,
        depth=1,
        n_cross=1,
        n_heads=3,
        smooth_mid=20,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=50, target_cv=0.9, cv_weight=0.01,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.85, cv_band_hi=1.25,
        allowed_types=[0],
        hf_version='h2_64_repro_single', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),


    # ── H2-class dodecahedron (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_dodecahedron_v2': dict(
        # H2_linear_matched architecture
        V=20,
        D=3,
        patch_size=4,
        hidden=192,
        depth=4,
        n_cross=4,
        n_heads=3,
        smooth_mid=3,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=50, target_cv=0.9, cv_weight=0.01,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.85, cv_band_hi=1.25,
        #allowed_types=[0],
        hf_version='h2_64_dodecahedron_v2_gauss_svd', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),

        # ── H2-class tesseract (tesseract-solver, the architecture used by h2-64 batteries) ──
    'h2_64_tesseract_v1': dict(
        # H2_linear_matched architecture
        V=8,
        D=4,
        patch_size=4,
        hidden=64,
        depth=1,
        n_cross=1,
        n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        # H2-class natural attractor: CV ~0.85-0.92 on noise content, NOT the
        # 0.13-0.30 noise-substrate band (that's for V=256/D=16 class). The
        # h2-class with V=32/D=4 lives in a different basin of CV-space because
        # of the small D and linear readout. Don't pull toward 0.215.
        lr=1e-3, epochs=50, target_cv=0.9, cv_weight=0.01,
        # H2-class natural band (per measured runs): 0.80-1.05
        cv_band_lo=0.85, cv_band_hi=1.25,
        allowed_types=[0],
        hf_version='h2_64_repro_single', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── BinaryTree substrate prototype ──
    'bintree_proto': dict(
        # H2-64 architecture exactly
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training on i.i.d. depth-4 binary trees (BFS-encoded, ±1 floats).
        # Bintree-iid measured CV trajectory: 0.80-1.01 across 20 ep.
        # H2-class natural attractor for ±1 bit content is CV~1.0.
        dataset='binary_tree', img_size=16, batch_size=256,
        lr=1e-3, epochs=20, target_cv=0.9, cv_weight=0.0,
        # H2-class band; bintree-iid landed CV 0.80-1.01
        cv_band_lo=0.80, cv_band_hi=1.05,
        hf_version='bintree_proto_v1', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Tree config
        tree_depth=4,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── SentencePiece-bit substrate prototype ──
    # First REAL-DATA prototype on the substrate path. Same h2-class architecture
    # as bintree_proto. Each patch holds the 16-bit binary representation of one
    # t5-base SentencePiece token ID (vocab 32128, fits in 15 bits, 16-bit
    # encoding leaves a 1-bit buffer above vocab range). Per-image = 16 patches
    # = 16 contiguous tokens from a wikitext-2 corpus excerpt. Reconstruction
    # objective. cv_weight=0 — let the substrate self-solve to whatever
    # geometry the SentencePiece bit distribution selects, not the noise band.
    'sentencepiece_proto': dict(
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        dataset='sentencepiece_bits', img_size=16, batch_size=256,
        # H2-class CV band; SP-bit content may shift this slightly but
        # cv_weight=0 means the value is informational only.
        lr=1e-3, epochs=20, target_cv=0.9, cv_weight=0.0,
        cv_band_lo=0.80, cv_band_hi=1.05,
        hf_version='sentencepiece_proto_v1', save_every=5,
        ds_size=200_000, val_size=2_000,
        # SentencePiece config
        sp_tokenizer='google-t5/t5-base',  # HF model id with spiece.model
        sp_corpus='wikitext-2-raw-v1',     # HF datasets id
        sp_n_bits=16,                      # bits per token (vocab 32128 < 2^15=32768)
        # Diagnostics cadence
        report_every=200,
    ),

    # Byte-trigrams as RGB pixel-equivalents at 256×256.
    # Each spatial cell of each 4×4 patch holds one (R,G,B) byte trigram from
    # the corpus stream. 4096 patches × 16 cells × 3 bytes = 196,608 bytes per
    # image (~192 KB of text, roughly a long Wikipedia article or chapter).
    # No padding; every float carries signal. 16.7M-cardinality input per cell
    # → ~1M-cardinality codebook capacity per patch on S^3 = real compression
    # work for the codebook to do (unlike sentencepiece_proto, which gave the
    # model a 1/3-filled patch with sign-only signal).
    'byte_trigram_proto': dict(
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        dataset='byte_trigram', img_size=256, batch_size=8,
        # H2-class natural band — informational. If the codebook actually
        # engages at this scale (cross-attn α leaves 0.023, erank drops below
        # 4.0, ratio leaves 1.0), CV will leave the band as well.
        lr=1e-3, epochs=20, target_cv=0.9, cv_weight=0.0,
        cv_band_lo=0.80, cv_band_hi=1.05,
        hf_version='byte_trigram_proto_v1', save_every=2,
        ds_size=20_000, val_size=200,
        # ByteTrigram config — no corpus cap by default
        bt_corpus='wikitext-103-raw-v1',
        # Diagnostics cadence
        report_every=100,
    ),

    # Byte-trigram at 128×128 with batch=256 for 100 epochs. The 256×256/batch=8
    # run (byte_trigram_proto_v1) plateaued at α=0.043 by ep 13, suggesting
    # under-saturation — the model found a stable point at only 22% of the α
    # cap (0.2). Hypothesis: larger batch (32× more gradient signal per step)
    # + more epochs + matched data volume will let α push higher and shift the
    # equilibrium. 128×128 cuts patches from 4096 to 1024 (4× cheaper per
    # image) but batch=256 means 8× more text per gradient step than the
    # prior run (12 MB vs 1.5 MB).
    #
    # ds_size=1M matches the noise-substrate training precedent (Johanna,
    # Fresnel, h2-64 batteries: 1M samples/epoch). Anything less is
    # under-data'd against that baseline, where 10M total sample-views
    # was sufficient to characterize the projective-clean codebook.
    # At 100 epochs that's 100M sample-views, ~28-40 hours wall-clock on A100.
    # If shorter wall-clock is needed, drop epochs to 20-30 (still hits or
    # exceeds the noise precedent's 10M total views in 10-30 hours).
    'byte_trigram_proto_64': dict(
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        dataset='byte_trigram', img_size=64, batch_size=1024,
        lr=1e-3, epochs=50, target_cv=1.0, cv_weight=0.01,
        cv_band_lo=0.80, cv_band_hi=1.3,
        hf_version='byte_trigram_proto_64_patch_2_v1', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # ByteTrigram config — same corpus as the 256×256 run
        # No max_corpus_bytes; load the full ~500MB wikitext-103.
        bt_corpus='wikitext-103-raw-v1',
        # Diagnostics cadence — at ds_size/batch ≈ 3906 batches/epoch,
        # report_every=500 gives ~8 reports per epoch including end.
        #pretrained='byte_trigram_proto_128_v1/checkpoints/best.pt',
        report_every=500,
    ),

    'byte_trigram_proto_64_radmag_1': dict(
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        readout_radial_power=1.0,
        dataset='byte_trigram', img_size=64, batch_size=1024,
        lr=1e-3, epochs=50, target_cv=1.0, cv_weight=0.01,
        cv_band_lo=0.80, cv_band_hi=1.3,
        hf_version='byte_trigram_proto_64_radmag_1', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # ByteTrigram config — same corpus as the 256×256 run
        # No max_corpus_bytes; load the full ~500MB wikitext-103.
        bt_corpus='wikitext-103-raw-v1',
        # Diagnostics cadence — at ds_size/batch ≈ 3906 batches/epoch,
        # report_every=500 gives ~8 reports per epoch including end.
        # pretrained='byte_trigram_proto_128_v1/checkpoints/best.pt',
        report_every=500,
    ),

    'byte_trigram_proto_64_pi_radmag_pi': dict(
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        readout_radial_power=math.pi,
        dataset='byte_trigram', img_size=64, batch_size=1024,
        lr=1e-3, epochs=50, target_cv=1.0, cv_weight=0.01,
        cv_band_lo=0.80, cv_band_hi=1.3,
        hf_version='byte_trigram_proto_64_pi_radmag_pi', save_every=5,
        ds_size=1_000_000, val_size=10_000,
        # ByteTrigram config — same corpus as the 256×256 run
        # No max_corpus_bytes; load the full ~500MB wikitext-103.
        bt_corpus='wikitext-103-raw-v1',
        # Diagnostics cadence — at ds_size/batch ≈ 3906 batches/epoch,
        # report_every=500 gives ~8 reports per epoch including end.
        # pretrained='byte_trigram_proto_128_v1/checkpoints/best.pt',
        report_every=500,
    ),
}

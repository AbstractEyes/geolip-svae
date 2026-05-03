"""
svae_proto.exp_001_vocab_trigram_recall.cfg
============================================
cfg dict(s) for the vocabulary trigram experiment.

Three variants pre-configured, each matching the architecture spec of
its reference preset in geolip_svae.train_presets exactly:

    CFG_PROTO_64    — H2-class sphere-solver, mirrors `h2_64_single`:
                      V=32 D=4 ps=4 hidden=64 depth=1 n_cross=1 n_heads=4,
                      linear_readout=True svd_mode='none' match_params=True.
                      ~57K params (the actual h2-64 battery cell).
    CFG_FRECKLES_64 — Freckles geometry, mirrors `freckles_64`:
                      V=48 D=4 ps=4 hidden=384 depth=4 n_cross=2,
                      with REAL SVD (svd_method='auto' → fused Triton N=4
                      kernel), so the comparison vs CFG_PROTO_64 isolates
                      "linear-readout sphere-solver vs real Vt" on the same
                      vocab data. ~2.5M params.
    CFG_FRESNEL_128 — Full geolip arch, mirrors `fresnel_small`:
                      V=256 D=16 ps=16 hidden=768 depth=4 n_cross=2.
                      ~17M params, the "production-scale" upper bound.

All three feed the same VocabTrigramDataset so per_token_exact_acc is
directly comparable across them.

Naming convention: every hf_version starts with the experiment id
(`exp_001_*`) so all artifacts on HuggingFace sort cleanly under the
experiment that produced them. Override `hf_version` (or `hf_repo`) per
run via run.py's --hf-version / --hf-repo flags if needed.
"""
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────
# _BASE — only DATASET / LOSS-band / OUTPUT keys live here.
#
# Architecture and training-schedule keys (V, D, hidden, depth, n_cross,
# batch_size, lr, epochs, ds_size, val_size, save_every, report_every)
# are SET PER-VARIANT because the three variants have very different
# parameter counts (~57K / ~2.5M / ~17M) and different per-step costs.
# Trying to share batch_size / ds_size between them would either starve
# the small one or OOM the big one.
# ─────────────────────────────────────────────────────────────────────
_BASE: Dict[str, Any] = dict(
    # ── Dataset ─────────────────────────────────────────────────────
    # Full wikitext-103 (~500 MB), uncapped. Matches byte_trigram_proto_64
    # in the main package, the proven reference scale for h2-class on text.
    dataset='vocab_trigram',           # registered at runtime by run.py
    vt_corpus='wikitext-103-raw-v1',
    vt_tokenizer='google-t5/t5-base',
    vt_max_corpus_chars=None,          # uncapped — load the full corpus
    vt_seed=0,

    # ── Loss / CV soft hand ─────────────────────────────────────────
    # H2-class natural attractor on text-like content sits in the
    # 0.80-1.25 band (per byte_trigram_proto_64 measurements). cv_weight
    # tiny but non-zero — soft hand toward the band, not a hard pull.
    target_cv=1.0,
    cv_weight=0.01,
    boost=0.5,
    sigma=0.15,
    cv_band_lo=0.80,
    cv_band_hi=1.25,

    # ── Output / IO ────────────────────────────────────────────────
    upload=True,                       # uploads to hf_repo via HF_TOKEN
    hf_repo='AbstractPhil/geolip-svae-text',

    # ── Codebook hook (post-train) ─────────────────────────────────
    build_codebook=True,
    build_topology=True,
)


# ─────────────────────────────────────────────────────────────────────
# CFG_PROTO_64 — H2-class sphere-solver, the canonical h2-64 cell.
#
# Architecture spec is BIT-FOR-BIT identical to `h2_64_single` from
# geolip_svae.train_presets — same V/D/ps/hidden/depth/n_cross/n_heads,
# same linear_readout/svd_mode/match_params, same lr=1e-3.
# ds_size=1M / batch=1024 / 50 ep matches `byte_trigram_proto_64`'s
# proven scale on text data with this architecture.
# ─────────────────────────────────────────────────────────────────────
CFG_PROTO_64: Dict[str, Any] = dict(
    _BASE,
    # Architecture (H2_linear_matched — matches h2_64_single)
    V=32, D=4, patch_size=4,
    hidden=64, depth=1, n_cross=1, n_heads=4,
    smooth_mid=16, channels=3,
    linear_readout=True, svd_mode='none', match_params=True,

    # Training — h2_64_single's lr=1e-3, byte_trigram_proto_64's data scale
    img_size=64, batch_size=1024,
    lr=1e-3, epochs=50,
    ds_size=1_000_000, val_size=10_000,
    save_every=5,
    report_every=500,                  # ~2 reports/epoch at ~975 batches/ep

    hf_version='exp_001_vocab_h2_64',
)


# ─────────────────────────────────────────────────────────────────────
# CFG_FRECKLES_64 — Freckles geometry with real SVD (Triton fused N=4).
#
# Architecture matches `freckles_64`: V=48 D=4 ps=4 hidden=384 depth=4
# n_cross=2, NO sphere-solver. svd_method='auto' fires the fused N=4
# Triton kernel on CUDA. ~2.5M params (40× larger than CFG_PROTO_64).
#
# Direct comparison: same data, same image size, same patch size,
# same D=4. CFG_PROTO_64 vs CFG_FRECKLES_64 isolates the question
# "does the learned-readout sphere-solver lose something the real
# SVD captures, on text recall?"
# ─────────────────────────────────────────────────────────────────────
CFG_FRECKLES_64: Dict[str, Any] = dict(
    _BASE,
    # Architecture (matches freckles_64)
    V=48, D=4, patch_size=4,
    hidden=384, depth=4, n_cross=2,
    channels=3,
    linear_readout=False,
    svd_mode='default',
    svd_method='auto',                 # fused Triton N=4 on CUDA
    svd_compute_dtype='fp64',

    # Training — freckles_64's lr=1e-4. Smaller batch than CFG_PROTO_64
    # because the model is 40× bigger; ds_size halved to keep wall time
    # in the same band (~2× per-step cost × 0.5× steps ≈ same).
    img_size=64, batch_size=256,
    lr=1e-4, epochs=50,
    ds_size=500_000, val_size=10_000,
    save_every=5,
    report_every=500,                  # ~4 reports/epoch at ~1953 batches/ep

    hf_version='exp_001_vocab_freckles_64',
)


# ─────────────────────────────────────────────────────────────────────
# CFG_FRESNEL_128 — Full geolip arch at 128². Production-scale upper
# bound on what the metric can achieve.
#
# Architecture matches `fresnel_small`: V=256 D=16 ps=16 hidden=768
# depth=4 n_cross=2. ~17M params. Default SVD path; D=16 falls outside
# the Triton-fused range (N=2..6 only) so it dispatches to the
# torch.linalg.eigh fp64 path automatically.
# ─────────────────────────────────────────────────────────────────────
CFG_FRESNEL_128: Dict[str, Any] = dict(
    _BASE,
    # Architecture (matches fresnel_small)
    V=256, D=16, patch_size=16,
    hidden=768, depth=4, n_cross=2,
    channels=3,
    linear_readout=False,
    svd_mode='default',
    svd_method='auto',                 # auto routes D=16 to gram_eigh fp64
    svd_compute_dtype='fp64',

    # Training — fresnel_small's lr=1e-4 batch=128 30 epochs is enough
    # at full corpus given 17M params. ds_size scaled down because each
    # sample is 49152 bytes (4× h2-64) and each step is much heavier.
    img_size=128, batch_size=128,
    lr=1e-4, epochs=30,
    ds_size=200_000, val_size=5_000,
    save_every=5,
    report_every=500,                  # ~3 reports/epoch at ~1562 batches/ep

    hf_version='exp_001_vocab_fresnel_128',
)

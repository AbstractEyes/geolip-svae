"""
prototypes.001_vocab_trigram_recall.cfg
========================================
cfg dict(s) for the vocabulary trigram experiment.

Three variants pre-configured:
    CFG_PROTO_64    — 64×64 image, h2-class arch, 100 ep — quick smoke test.
    CFG_FRECKLES_64 — 64×64 image, freckles arch (D=4 SVD path) — same data,
                      different geometry; the comparison that motivates this
                      experiment.
    CFG_FRESNEL_128 — 128×128 image, full geolip arch, longer run — the
                      "production-scale" datapoint to anchor the metric.

All three feed the same VocabTrigramDataset so per_token_exact_acc is
directly comparable across them.
"""
from typing import Any, Dict


# Common cfg keys shared by all three variants.
_BASE: Dict[str, Any] = dict(
    # Dataset
    dataset='vocab_trigram',           # registered at runtime by run.py
    vt_corpus='wikitext-2-raw-v1',
    vt_tokenizer='google-t5/t5-base',
    vt_max_corpus_chars=4_000_000,
    ds_size=10_000,
    val_size=1_000,
    vt_seed=0,

    # Loss / band
    target_cv=1.0,
    cv_weight=0.01,
    boost=0.5,
    sigma=0.15,
    cv_band_lo=0.80,
    cv_band_hi=1.25,

    # Output
    save_every=10,
    report_every=200,
    upload=False,                      # local until results are graduated

    # Codebook hook stays default-on so post-train artifact captures the
    # axes alongside the vocab eval.
    build_codebook=True,
    build_topology=True,
)


# h2-class single-battery sphere-solver, 64x64 byte_trigram-derived
CFG_PROTO_64: Dict[str, Any] = dict(
    _BASE,
    # Architecture (h2-class, sphere-solver path)
    V=32, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
    channels=3,
    linear_readout=True,
    svd_mode='none',
    match_params=True,
    smooth_mid=16,

    # Training
    img_size=64, batch_size=128,
    lr=1e-4, epochs=100,
    hf_version='proto001_vocab_h2_64',
)


# Freckles-class D=4 with REAL SVD (Triton fused N=4 kernel) — same
# geometry as h2-64 except the SVD is genuine, not the linear-readout
# workaround. Comparing CFG_FRECKLES_64 vs CFG_PROTO_64 isolates whether
# real Vt buys us anything for token-level recall.
CFG_FRECKLES_64: Dict[str, Any] = dict(
    _BASE,
    V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
    channels=3,
    linear_readout=False,
    svd_mode='default',
    svd_method='auto',                 # fused Triton N=4 fires on CUDA
    svd_compute_dtype='fp64',
    smooth_mid=16,

    img_size=64, batch_size=128,
    lr=1e-4, epochs=100,
    hf_version='proto001_vocab_freckles_64',
)


# Full geolip D=16 at 128x128 — the "the model is big enough to encode
# language structure" datapoint. Slower but gives the upper bound for the
# experiment's success criterion.
CFG_FRESNEL_128: Dict[str, Any] = dict(
    _BASE,
    V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
    channels=3,
    linear_readout=False,
    svd_mode='default',
    svd_method='auto',
    svd_compute_dtype='fp64',

    img_size=128, batch_size=64,
    lr=1e-4, epochs=50,
    hf_version='proto001_vocab_fresnel_128',
)

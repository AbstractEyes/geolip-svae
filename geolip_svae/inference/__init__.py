"""
geolip_svae.inference
======================
Production inference framework.

Public API::

    # Loading
    from geolip_svae.inference import (
        load_model, list_versions, VERSIONS,
        UnsupportedCheckpointError,
    )

    # Resolution-aware encode/reconstruct (free-function form)
    from geolip_svae.inference import (
        encode_at_scale, reconstruct_at_scale,
    )

    # Calibration data
    from geolip_svae.inference import (
        make_calibration, get_calibration, register_calibration,
        gen_gaussian, gen_uniform, gen_sixteen_noise,
        NOISE_NAMES, CALIBRATION_REGISTRY,
    )

    # Codebook
    from geolip_svae.inference import (
        Codebook, CodebookMetadata,
        extract_codebook,
        uniform_projective_angle, codebook_mean_projective_angle,
        identify_antipodal_pairs, collapse_to_axes,
    )

    # Orchestrator (recommended entry point)
    from geolip_svae.inference import (
        InferenceEngine,
        CodebookMissingError, CodebookIncompatibleError,
    )

    # Text-side wrapper (byte-trigram sentence similarity)
    from geolip_svae.inference import (
        SentenceEncoder,
    )

    # Legacy flat-module API (back-compat shims)
    from geolip_svae.inference import (
        encode, decode, reconstruct, batched_forward, compute_axis_codebook,
    )

Quick start::

    from geolip_svae.inference import (
        load_model, InferenceEngine, make_calibration,
    )

    model, cfg = load_model(hf_version='v50_fresnel_64')
    engine = InferenceEngine(model)

    # Reconstruct at any resolution
    result = engine.reconstruct(images_512x512)

    # Extract and use a codebook
    calib = make_calibration('sixteen_noise', n=64, size=64)
    cb = engine.extract_codebook(calib, attach=True,
                                  model_id='v50_fresnel_64',
                                  calibration_name='sixteen_noise')
    print(cb)  # → Codebook(D=4, n_axes=27, dev=+0.012, clean=True)

    # Project onto axes
    out = engine.encode_axes(test_images)
    activations = out['activations']  # [B, n_patches, V, n_axes]

Module layout
-------------
    loading      — ``load_model``, version registry, checkpoint resolution
    scaling      — ``encode_at_scale`` / ``reconstruct_at_scale`` direct/tile
                   dispatch, padding helpers, patch-size override
    calibration  — calibration data generators + registry
    codebook     — antipodal-collapse helpers, ``Codebook`` artifact,
                   ``extract_codebook``
    engine       — ``InferenceEngine`` orchestrator, codebook lifecycle,
                   compatibility checks
    text         — ``SentenceEncoder`` text-side wrapper for byte-trigram
                   models: string → image, sentence signatures, similarity
    legacy       — pre-rebuild flat-module shims (``encode``, ``decode``,
                   ``reconstruct``, ``batched_forward``,
                   ``compute_axis_codebook``)
"""

# ── Loading ──
from geolip_svae.inference.loading import (
    load_model,
    list_versions,
    VERSIONS,
    HF_REPO,
    UnsupportedCheckpointError,
)

# ── Scaling (resolution-aware) ──
from geolip_svae.inference.scaling import (
    encode_at_scale,
    reconstruct_at_scale,
)

# ── Calibration ──
from geolip_svae.inference.calibration import (
    CalibrationFn,
    NOISE_NAMES,
    gen_gaussian,
    gen_uniform,
    gen_sixteen_noise,
    CALIBRATION_REGISTRY,
    register_calibration,
    get_calibration,
    make_calibration,
)

# ── Codebook (artifact + helpers) ──
from geolip_svae.inference.codebook import (
    # Helpers (canonical home; arrays/model.py re-imports from here)
    identify_antipodal_pairs,
    collapse_to_axes,
    SUPPORTED_AGG,
    # Geometric primitives
    uniform_projective_angle,
    codebook_mean_projective_angle,
    # Codebook artifact
    CodebookMetadata,
    Codebook,
    # Extraction
    extract_codebook,
)

# ── Engine ──
from geolip_svae.inference.engine import (
    InferenceEngine,
    InferenceEngineDefaults,
    CodebookMissingError,
    CodebookIncompatibleError,
)

# ── Text-side wrapper (byte-trigram I/O + per-patch similarity) ──
from geolip_svae.inference.text import (
    SentenceEncoder,
    PAD_STRATEGIES,
    SIGNATURE_MODES,
    AGG_METHODS,
    text_to_image,
    image_to_text,
    text_real_patch_mask,
    text_features,
    text_recovery_metrics,
    per_patch_similarity,
)

# ── Legacy back-compat shims ──
from geolip_svae.inference.legacy import (
    encode,
    decode,
    reconstruct,
    batched_forward,
    compute_axis_codebook,
)


__all__ = [
    # Loading
    'load_model',
    'list_versions',
    'VERSIONS',
    'HF_REPO',
    'UnsupportedCheckpointError',
    # Scaling
    'encode_at_scale',
    'reconstruct_at_scale',
    # Calibration
    'NOISE_NAMES',
    'gen_gaussian',
    'gen_uniform',
    'gen_sixteen_noise',
    'CALIBRATION_REGISTRY',
    'register_calibration',
    'get_calibration',
    'make_calibration',
    # Codebook helpers
    'identify_antipodal_pairs',
    'collapse_to_axes',
    'SUPPORTED_AGG',
    # Codebook geometry
    'uniform_projective_angle',
    'codebook_mean_projective_angle',
    # Codebook artifact
    'CodebookMetadata',
    'Codebook',
    'extract_codebook',
    # Engine
    'InferenceEngine',
    'InferenceEngineDefaults',
    'CodebookMissingError',
    'CodebookIncompatibleError',
    # Text wrapper
    'SentenceEncoder',
    'PAD_STRATEGIES',
    'SIGNATURE_MODES',
    'AGG_METHODS',
    'text_to_image',
    'image_to_text',
    'text_real_patch_mask',
    'text_features',
    'text_recovery_metrics',
    'per_patch_similarity',
    # Legacy
    'encode',
    'decode',
    'reconstruct',
    'batched_forward',
    'compute_axis_codebook',
]
"""
geolip_svae.inference
======================
Production inference framework.

Public API::

    from geolip_svae.inference import (
        load_model, list_versions, VERSIONS, UnsupportedCheckpointError,
        encode_at_scale, reconstruct_at_scale,
        make_calibration, get_calibration, register_calibration,
        gen_gaussian, gen_uniform, gen_sixteen_noise,
        NOISE_NAMES, CALIBRATION_REGISTRY,
        Codebook, CodebookMetadata, extract_codebook,
        uniform_projective_angle, codebook_mean_projective_angle,
        identify_antipodal_pairs, collapse_to_axes,
        InferenceEngine, CodebookMissingError, CodebookIncompatibleError,
        create_codebook,
        run_topology_analysis, run_array_topology_analysis,
        TopologyReport, ArrayTopologyReport,
        DEFAULT_CALIBRATIONS, infer_class_from_cfg, HAVE_RIPSER,
        SentenceEncoder,
        encode, decode, reconstruct, batched_forward, compute_axis_codebook,
    )

Module layout
-------------
    loading        - load_model, version registry, checkpoint resolution
    scaling        - encode_at_scale / reconstruct_at_scale
    calibration    - calibration data generators + registry
    codebook       - antipodal-collapse helpers, Codebook artifact, extract_codebook
    engine         - InferenceEngine orchestrator, codebook lifecycle
    train_codebook - codebook creation pipeline + topology probes
                     (kNN-graph / local PCA / ripser PH).
                     Trainer-integrated; produces Codebook + TopologyReport
                     alongside the final checkpoint.
    text           - SentenceEncoder text-side wrapper for byte-trigram models
    legacy         - pre-rebuild flat-module shims
"""

from geolip_svae.inference.loading import (
    load_model, list_versions, VERSIONS, HF_REPO,
    UnsupportedCheckpointError,
)

from geolip_svae.inference.scaling import (
    encode_at_scale, reconstruct_at_scale,
)

from geolip_svae.inference.calibration import (
    CalibrationFn, NOISE_NAMES,
    gen_gaussian, gen_uniform, gen_sixteen_noise,
    CALIBRATION_REGISTRY,
    register_calibration, get_calibration, make_calibration,
)

from geolip_svae.inference.codebook import (
    identify_antipodal_pairs, collapse_to_axes, SUPPORTED_AGG,
    uniform_projective_angle, codebook_mean_projective_angle,
    CodebookMetadata, Codebook, extract_codebook,
)

from geolip_svae.inference.engine import (
    InferenceEngine, InferenceEngineDefaults,
    CodebookMissingError, CodebookIncompatibleError,
)

from geolip_svae.inference.train_codebook import (
    DEFAULT_CALIBRATIONS, infer_class_from_cfg,
    create_codebook,
    run_topology_analysis, run_array_topology_analysis,
    TopologyReport, ArrayTopologyReport,
    HAVE_RIPSER,
)

from geolip_svae.inference.text import (
    SentenceEncoder,
    PAD_STRATEGIES, SIGNATURE_MODES, AGG_METHODS,
    text_to_image, image_to_text, text_real_patch_mask,
    text_features, text_recovery_metrics, per_patch_similarity,
)

from geolip_svae.inference.legacy import (
    encode, decode, reconstruct, batched_forward, compute_axis_codebook,
)


__all__ = [
    'load_model', 'list_versions', 'VERSIONS', 'HF_REPO',
    'UnsupportedCheckpointError',
    'encode_at_scale', 'reconstruct_at_scale',
    'CalibrationFn', 'NOISE_NAMES',
    'gen_gaussian', 'gen_uniform', 'gen_sixteen_noise',
    'CALIBRATION_REGISTRY',
    'register_calibration', 'get_calibration', 'make_calibration',
    'identify_antipodal_pairs', 'collapse_to_axes', 'SUPPORTED_AGG',
    'uniform_projective_angle', 'codebook_mean_projective_angle',
    'CodebookMetadata', 'Codebook', 'extract_codebook',
    'InferenceEngine', 'InferenceEngineDefaults',
    'CodebookMissingError', 'CodebookIncompatibleError',
    'DEFAULT_CALIBRATIONS', 'infer_class_from_cfg',
    'create_codebook',
    'run_topology_analysis', 'run_array_topology_analysis',
    'TopologyReport', 'ArrayTopologyReport',
    'HAVE_RIPSER',
    'SentenceEncoder',
    'PAD_STRATEGIES', 'SIGNATURE_MODES', 'AGG_METHODS',
    'text_to_image', 'image_to_text', 'text_real_patch_mask',
    'text_features', 'text_recovery_metrics', 'per_patch_similarity',
    'encode', 'decode', 'reconstruct', 'batched_forward',
    'compute_axis_codebook',
]

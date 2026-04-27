"""
geolip-svae — Spectral Variational Autoencoder
=================================================
Omega Tokens on S^15. Patch-based SVD autoencoder with
spectral cross-attention and sphere-normalized encoding.

    from geolip_svae import PatchSVAE, load_model
    model, cfg = load_model(hf_version='v13_imagenet256')
    out = model(images)  # out['recon'], out['svd']

    from geolip_svae import PatchSVAEv2
    v2 = PatchSVAEv2()  # hierarchical spectral cascade

    from geolip_svae import SpectralTokenizer, build_codebook
    codebook = build_codebook(save_path='codebook.json')
    tokenizer = SpectralTokenizer(codebook)
    image, ids, strings = tokenizer.text_to_image("Hello, world!")
"""

__version__ = "0.9.0"

from geolip_svae.model import (
    PatchSVAE,
    SpectralCrossAttention,
    BoundarySmooth,
    gram_eigh_svd,
    gram_eigh_svd_conduit,
    cv_of,
    extract_patches,
    stitch_patches,
)
from geolip_svae.inference import load_model, encode, decode, reconstruct
from geolip_svae.experimental.experimental_codebook import (
    SpectralTokenizer,
    build_codebook,
    generate_patch,
)
from geolip_svae.arrays import (
    BatteryArrayConfig,
    BatteryArrayModel,
    build_array,
)

__all__ = [
    # v1 model
    "PatchSVAE",
    "SpectralCrossAttention",
    "BoundarySmooth",
    "gram_eigh_svd",
    "gram_eigh_svd_conduit",
    "cv_of",
    "extract_patches",
    "stitch_patches",
    # Inference
    "load_model",
    "encode",
    "decode",
    "reconstruct",
    # Codebook
    "SpectralTokenizer",
    "build_codebook",
    "generate_patch",
    # Battery arrays
    "BatteryArrayConfig",
    "BatteryArrayModel",
    "build_array",
]
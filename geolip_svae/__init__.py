"""
geolip-svae — Spectral Variational Autoencoder
=================================================
Omega Tokens on S^15. Patch-based SVD autoencoder with
spectral cross-attention and sphere-normalized encoding.

    from geolip_svae import PatchSVAE, load_model
    model, cfg = load_model(hf_version='v13_imagenet256')
    out = model(images)  # out['recon'], out['svd']

    from geolip_svae import PatchSVAEv2
    v2 = PatchSVAEv2.from_v1(model)  # conduit-forced decoder

    from geolip_svae import SpectralTokenizer, build_codebook
    codebook = build_codebook(save_path='codebook.json')
    tokenizer = SpectralTokenizer(codebook)
    image, ids, strings = tokenizer.text_to_image("Hello, world!")
"""

__version__ = "0.3.0"

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
from geolip_svae.model_v2 import (
    PatchSVAEv2,
    ConduitDecoder,
    ModeProcessor,
)
from geolip_svae.inference import load_model, encode, decode, reconstruct
from geolip_svae.spectral_codebook import (
    SpectralTokenizer,
    build_codebook,
    generate_patch,
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
    # v2 model (conduit-forced decoder)
    "PatchSVAEv2",
    "ConduitDecoder",
    "ModeProcessor",
    # Inference
    "load_model",
    "encode",
    "decode",
    "reconstruct",
    # Codebook
    "SpectralTokenizer",
    "build_codebook",
    "generate_patch",
]
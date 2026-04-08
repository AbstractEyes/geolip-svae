"""
geolip-svae — Spectral Variational Autoencoder
=================================================
Omega Tokens on S^15. Patch-based SVD autoencoder with
spectral cross-attention and sphere-normalized encoding.

    from geolip_svae import PatchSVAE, load_model
    model, cfg = load_model(hf_version='v13_imagenet256')
    out = model(images)  # out['recon'], out['svd']
"""

__version__ = "0.1.0"

from geolip_svae.model import (
    PatchSVAE,
    SpectralCrossAttention,
    BoundarySmooth,
)
from geolip_svae.inference import load_model, encode, decode, reconstruct

__all__ = [
    "PatchSVAE",
    "SpectralCrossAttention",
    "BoundarySmooth",
    "load_model",
    "encode",
    "decode",
    "reconstruct",
]
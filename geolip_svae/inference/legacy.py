"""
geolip_svae.inference.legacy
=============================
Back-compat shims for the pre-rebuild flat-module API.

Code written against the old flat ``inference.py`` (pre-v0.7.0) may
import any of these symbols::

    encode, decode, reconstruct, batched_forward, compute_axis_codebook

Each shim preserves the legacy call signature and behavior. New code
should prefer the ``InferenceEngine`` + ``extract_codebook`` surface
exposed by ``geolip_svae.inference``.

Stability
---------
These functions exist for the rebuild transition and may be removed
in a future major version. They are NOT the recommended path for new
work. Their docstrings call out the recommended replacement.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from geolip_svae.inference.codebook import extract_codebook
from geolip_svae.inference.scaling import encode_at_scale


@torch.no_grad()
def encode(model, images: torch.Tensor) -> Dict[str, Any]:
    """Encode images via direct mode at native resolution. (legacy)

    Returns the SVD dict augmented with ``gh``/``gw`` patch grid hints.

    Recommended replacement::

        from geolip_svae.inference import encode_at_scale
        out = encode_at_scale(model, images, mode='direct')

    Or via the engine::

        engine = InferenceEngine(model)
        out = engine.encode(images, mode='direct')
    """
    return encode_at_scale(model, images, mode='direct')


@torch.no_grad()
def decode(model, svd: Dict[str, Any]) -> torch.Tensor:
    """Reconstruct images from a stored SVD dict. (legacy, v1-only)

    Args:
        model: PatchSVAE in eval mode.
        svd: dict returned by ``encode()``; must contain ``U``, ``S``,
            ``Vt``, ``gh``, ``gw``.

    Recommended replacement: hold images instead of an SVD dict and use
    ``reconstruct(model, images)`` or ``engine.reconstruct(images)``,
    which work for any model and any resolution. The legacy ``decode``
    pathway requires the exact U/S/Vt cache the v1 model produced and
    cannot be applied to newer architectures.
    """
    from geolip_svae.model import stitch_patches
    decoded = model.decode_patches(svd['U'], svd['S'], svd['Vt'])
    recon = stitch_patches(
        decoded, svd['gh'], svd['gw'], model.patch_size,
        channels=getattr(model, 'channels', 3),
    )
    return model.boundary_smooth(recon)


@torch.no_grad()
def reconstruct(model, images: torch.Tensor) -> torch.Tensor:
    """Full round-trip reconstruction. Returns just the tensor. (legacy)

    Recommended replacement::

        out = reconstruct_at_scale(model, images, mode='direct')
        recon = out['recon']  # plus mse_per_image, mode_used, ...

    Or via the engine::

        engine = InferenceEngine(model)
        recon = engine.reconstruct(images)['recon']
    """
    return model(images)['recon']


@torch.no_grad()
def batched_forward(
    model,
    images: torch.Tensor,
    max_batch: int = 16,
) -> Dict[str, torch.Tensor]:
    """Chunked forward pass for OOM-safe inference. (legacy)

    Returns dict with ``recon``, ``S``, ``S_orig``, ``M`` — all on CPU.
    """
    device = next(model.parameters()).device
    all_recon, all_S, all_S_orig, all_M = [], [], [], []
    model.eval()
    for i in range(0, len(images), max_batch):
        batch = images[i:i + max_batch].to(device)
        out = model(batch)
        all_recon.append(out['recon'].cpu())
        all_S.append(out['svd']['S'].cpu())
        all_S_orig.append(out['svd']['S_orig'].cpu())
        all_M.append(out['svd']['M'].cpu())
    return {
        'recon': torch.cat(all_recon),
        'S': torch.cat(all_S),
        'S_orig': torch.cat(all_S_orig),
        'M': torch.cat(all_M),
    }


@torch.no_grad()
def compute_axis_codebook(
    model,
    calibration_images: torch.Tensor,
    sample_agg: str = 'mean',
    patch_agg: str = 'mean',
    patch_idx: Optional[int] = None,
    threshold: float = -0.9,
    batch_size: int = 64,
) -> torch.Tensor:
    """Extract a codebook as a raw ``[n_axes, D]`` tensor. (legacy)

    Pre-rebuild API: returned just the axes tensor. This shim discards
    the metadata that ``extract_codebook`` produces and returns only the
    axes for backward compatibility.

    Recommended replacement::

        from geolip_svae.inference import extract_codebook
        cb = extract_codebook(model, calibration_images, ...)
        # cb is a Codebook artifact with full provenance:
        #   cb.axes, cb.metadata, cb.pairs, cb.unpaired
        #   cb.deviation(), cb.is_projective_clean()
        #   cb.save(path), Codebook.load(path)

    Or via an engine::

        engine = InferenceEngine(model)
        cb = engine.extract_codebook(
            calibration_images,
            model_id='v50_fresnel_64',
            calibration_name='gaussian',
            attach=True,
        )
    """
    cb = extract_codebook(
        model, calibration_images,
        sample_agg=sample_agg, patch_agg=patch_agg,
        patch_idx=patch_idx, threshold=threshold,
        batch_size=batch_size,
    )
    return cb.axes


__all__ = [
    'encode',
    'decode',
    'reconstruct',
    'batched_forward',
    'compute_axis_codebook',
]
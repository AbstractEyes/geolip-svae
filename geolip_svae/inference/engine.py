"""
geolip_svae.inference.engine
=============================
``InferenceEngine`` — orchestrator for production inference.

Holds a model + optional default codebook + override hooks. Inference
runs through the engine instance; codebook usage is decoupled from
encoding/reconstruction.

Design (Abstract Powered Research framework v0):
    - Codebook is a first-class artifact, attached to an engine
      separately from the model. An engine without a codebook still
      works for raw encode/reconstruct.
    - All overrides explicit at construction OR at call-site:
      ``patch_size``, ``tile_size``, ``batch_size``, ``mode``,
      ``codebook=``.
    - Loud failures over silent fallback. If a user requests
      ``encode_axes`` without a codebook attached and without passing
      one, the call raises rather than returning raw M.

Usage::

    from geolip_svae.inference import (
        load_model, InferenceEngine, extract_codebook, make_calibration,
    )

    model, cfg = load_model(hf_version='v50_fresnel_64')
    engine = InferenceEngine(model)

    # Raw encode / reconstruct (no codebook needed)
    enc = engine.encode(images)
    recon = engine.reconstruct(images)

    # Extract codebook on demand
    calib = make_calibration('sixteen_noise', n=64, size=64)
    codebook = engine.extract_codebook(calib, model_id='v50_fresnel_64')

    # Or attach a previously-saved one
    codebook = Codebook.load('codebooks/freckles_v40.safetensors')
    engine.attach_codebook(codebook)
    activations = engine.encode_axes(images)  # uses attached codebook

    # Or pass one ad-hoc without attaching
    activations = engine.encode_axes(images, codebook=other_codebook)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from geolip_svae.inference.codebook import (
    Codebook,
    extract_codebook as _extract_codebook,
)
from geolip_svae.inference.scaling import (
    encode_at_scale,
    reconstruct_at_scale,
)


# ════════════════════════════════════════════════════════════════════
# CodebookMissingError
# ════════════════════════════════════════════════════════════════════

class CodebookMissingError(RuntimeError):
    """Raised when an engine method requires a codebook and none is available.

    Distinguishable from generic RuntimeError so callers can catch and
    fall back (e.g. extract a codebook on the fly) without swallowing
    unrelated errors.
    """


class CodebookIncompatibleError(RuntimeError):
    """Raised when a provided codebook fails ``compatible_with(model)``."""


# ════════════════════════════════════════════════════════════════════
# InferenceEngine
# ════════════════════════════════════════════════════════════════════

@dataclass
class InferenceEngineDefaults:
    """Per-engine defaults applied when call-site doesn't override.

    Each field is also overridable on every method that accepts it.
    """
    tile_size: Optional[int] = None       # None → 64 inside scaling.py
    mode: str = 'auto'                     # 'direct' / 'tile' / 'auto'
    batch_size: int = 16
    patch_size: Optional[int] = None      # None → use model.patch_size
    codebook_threshold: float = 0.05      # for projective-clean checks


class InferenceEngine:
    """Production inference orchestrator for any sphere-solver model.

    Args:
        model: PatchSVAE (or any model whose forward returns dict
            with ``'svd'`` and ``'recon'`` keys; e.g. a single bank
            from BatteryArrayModel).
        codebook: optional default ``Codebook`` to use for ``encode_axes``.
        tile_size: default tile size for at-scale operations
            (None → 64 inside scaling.py).
        mode: default scaling mode ('auto', 'direct', or 'tile').
        batch_size: default forward-pass chunk size.
        patch_size: default patch size override (None → use model's own).
        require_codebook_compatibility: if True, ``attach_codebook`` and
            codebook-using methods will raise if the codebook's D doesn't
            match the model's. Default True; set False for cross-model
            experiments where you explicitly want to apply a foreign
            codebook.
    """

    def __init__(
        self,
        model,
        codebook: Optional[Codebook] = None,
        *,
        tile_size: Optional[int] = None,
        mode: str = 'auto',
        batch_size: int = 16,
        patch_size: Optional[int] = None,
        require_codebook_compatibility: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.defaults = InferenceEngineDefaults(
            tile_size=tile_size,
            mode=mode,
            batch_size=batch_size,
            patch_size=patch_size,
        )
        self.require_codebook_compatibility = require_codebook_compatibility
        self._codebook: Optional[Codebook] = None
        if codebook is not None:
            self.attach_codebook(codebook)

    # ── Codebook lifecycle ───────────────────────────────────────────

    @property
    def codebook(self) -> Optional[Codebook]:
        """The currently attached codebook, or None."""
        return self._codebook

    def attach_codebook(self, codebook: Codebook) -> None:
        """Attach a codebook as this engine's default for ``encode_axes``.

        Checks compatibility if ``require_codebook_compatibility`` is True.
        Raises ``CodebookIncompatibleError`` on mismatch.
        """
        if self.require_codebook_compatibility:
            ok, reason = codebook.compatible_with(self.model)
            if not ok:
                raise CodebookIncompatibleError(
                    f"Codebook incompatible with model: {reason}. "
                    f"To override, instantiate the engine with "
                    f"require_codebook_compatibility=False."
                )
        self._codebook = codebook

    def detach_codebook(self) -> Optional[Codebook]:
        """Remove and return the currently attached codebook (if any)."""
        cb = self._codebook
        self._codebook = None
        return cb

    @torch.no_grad()
    def extract_codebook(
        self,
        calibration_images: torch.Tensor,
        sample_agg: str = 'mean',
        patch_agg: str = 'mean',
        patch_idx: Optional[int] = None,
        threshold: float = -0.9,
        batch_size: Optional[int] = None,
        *,
        model_id: str = '',
        calibration_name: str = '',
        attach: bool = False,
    ) -> Codebook:
        """Extract a codebook from this engine's model.

        Convenience wrapper over ``geolip_svae.inference.codebook.extract_codebook``
        that fills ``model_class`` automatically and (optionally)
        attaches the result.

        Args:
            calibration_images: ``[N, C, H, W]``.
            sample_agg, patch_agg, patch_idx, threshold: see codebook module.
            batch_size: forward chunk size; defaults to engine default.
            model_id: provenance string for metadata.
            calibration_name: provenance string for metadata.
            attach: if True, also call ``attach_codebook`` on the result.

        Returns:
            The new ``Codebook`` (regardless of whether it was attached).
        """
        bs = batch_size if batch_size is not None else self.defaults.batch_size
        cb = _extract_codebook(
            self.model,
            calibration_images,
            sample_agg=sample_agg,
            patch_agg=patch_agg,
            patch_idx=patch_idx,
            threshold=threshold,
            batch_size=bs,
            model_id=model_id,
            model_class=type(self.model).__name__,
            calibration_name=calibration_name,
        )
        if attach:
            self.attach_codebook(cb)
        return cb

    # ── Encoding / reconstruction (codebook-independent) ─────────────

    def _resolve_kwargs(self, **call_kwargs) -> Dict[str, Any]:
        """Merge per-call overrides with engine defaults."""
        merged = {
            'tile_size': self.defaults.tile_size,
            'mode': self.defaults.mode,
            'batch_size': self.defaults.batch_size,
            'patch_size': self.defaults.patch_size,
        }
        for k, v in call_kwargs.items():
            if v is not None:
                merged[k] = v
        return merged

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        tile_size: Optional[int] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Encode images at arbitrary resolution. Returns SVD dict.

        Returns the full SVD dict from the model (with ``M``, ``S``,
        ``S_orig``, optional ``U``/``Vt``, plus ``gh``/``gw`` patch grid
        and ``mode_used``). Codebook is not used.
        """
        kw = self._resolve_kwargs(
            tile_size=tile_size, mode=mode,
            batch_size=batch_size, patch_size=patch_size,
        )
        return encode_at_scale(self.model, images, **kw)

    @torch.no_grad()
    def reconstruct(
        self,
        images: torch.Tensor,
        *,
        tile_size: Optional[int] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Round-trip reconstruction at arbitrary resolution.

        Returns dict with ``recon``, ``mode_used``, ``mse_per_image``,
        and tile diagnostics if tile mode was used. Codebook is not used.
        """
        kw = self._resolve_kwargs(
            tile_size=tile_size, mode=mode,
            batch_size=batch_size, patch_size=patch_size,
        )
        return reconstruct_at_scale(self.model, images, **kw)

    # ── Codebook-dependent operations ────────────────────────────────

    def _resolve_codebook(
        self,
        codebook: Optional[Codebook],
    ) -> Codebook:
        """Return the codebook to use: arg > attached > raise."""
        if codebook is not None:
            if self.require_codebook_compatibility:
                ok, reason = codebook.compatible_with(self.model)
                if not ok:
                    raise CodebookIncompatibleError(
                        f"Provided codebook incompatible: {reason}"
                    )
            return codebook
        if self._codebook is not None:
            return self._codebook
        raise CodebookMissingError(
            "encode_axes requires a codebook. Either pass codebook=... "
            "to this call, attach one via engine.attach_codebook(cb), "
            "or extract one via engine.extract_codebook(calib, attach=True)."
        )

    @torch.no_grad()
    def encode_axes(
        self,
        images: torch.Tensor,
        *,
        codebook: Optional[Codebook] = None,
        tile_size: Optional[int] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Encode and project onto the codebook axes.

        Returns dict with:
            ``M``: raw [B, n_patches, V, D] tensor from encode
            ``axes``: ``[n_axes, D]`` codebook axes used
            ``activations``: ``[B, n_patches, V, n_axes]`` projection of M
                onto axes (cosine similarity per row × axis pair)
            ``mode_used``: 'direct' or 'tile'
            plus all other encode_at_scale diagnostics

        If no codebook is attached and none is passed, raises
        ``CodebookMissingError``.
        """
        cb = self._resolve_codebook(codebook)
        enc = self.encode(
            images,
            tile_size=tile_size, mode=mode,
            batch_size=batch_size, patch_size=patch_size,
        )

        M = enc['M']  # [B, n_patches, V, D]
        # Project: M_unit · axes.T per (B, P, V, n_axes)
        axes = cb.axes.to(M.device).to(M.dtype)
        # Sphere-norm M rows just in case (model output should already be)
        norms = M.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        M_unit = M / norms
        # Activations: [B, P, V, D] @ [D, n_axes] → [B, P, V, n_axes]
        activations = M_unit @ axes.T

        result = dict(enc)
        result['axes'] = axes.cpu()
        result['activations'] = activations.cpu()
        return result

    @torch.no_grad()
    def quantize_axes(
        self,
        images: torch.Tensor,
        *,
        codebook: Optional[Codebook] = None,
        tile_size: Optional[int] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Encode and assign each row of M to its nearest codebook axis.

        Returns dict with:
            ``codes``: ``[B, n_patches, V]`` int64 codebook indices
            ``confidence``: ``[B, n_patches, V]`` cosine-sim of best match
            plus all encode diagnostics
        """
        out = self.encode_axes(
            images,
            codebook=codebook,
            tile_size=tile_size, mode=mode,
            batch_size=batch_size, patch_size=patch_size,
        )
        activations = out['activations']  # [B, P, V, n_axes]
        # Use absolute value because antipodal axes match either sign
        confidence, codes = activations.abs().max(dim=-1)
        out['codes'] = codes
        out['confidence'] = confidence
        return out

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cb_str = (f"codebook={self._codebook!r}"
                   if self._codebook is not None
                   else "codebook=None")
        return (
            f"InferenceEngine(model={type(self.model).__name__}, "
            f"{cb_str}, mode={self.defaults.mode!r})"
        )


__all__ = [
    'InferenceEngine',
    'InferenceEngineDefaults',
    'CodebookMissingError',
    'CodebookIncompatibleError',
]
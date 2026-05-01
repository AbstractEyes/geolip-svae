"""
geolip_svae.inference.text
==========================
Text-side I/O and per-patch similarity for byte-trigram-trained sphere-solvers.

Wraps ``ByteTrigramDataset.bytes_to_image`` / ``image_to_bytes`` for
text↔image conversion and forwards through an ``InferenceEngine`` for
per-patch feature extraction. Per-patch similarity preserves the
architecture's per-patch granularity: features stay ``[P, feat_dim]``
and aggregation runs on cosine scalars, not on features.

Public surface
--------------
    text_to_image       — string → (3, H, W)
    image_to_text       — (3, H, W) → string
    text_real_patch_mask— mask in MODEL patch grid, real-byte patches
    text_features       — per-patch features + mask, mode ∈ SIGNATURE_MODES
    text_recovery_metrics— text → image → recon → text byte-fidelity check
    per_patch_similarity— pairwise [K_a, K_b] cosine, aggregated via AGG_METHODS
    SentenceEncoder     — dataclass bundling engine + (img_size, patch_size, pad)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from geolip_svae.inference.engine import InferenceEngine, CodebookMissingError


PAD_STRATEGIES = ('space', 'zero', 'repeat', 'truncate')
SIGNATURE_MODES = ('M', 'codes')
AGG_METHODS = ('patch_mean', 'best_match', 'aligned')


# ── Byte-text I/O primitives ──────────────────────────────────────────

def _padded_bytes(text: str, target: int, pad: str) -> Tuple[np.ndarray, int]:
    """UTF-8 encode and pad/truncate to ``target`` bytes per ``pad``."""
    b = text.encode('utf-8')
    n_real = min(len(b), target)
    if n_real == target:
        return np.frombuffer(b[:target], dtype=np.uint8), target
    if pad == 'truncate':
        raise ValueError(
            f"text is {len(b)} bytes, image needs {target}; "
            f"pad='truncate' refuses to pad"
        )
    if pad == 'zero':
        buf = bytearray(target)
        buf[:n_real] = b[:n_real]
    elif pad == 'space':
        buf = bytearray(b' ' * target)
        buf[:n_real] = b[:n_real]
    elif pad == 'repeat':
        if n_real == 0:
            buf = bytearray(b' ' * target)
        else:
            n = (target + n_real - 1) // n_real
            buf = bytearray((b[:n_real] * n)[:target])
    else:
        raise ValueError(f"pad={pad!r} not in {PAD_STRATEGIES}")
    return np.frombuffer(bytes(buf), dtype=np.uint8), n_real


def text_to_image(
    text: str,
    img_size: int = 64,
    patch_size: int = 2,
    pad: str = 'space',
    channels: int = 3,
) -> torch.Tensor:
    """Encode a string as a ``(channels, img_size, img_size)`` byte-n-gram image.

    Same path as training. ``patch_size`` controls the byte→pixel layout
    in ``ByteTrigramDataset.bytes_to_image``; ``channels`` is the n-gram
    size (3 = RGB trigram, default; matches the dataset's ``channels``
    kwarg). For an in-distribution forward, ``channels`` should equal
    the model's ``channels`` attribute.
    """
    from geolip_svae.train import ByteTrigramDataset
    target = img_size * img_size * channels
    chunk, _ = _padded_bytes(text, target, pad)
    img = ByteTrigramDataset.bytes_to_image(
        chunk, img_size, patch_size, channels,
    )
    return torch.from_numpy(img)


def image_to_text(
    image: torch.Tensor,
    patch_size: int = 2,
    n_bytes: Optional[int] = None,
    channels: int = 3,
) -> str:
    """Decode a ``(channels, H, W)`` image back to a UTF-8 string.

    If ``n_bytes`` is given, only the first ``n_bytes`` of the recovered
    byte stream are decoded (drops padded patches).
    """
    from geolip_svae.train import ByteTrigramDataset
    if image.ndim == 3:
        image = image.unsqueeze(0)
    bytes_t = ByteTrigramDataset.image_to_bytes(image, patch_size, channels)
    flat = bytes_t.reshape(-1).cpu().numpy().astype(np.uint8)
    if n_bytes is not None:
        flat = flat[:n_bytes]
    return bytes(flat).decode('utf-8', errors='replace')


def text_real_patch_mask(
    text: str,
    img_size: int,
    patch_size: int,
    model_patch_size: int,
    pad: str,
    channels: int = 3,
) -> torch.Tensor:
    """Boolean mask over the model's patch grid: True iff the patch
    overlaps at least one real (un-padded) text byte.

    When ``patch_size != model_patch_size``, the byte→pixel layout uses
    ``patch_size`` but the model's feature output is on a grid of
    ``model_patch_size``-patches, so the mask is computed in model-grid
    coordinates by laying out a byte-level marker through ``bytes_to_image``
    and aggregating max-over-pixel to model patches.
    """
    from geolip_svae.train import ByteTrigramDataset

    n_model_patches = (img_size // model_patch_size) ** 2
    if pad == 'repeat':
        return torch.ones(n_model_patches, dtype=torch.bool)

    target = img_size * img_size * channels
    n_real = min(len(text.encode('utf-8')), target)
    if n_real >= target:
        return torch.ones(n_model_patches, dtype=torch.bool)
    if n_real == 0:
        return torch.zeros(n_model_patches, dtype=torch.bool)

    byte_mask = np.zeros(target, dtype=np.uint8)
    byte_mask[:n_real] = 0xFF
    mask_img = torch.from_numpy(
        ByteTrigramDataset.bytes_to_image(
            byte_mask, img_size, patch_size, channels,
        )
    )
    gh = gw = img_size // model_patch_size
    blocks = mask_img.reshape(channels, gh, model_patch_size, gw, model_patch_size)
    return (blocks.amax(dim=(0, 2, 4)) > 0).flatten()


# ── Per-patch features ────────────────────────────────────────────────

@torch.no_grad()
def text_features(
    engine: InferenceEngine,
    text: Union[str, Sequence[str]],
    img_size: int = 64,
    patch_size: int = 2,
    pad: str = 'space',
    mode: str = 'M',
    channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-patch features and real-patch mask for one or more texts.

    Args:
        engine: ``InferenceEngine`` carrying the model (and codebook for
            ``mode='codes'``).
        mode:
            ``'M'``     — flat sphere-norm encoder rows ``[V*D]`` per patch.
            ``'codes'`` — per-row argmax over codebook axes, one-hot
                          flattened ``[V*n_axes]`` per patch. Requires an
                          attached codebook.

    Returns:
        features: ``[P, feat_dim]`` for single text, ``[B, P, feat_dim]``
                  for sequence input. CPU.
        mask:     ``[P]`` or ``[B, P]`` of bool, True for real-byte patches.
    """
    if mode not in SIGNATURE_MODES:
        raise ValueError(f"mode={mode!r} not in {SIGNATURE_MODES}")

    single = isinstance(text, str)
    texts = [text] if single else list(text)

    device = next(engine.model.parameters()).device
    images = torch.stack([
        text_to_image(t, img_size, patch_size, pad, channels) for t in texts
    ]).to(device)

    # mode='direct' through the engine: tile mode permutes patches across
    # sub-tiles, which would invalidate the byte→patch mask alignment.
    if mode == 'M':
        enc = engine.encode(images, mode='direct')
        M = enc['M']                                  # [B, P, V, D]
        feat = M.reshape(*M.shape[:2], -1)            # [B, P, V*D]
    else:  # 'codes'
        out = engine.encode_axes(images, mode='direct')
        acts = out['activations']                     # [B, P, V, n_axes]
        n_axes = acts.shape[-1]
        codes = acts.abs().argmax(dim=-1)             # [B, P, V] int
        oh = F.one_hot(codes, num_classes=n_axes).float()
        feat = oh.reshape(*oh.shape[:2], -1)          # [B, P, V*n_axes]

    model_ps = int(engine.model.patch_size)
    masks = torch.stack([
        text_real_patch_mask(t, img_size, patch_size, model_ps, pad, channels)
        for t in texts
    ])

    feat = feat.cpu()
    masks = masks.cpu()
    if single:
        return feat[0], masks[0]
    return feat, masks


# ── Round-trip verification ───────────────────────────────────────────

@torch.no_grad()
def text_recovery_metrics(
    engine: InferenceEngine,
    text: str,
    img_size: int = 64,
    patch_size: int = 2,
    pad: str = 'space',
    channels: int = 3,
) -> Dict[str, Any]:
    """Run text → image → model recon → image → text and report byte fidelity.

    Returns dict with ``n_real_bytes``, ``recon_mse``, ``real_byte_acc``,
    ``real_byte_l1``, ``recon_text``. The model can only encode bytes it
    was trained to encode; this confirms the round-trip works on the
    specific text under test before any similarity comparison.
    """
    from geolip_svae.train import ByteTrigramDataset

    target = img_size * img_size * channels
    chunk, n_real = _padded_bytes(text, target, pad)
    img = torch.from_numpy(
        ByteTrigramDataset.bytes_to_image(chunk, img_size, patch_size, channels)
    )
    device = next(engine.model.parameters()).device
    out = engine.reconstruct(img.unsqueeze(0).to(device), mode='direct')
    recon = out['recon'].cpu()
    recon_mse = float(out['mse_per_image'][0]) if 'mse_per_image' in out \
        else float('nan')

    recon_bytes = ByteTrigramDataset.image_to_bytes(recon, patch_size, channels)
    flat = recon_bytes.reshape(-1).numpy().astype(np.uint8)
    orig = chunk.astype(np.uint8)

    if n_real == 0:
        return dict(
            n_real_bytes=0, recon_mse=recon_mse,
            real_byte_acc=float('nan'), real_byte_l1=float('nan'),
            recon_text='',
        )
    eq = (flat[:n_real] == orig[:n_real])
    return dict(
        n_real_bytes=int(n_real),
        recon_mse=recon_mse,
        real_byte_acc=float(eq.mean()),
        real_byte_l1=float(np.abs(
            flat[:n_real].astype(np.int32) - orig[:n_real].astype(np.int32)
        ).mean()),
        recon_text=bytes(flat[:n_real]).decode('utf-8', errors='replace'),
    )


# ── Per-patch similarity ──────────────────────────────────────────────

def per_patch_similarity(
    feat_a: torch.Tensor,
    mask_a: torch.Tensor,
    feat_b: torch.Tensor,
    mask_b: torch.Tensor,
    agg: str = 'best_match',
) -> float:
    """Cosine similarity between two texts' per-patch features.

    Subsets each side to its real patches, computes the pairwise
    ``[K_a, K_b]`` cosine matrix, aggregates to a scalar via ``agg``:

    - ``'best_match'`` (default): symmetric mean of row-max + col-max.
      Hausdorff-like; tolerates length mismatch. Preserves identity:
      similarity(x, x) = 1.
    - ``'aligned'``: per-position cosine; preserves identity. Requires
      ``K_a == K_b`` (raises otherwise).
    - ``'patch_mean'``: mean over all entries of the [K_a, K_b] matrix.
      Order-agnostic content-distribution similarity. Does NOT preserve
      identity (similarity(x, x) < 1 whenever any two patches of x
      differ from each other) — use only when that's what you want.
    """
    if agg not in AGG_METHODS:
        raise ValueError(f"agg={agg!r} not in {AGG_METHODS}")

    real_a = feat_a[mask_a]
    real_b = feat_b[mask_b]
    if real_a.numel() == 0 or real_b.numel() == 0:
        return float('nan')

    a = real_a / real_a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = real_b / real_b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = a @ b.T                                     # [K_a, K_b]

    if agg == 'patch_mean':
        return float(sim.mean())
    if agg == 'best_match':
        return float((sim.amax(dim=1).mean() + sim.amax(dim=0).mean()) / 2)
    if real_a.shape[0] != real_b.shape[0]:
        raise ValueError(
            f"agg='aligned' requires K_a == K_b; got "
            f"{real_a.shape[0]} vs {real_b.shape[0]}"
        )
    return float((a * b).sum(dim=-1).mean())


# ── Orchestrator ──────────────────────────────────────────────────────

@dataclass
class SentenceEncoder:
    """Bundles an ``InferenceEngine`` with text-side config.

    Holds ``(engine, img_size, patch_size, pad, channels)`` and delegates
    to the module-level free functions. Use the free functions directly
    if you'd rather not carry state. ``channels`` should match the
    model's ``channels`` attribute for in-distribution behavior.
    """
    engine: InferenceEngine
    img_size: int = 64
    patch_size: int = 2
    pad: str = 'space'
    channels: int = 3

    def text_to_image(self, text: str) -> torch.Tensor:
        return text_to_image(
            text, self.img_size, self.patch_size, self.pad, self.channels,
        )

    def image_to_text(
        self, image: torch.Tensor, n_bytes: Optional[int] = None,
    ) -> str:
        return image_to_text(image, self.patch_size, n_bytes, self.channels)

    def features(
        self,
        text: Union[str, Sequence[str]],
        mode: str = 'M',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return text_features(
            self.engine, text,
            self.img_size, self.patch_size, self.pad, mode, self.channels,
        )

    def recovery(self, text: str) -> Dict[str, Any]:
        return text_recovery_metrics(
            self.engine, text,
            self.img_size, self.patch_size, self.pad, self.channels,
        )

    def similarity(
        self, text_a: str, text_b: str,
        mode: str = 'M', agg: str = 'patch_mean',
    ) -> float:
        fa, ma = self.features(text_a, mode=mode)
        fb, mb = self.features(text_b, mode=mode)
        return per_patch_similarity(fa, ma, fb, mb, agg)

    def similarity_matrix(
        self, texts: Sequence[str],
        mode: str = 'M', agg: str = 'patch_mean',
    ) -> torch.Tensor:
        feats, masks = self.features(list(texts), mode=mode)
        n = len(texts)
        out = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                out[i, j] = per_patch_similarity(
                    feats[i], masks[i], feats[j], masks[j], agg,
                )
        return out


__all__ = [
    'PAD_STRATEGIES',
    'SIGNATURE_MODES',
    'AGG_METHODS',
    'text_to_image',
    'image_to_text',
    'text_real_patch_mask',
    'text_features',
    'text_recovery_metrics',
    'per_patch_similarity',
    'SentenceEncoder',
]

"""
geolip_svae.inference.scaling
==============================
Resolution-aware encoding and reconstruction.

Three modes:
    'direct' — feed images straight through model(images). Fast, but
               relies on the architecture tolerating non-native patch
               counts in its spatial-aggregation layers.
    'tile'   — pad to tile_size multiple, iterate over tiles, run model
               per tile, stitch outputs (and crop padding for recon).
               Always works at the cost of losing cross-tile context.
    'auto'   — try 'direct' first, fall back to 'tile' on failure. The
               diagnostic mode for "does this architecture handle larger
               inputs natively?"

Design principles (Abstract Powered Research framework v0):
    - No core limiters. patch_size, tile_size, batch_size all overridable.
    - Loud failures, never silent miscomputation.
    - Diagnostics in the return dict so callers can see what happened.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
# Padding helpers
# ════════════════════════════════════════════════════════════════════

def _pad_to_multiple(
    images: torch.Tensor,
    multiple: int,
    mode: str = 'reflect',
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad ``(B, C, H, W)`` so H and W are multiples of ``multiple``.

    Returns:
        padded:   (B, C, H', W') with H', W' multiples of ``multiple``
        pad_info: (pad_h, pad_w) — how much was added on each axis
    """
    if images.dim() != 4:
        raise ValueError(
            f"_pad_to_multiple expects (B, C, H, W); got shape {tuple(images.shape)}"
        )
    _, _, H, W = images.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return images, (0, 0)
    padded = F.pad(images, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (pad_h, pad_w)


def _crop_pad(
    images: torch.Tensor,
    pad_info: Tuple[int, int],
) -> torch.Tensor:
    """Inverse of ``_pad_to_multiple``. Crops bottom/right padding."""
    pad_h, pad_w = pad_info
    if pad_h == 0 and pad_w == 0:
        return images
    H_new = images.shape[-2] - pad_h
    W_new = images.shape[-1] - pad_w
    return images[..., :H_new, :W_new]


# ════════════════════════════════════════════════════════════════════
# Patch-size resolution
# ════════════════════════════════════════════════════════════════════

def _resolve_patch_size(
    model,
    patch_size_override: Optional[int],
) -> int:
    """Get the patch size to use, honoring user override.

    Args:
        model: any model with a ``patch_size`` attribute
        patch_size_override: if not None, use this; else read from model

    Honors the no-core-limiters principle: if the user wants to inference
    with a non-native patch size and provides it explicitly, that's their
    call. The model itself will probably fail loudly if it doesn't tolerate
    the override, which is the desired behavior.
    """
    if patch_size_override is not None:
        return int(patch_size_override)
    if hasattr(model, 'patch_size'):
        return int(model.patch_size)
    raise AttributeError(
        "Cannot resolve patch_size: model has no 'patch_size' attribute "
        "and no override was provided."
    )


# ════════════════════════════════════════════════════════════════════
# encode_at_scale
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_at_scale(
    model,
    images: torch.Tensor,
    tile_size: Optional[int] = None,
    mode: str = 'auto',
    batch_size: int = 16,
    patch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Encode images at arbitrary resolution.

    Args:
        model: any model whose forward returns ``dict`` with ``'svd'`` key
        images: (B, C, H, W)
        tile_size: native resolution for tile mode (default 64)
        mode: 'direct', 'tile', or 'auto'
        batch_size: forward-pass chunk size for tile mode
        patch_size: override the model's patch_size (default = model's own)

    Returns:
        dict with at minimum:
            M, S, S_orig (and U, Vt if the model exposes them)
            gh, gw         — full-resolution patch grid
            mode_used      — 'direct' or 'tile'
        Tile mode additionally includes:
            tile_size, pad_h, pad_w, n_tiles
    """
    model.eval()
    if tile_size is None:
        tile_size = 64
    if mode not in ('direct', 'tile', 'auto'):
        raise ValueError(f"mode must be 'direct'/'tile'/'auto', got {mode!r}")

    ps = _resolve_patch_size(model, patch_size)

    # ── direct ──
    if mode in ('direct', 'auto'):
        try:
            out = model(images)
            svd = dict(out['svd'])
            _, _, H, W = images.shape
            svd['gh'] = H // ps
            svd['gw'] = W // ps
            svd['mode_used'] = 'direct'
            return svd
        except Exception as e:
            if mode == 'direct':
                raise
            print(
                f"[encode_at_scale] direct mode failed "
                f"({type(e).__name__}: {str(e)[:120]}), "
                f"falling back to tile mode"
            )

    # ── tile ──
    return _encode_tile(
        model, images,
        tile_size=tile_size, batch_size=batch_size, patch_size=ps,
    )


@torch.no_grad()
def _encode_tile(
    model,
    images: torch.Tensor,
    tile_size: int,
    batch_size: int = 16,
    patch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Tile-and-stitch encode. Returns concatenated SVDs across all tiles."""
    device = next(model.parameters()).device
    images = images.to(device)
    ps = _resolve_patch_size(model, patch_size)

    padded, (pad_h, pad_w) = _pad_to_multiple(images, tile_size)
    B, C, H, W = padded.shape
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    gh_tile = tile_size // ps
    gw_tile = tile_size // ps

    # Collect tiles into a flat batch — (B * n_tiles_per_image, C, ts, ts)
    tile_batch = []
    for th in range(tiles_h):
        for tw in range(tiles_w):
            t = padded[
                :, :,
                th * tile_size:(th + 1) * tile_size,
                tw * tile_size:(tw + 1) * tile_size,
            ]
            tile_batch.append(t)
    tile_stack = torch.stack(tile_batch, dim=1)  # (B, n_tiles, C, ts, ts)
    n_tiles = tile_stack.shape[1]
    flat = tile_stack.reshape(B * n_tiles, C, tile_size, tile_size)

    # Forward in chunks
    all_M, all_S, all_S_orig = [], [], []
    all_U, all_Vt = [], []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start:start + batch_size]
        out = model(chunk)
        all_M.append(out['svd']['M'].cpu())
        all_S.append(out['svd']['S'].cpu())
        all_S_orig.append(out['svd']['S_orig'].cpu())
        if 'U' in out['svd']:
            all_U.append(out['svd']['U'].cpu())
        if 'Vt' in out['svd']:
            all_Vt.append(out['svd']['Vt'].cpu())

    M = torch.cat(all_M, dim=0)
    S = torch.cat(all_S, dim=0)
    S_orig = torch.cat(all_S_orig, dim=0)

    n_patches_per_tile = M.shape[1]
    V_dim = M.shape[2]
    D_dim = M.shape[3]

    # Reshape (B*n_tiles, n_patches_per_tile, V, D)
    #     → (B, n_tiles*n_patches_per_tile, V, D)
    M = M.reshape(B, n_tiles, n_patches_per_tile, V_dim, D_dim) \
         .reshape(B, n_tiles * n_patches_per_tile, V_dim, D_dim)
    S = S.reshape(B, n_tiles, n_patches_per_tile, -1) \
         .reshape(B, n_tiles * n_patches_per_tile, -1)
    S_orig = S_orig.reshape(B, n_tiles, n_patches_per_tile, -1) \
                   .reshape(B, n_tiles * n_patches_per_tile, -1)

    out_dict: Dict[str, Any] = {
        'M': M,
        'S': S,
        'S_orig': S_orig,
        'gh': tiles_h * gh_tile,
        'gw': tiles_w * gw_tile,
        'mode_used': 'tile',
        'tile_size': tile_size,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'n_tiles': n_tiles,
    }
    if all_U:
        out_dict['U'] = torch.cat(all_U, dim=0)
    if all_Vt:
        out_dict['Vt'] = torch.cat(all_Vt, dim=0)
    return out_dict


# ════════════════════════════════════════════════════════════════════
# reconstruct_at_scale
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct_at_scale(
    model,
    images: torch.Tensor,
    tile_size: Optional[int] = None,
    mode: str = 'auto',
    batch_size: int = 16,
    patch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Reconstruct images at arbitrary resolution.

    Args:
        model: any model whose forward returns ``dict`` with ``'recon'``
        images: (B, C, H, W)
        tile_size: native resolution for tile mode (default 64)
        mode: 'direct', 'tile', or 'auto'
        batch_size: chunk size for tile mode
        patch_size: override the model's patch_size

    Returns:
        dict with:
            recon: (B, C, H, W) reconstruction at original input size
            mode_used: 'direct' or 'tile'
            mse_per_image: (B,) per-image MSE vs original
        Tile mode additionally includes:
            tile_size, pad_h, pad_w, n_tiles
    """
    model.eval()
    if tile_size is None:
        tile_size = 64
    if mode not in ('direct', 'tile', 'auto'):
        raise ValueError(f"mode must be 'direct'/'tile'/'auto', got {mode!r}")

    _resolve_patch_size(model, patch_size)  # validate early

    # ── direct ──
    if mode in ('direct', 'auto'):
        try:
            out = model(images)
            recon = out['recon']
            mse = ((recon - images.to(recon.device)) ** 2) \
                .mean(dim=[1, 2, 3])
            return {
                'recon': recon,
                'mode_used': 'direct',
                'mse_per_image': mse.cpu(),
            }
        except Exception as e:
            if mode == 'direct':
                raise
            print(
                f"[reconstruct_at_scale] direct mode failed "
                f"({type(e).__name__}: {str(e)[:120]}), "
                f"falling back to tile mode"
            )

    # ── tile ──
    device = next(model.parameters()).device
    images = images.to(device)
    padded, (pad_h, pad_w) = _pad_to_multiple(images, tile_size)
    B, C, H, W = padded.shape
    tiles_h = H // tile_size
    tiles_w = W // tile_size

    recon_padded = torch.zeros_like(padded)
    for th in range(tiles_h):
        for tw in range(tiles_w):
            slc_h = slice(th * tile_size, (th + 1) * tile_size)
            slc_w = slice(tw * tile_size, (tw + 1) * tile_size)
            tile = padded[:, :, slc_h, slc_w]
            recon_chunks = []
            for start in range(0, tile.shape[0], batch_size):
                chunk = tile[start:start + batch_size]
                recon_chunks.append(model(chunk)['recon'])
            recon_tile = torch.cat(recon_chunks, dim=0)
            recon_padded[:, :, slc_h, slc_w] = recon_tile

    recon = _crop_pad(recon_padded, (pad_h, pad_w))
    mse = ((recon - images) ** 2).mean(dim=[1, 2, 3])

    return {
        'recon': recon,
        'mode_used': 'tile',
        'mse_per_image': mse.cpu(),
        'tile_size': tile_size,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'n_tiles': tiles_h * tiles_w,
    }
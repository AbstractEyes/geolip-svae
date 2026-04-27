"""
geolip_svae.inference.loading
==============================
Checkpoint resolution and model loading.

PatchSVAE-only as of the inference framework rebuild (scratchpad 000107).
The earlier ``PatchSVAEv2`` variant has been removed entirely; legacy v2
checkpoints raise a clear error rather than silently mis-loading.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import torch

from geolip_svae.model import PatchSVAE


# ── HuggingFace Repository ──────────────────────────────────────────

HF_REPO = "AbstractPhil/geolip-SVAE"

VERSIONS: Dict[str, str] = {
    # ── Fresnel (images) ──
    'v12_imagenet128':        'Fresnel-small 128×128 (ImageNet, 50 ep, MSE=0.0000734)',
    'v13_imagenet256':        'Fresnel-base 256×256 (ImageNet, 20 ep, MSE=0.000061)',
    'v19_fresnel_tiny':       'Fresnel-tiny 64×64 (TinyImageNet, 300 ep)',
    'v50_fresnel_64':         'Fresnel v50 64×64 (clean ImageNet, D=4, streaming, MSE=5e-6)',

    # ── Johanna (noise, D=16) ──
    'v14_noise':              'Johanna-small Gaussian 128×128 (200 ep)',
    'v16_johanna_omega':      'Johanna-small omega 128×128 (16 types, 380 ep, MSE=0.008)',
    'v18_johanna_curriculum': 'Johanna-tiny curriculum 64×64 (16 types, 300 ep)',
    'v20_johanna_base':       'Johanna-base 256×256 (scheduled curriculum, 60 ep)',

    # ── Alexandria (text) ──
    'v22_alexandria_small':   'Alexandria-small 128×128 (Wikipedia text, 100 ep)',

    # ── Grandmaster (denoiser) ──
    'v30_grandmaster':        'Grandmaster 128×128 (ImageNet, Johanna→denoiser, 50 ep)',

    # ── Freckles (D=4, 4×4 patches) ──
    'v40_freckles_noise':     'Freckles 64×64 (16 noise types, 100 ep, MSE=5e-6, 2.5M params)',
}


def list_versions() -> None:
    """Print available named model versions."""
    print("Available geolip-SVAE versions:")
    print(f"  {'Version':<28s} Description")
    print("-" * 72)
    for k, v in VERSIONS.items():
        print(f"  {k:<28s} {v}")


# ── Internal: checkpoint path resolution ─────────────────────────────

def _resolve_checkpoint(
    hf_version: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    hf_file: Optional[str] = None,
    repo_id: str = HF_REPO,
) -> str:
    """Resolve a checkpoint path from one of three sources.

    Priority: explicit local path > explicit hf_file > named hf_version.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
    if hf_file:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo_id, filename=hf_file, repo_type='model',
        )
    if hf_version:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo_id,
            filename=f'{hf_version}/checkpoints/best.pt',
            repo_type='model',
        )
    raise ValueError(
        "Provide hf_version, hf_file, or checkpoint_path"
    )


# ── Public: model loading ────────────────────────────────────────────

class UnsupportedCheckpointError(RuntimeError):
    """Raised when a checkpoint targets a removed/unsupported model variant."""


def load_model(
    hf_version: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    hf_file: Optional[str] = None,
    device: Optional[str] = None,
    repo_id: str = HF_REPO,
) -> Tuple[PatchSVAE, Dict[str, Any]]:
    """Load a PatchSVAE checkpoint.

    Args:
        hf_version: named version (e.g. 'v50_fresnel_64') — loads best.pt
        checkpoint_path: local .pt file path
        hf_file: specific filename in HF repo (overrides hf_version's default)
        device: 'cuda' / 'cpu' / None (auto-detect)
        repo_id: HuggingFace repository ID

    Returns:
        (model, cfg) where model is on `device` in eval mode and cfg is
        the checkpoint config dict augmented with these fields:
            cfg['_epoch']      — checkpoint epoch (or None)
            cfg['_test_mse']   — checkpoint test/val MSE (or None)
            cfg['_path']       — resolved local checkpoint path
            cfg['_model_type'] — always 'v1' as of the rebuild

    Raises:
        UnsupportedCheckpointError: if the checkpoint config declares
            ``model_type='v2'``. PatchSVAEv2 was removed in the inference
            framework rebuild (scratchpad 000107). Re-train as v1.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = _resolve_checkpoint(hf_version, checkpoint_path, hf_file, repo_id)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    model_type = cfg.get('model_type', 'v1')
    if model_type == 'v2':
        raise UnsupportedCheckpointError(
            f"Checkpoint at {path} declares model_type='v2'. "
            f"PatchSVAEv2 has been removed from geolip-svae; this "
            f"checkpoint cannot be loaded. If you need this checkpoint, "
            f"re-train it as a v1 PatchSVAE configuration."
        )

    model = PatchSVAE(
        V=cfg['V'],
        D=cfg['D'],
        ps=cfg['patch_size'],
        hidden=cfg['hidden'],
        depth=cfg['depth'],
        n_cross=cfg['n_cross_layers'],
        n_heads=cfg.get('n_heads', None),
        smooth_mid=cfg.get('smooth_mid', None),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device).eval()

    cfg['_epoch'] = ckpt.get('epoch')
    cfg['_test_mse'] = ckpt.get('test_mse') or ckpt.get('val_mse')
    cfg['_path'] = path
    cfg['_model_type'] = 'v1'
    return model, cfg
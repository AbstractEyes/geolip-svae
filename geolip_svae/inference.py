"""
Inference utilities for geolip-svae.
=====================================
Load checkpoints from HuggingFace or local paths,
encode images to omega tokens, decode back.

Usage:
    from geolip_svae import load_model, encode, decode, reconstruct

    model, cfg = load_model(hf_version='v13_imagenet256')
    omega = encode(model, images)        # dict with S, U, Vt, M
    recon = reconstruct(model, images)   # (B, 3, H, W)

    # Or from specific checkpoint:
    model, cfg = load_model(hf_file='v20_johanna_base/checkpoints/epoch_0030.pt')
"""

import os
import torch
import torch.nn.functional as F
from geolip_svae.model import PatchSVAE


# ── HuggingFace Repository ──────────────────────────────────────

HF_REPO = "AbstractPhil/geolip-SVAE"

# Known versions and their descriptions
VERSIONS = {
    'v12_imagenet128':      'Fresnel-small 128×128 (ImageNet, 50 ep)',
    'v13_imagenet256':      'Fresnel-base 256×256 (ImageNet, 20 ep, MSE=0.000061)',
    'v14_noise':            'Johanna-small Gaussian 128×128 (200 ep)',
    'v16_johanna_omega':    'Johanna-small omega 128×128 (16 types, 59 ep)',
    'v18_johanna_curriculum': 'Johanna-tiny curriculum 64×64 (16 types, 300 ep)',
    'v19_fresnel_tiny':     'Fresnel-tiny 64×64 (TinyImageNet)',
    'v20_johanna_base':     'Johanna-base 256×256 (scheduled curriculum, 30 ep)',
    'v22_alexandria_small': 'Alexandria-small 128×128 (Wikipedia text, 100 ep)',
}


def list_versions():
    """Print available model versions."""
    print("Available geolip-SVAE versions:")
    print(f"  {'Version':<28s} Description")
    print("-" * 72)
    for k, v in VERSIONS.items():
        print(f"  {k:<28s} {v}")


# ── Model Loading ────────────────────────────────────────────────

def load_model(hf_version: str = None, checkpoint_path: str = None,
               hf_file: str = None, device: str = None,
               repo_id: str = HF_REPO) -> tuple:
    """Load a PatchSVAE model from checkpoint.

    Args:
        hf_version: named version (e.g. 'v13_imagenet256') — loads best.pt
        checkpoint_path: local .pt file path
        hf_file: specific file in HF repo (e.g. 'v20.../epoch_0030.pt')
        device: 'cuda', 'cpu', or None (auto-detect)
        repo_id: HuggingFace repository ID

    Returns:
        model: PatchSVAE on device, eval mode
        cfg: dict of model config from checkpoint

    Example:
        model, cfg = load_model(hf_version='v13_imagenet256')
        model, cfg = load_model(hf_file='v18_johanna_curriculum/checkpoints/epoch_0300.pt')
        model, cfg = load_model(checkpoint_path='/path/to/best.pt')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resolve checkpoint path
    if checkpoint_path and os.path.exists(checkpoint_path):
        path = checkpoint_path
    elif hf_file:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=hf_file,
                               repo_type='model')
    elif hf_version:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id,
                               filename=f'{hf_version}/checkpoints/best.pt',
                               repo_type='model')
    else:
        raise ValueError("Provide hf_version, hf_file, or checkpoint_path")

    # Load checkpoint
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    # Build model
    model = PatchSVAE(
        V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
        hidden=cfg['hidden'], depth=cfg['depth'],
        n_cross=cfg['n_cross_layers'],
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device).eval()

    # Attach metadata
    cfg['_epoch'] = ckpt.get('epoch')
    cfg['_test_mse'] = ckpt.get('test_mse')
    cfg['_path'] = path

    return model, cfg


# ── Encode / Decode ──────────────────────────────────────────────

@torch.no_grad()
def encode(model: PatchSVAE, images: torch.Tensor) -> dict:
    """Encode images to omega tokens.

    Args:
        model: PatchSVAE in eval mode
        images: (B, 3, H, W) — normalized, on same device as model

    Returns:
        dict with:
            S:      (B, N, D)     omega tokens (coordinated singular values)
            S_orig: (B, N, D)     raw singular values (pre-coordination)
            U:      (B, N, V, D)  left singular vectors
            Vt:     (B, N, D, D)  right singular vectors
            M:      (B, N, V, D)  sphere-normalized encoding matrix
    """
    model.eval()
    from geolip_svae.model import extract_patches
    patches, gh, gw = extract_patches(images, model.patch_size)
    svd = model.encode_patches(patches)
    svd['gh'] = gh
    svd['gw'] = gw
    return svd


@torch.no_grad()
def decode(model: PatchSVAE, svd: dict) -> torch.Tensor:
    """Decode omega tokens back to images.

    Args:
        model: PatchSVAE in eval mode
        svd: dict from encode() — must contain U, S, Vt, gh, gw

    Returns:
        images: (B, 3, H, W)
    """
    model.eval()
    from geolip_svae.model import stitch_patches
    decoded = model.decode_patches(svd['U'], svd['S'], svd['Vt'])
    recon = stitch_patches(decoded, svd['gh'], svd['gw'], model.patch_size)
    return model.boundary_smooth(recon)


@torch.no_grad()
def reconstruct(model: PatchSVAE, images: torch.Tensor) -> torch.Tensor:
    """Full round-trip: encode → decode.

    Args:
        model: PatchSVAE in eval mode
        images: (B, 3, H, W) — normalized, on same device as model

    Returns:
        recon: (B, 3, H, W) — reconstructed images
    """
    model.eval()
    return model(images)['recon']


# ── Batched Inference ────────────────────────────────────────────

@torch.no_grad()
def batched_forward(model: PatchSVAE, images: torch.Tensor,
                    max_batch: int = 16) -> dict:
    """Forward pass in chunks to avoid OOM on large batches.

    Args:
        model: PatchSVAE in eval mode
        images: (N, 3, H, W) — can be on CPU, will be moved per chunk
        max_batch: maximum batch size per forward pass

    Returns:
        dict with recon, S, S_orig, M — all on CPU
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
"""
Inference utilities for geolip-svae.
=====================================
Load checkpoints from HuggingFace or local paths,
encode images to omega tokens, decode back.

Supports both v1 (PatchSVAE) and v2 (PatchSVAEv2) models.

Usage:
    from geolip_svae import load_model, encode, decode, reconstruct

    # v1 model
    model, cfg = load_model(hf_version='v50_fresnel_64')
    omega = encode(model, images)
    recon = reconstruct(model, images)

    # v2 model (conduit-forced decoder)
    from geolip_svae import load_model_v2
    v2, cfg = load_model_v2(hf_version='v50_fresnel_64')  # init from v1 weights
    out = v2(images)  # conduit telemetry in out['svd']['conduit_packet']

    # Or load a trained v2 checkpoint directly
    v2, cfg = load_model(hf_version='v51_fresnel_conduit')
"""

import os
import torch
import torch.nn.functional as F
from geolip_svae.model import PatchSVAE
from geolip_svae.model_v2 import PatchSVAEv2


# ── HuggingFace Repository ──────────────────────────────────────

HF_REPO = "AbstractPhil/geolip-SVAE"

# Known versions and their descriptions
VERSIONS = {
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


def list_versions():
    """Print available model versions."""
    print("Available geolip-SVAE versions:")
    print(f"  {'Version':<28s} Description")
    print("-" * 72)
    for k, v in VERSIONS.items():
        print(f"  {k:<28s} {v}")


# ── Model Loading ────────────────────────────────────────────────

def _resolve_checkpoint(hf_version=None, checkpoint_path=None,
                        hf_file=None, repo_id=HF_REPO):
    """Resolve checkpoint path from various sources."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
    elif hf_file:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=hf_file,
                               repo_type='model')
    elif hf_version:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id,
                               filename=f'{hf_version}/checkpoints/best.pt',
                               repo_type='model')
    else:
        raise ValueError("Provide hf_version, hf_file, or checkpoint_path")


def load_model(hf_version: str = None, checkpoint_path: str = None,
               hf_file: str = None, device: str = None,
               repo_id: str = HF_REPO) -> tuple:
    """Load a PatchSVAE model from checkpoint.

    Automatically detects v1 vs v2 from checkpoint config.

    Args:
        hf_version: named version (e.g. 'v50_fresnel_64') — loads best.pt
        checkpoint_path: local .pt file path
        hf_file: specific file in HF repo
        device: 'cuda', 'cpu', or None (auto-detect)
        repo_id: HuggingFace repository ID

    Returns:
        model: PatchSVAE or PatchSVAEv2 on device, eval mode
        cfg: dict of model config from checkpoint
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = _resolve_checkpoint(hf_version, checkpoint_path, hf_file, repo_id)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    # Detect model type from config
    model_type = cfg.get('model_type', 'v1')

    if model_type == 'v2':
        model = PatchSVAEv2(
            V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
            hidden=cfg['hidden'], depth=cfg['depth'],
            n_cross=cfg['n_cross_layers'],
            n_heads=cfg.get('n_heads', None),
            smooth_mid=cfg.get('smooth_mid', None),
        )
    else:
        model = PatchSVAE(
            V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
            hidden=cfg['hidden'], depth=cfg['depth'],
            n_cross=cfg['n_cross_layers'],
            n_heads=cfg.get('n_heads', None),
            smooth_mid=cfg.get('smooth_mid', None),
        )

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(device).eval()

    # Attach metadata
    cfg['_epoch'] = ckpt.get('epoch')
    cfg['_test_mse'] = ckpt.get('test_mse')
    cfg['_path'] = path
    cfg['_model_type'] = model_type

    return model, cfg


def load_model_v2(hf_version: str = None, checkpoint_path: str = None,
                  hf_file: str = None, device: str = None,
                  repo_id: str = HF_REPO,
                  freeze_encoder: bool = False) -> tuple:
    """Load a v1 checkpoint and initialize PatchSVAEv2 from it.

    Copies encoder, cross-attention, and boundary smoothing from v1.
    Decoder is NEW (conduit-forced, random init). Must be retrained.

    Args:
        hf_version: v1 checkpoint to initialize from
        checkpoint_path: local v1 .pt file
        hf_file: specific v1 file in HF repo
        device: 'cuda', 'cpu', or None
        repo_id: HuggingFace repository ID
        freeze_encoder: if True, freeze all encoder parameters

    Returns:
        model: PatchSVAEv2 on device (train mode — decoder needs training)
        cfg: dict of model config
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load v1 model first
    v1_model, cfg = load_model(
        hf_version=hf_version, checkpoint_path=checkpoint_path,
        hf_file=hf_file, device=device, repo_id=repo_id)

    # Convert to v2
    v2_model = PatchSVAEv2.from_v1(v1_model)
    v2_model = v2_model.to(device)

    # Optionally freeze encoder
    if freeze_encoder:
        for name, param in v2_model.named_parameters():
            if name.startswith(('enc_in', 'enc_blocks', 'enc_out')):
                param.requires_grad = False
        n_frozen = sum(1 for p in v2_model.parameters() if not p.requires_grad)
        n_total = sum(1 for p in v2_model.parameters())
        print(f"  Encoder frozen: {n_frozen}/{n_total} parameter groups")

    cfg['_model_type'] = 'v2'
    cfg['_v1_source'] = cfg.get('_path', hf_version)

    return v2_model, cfg


# ── Encode / Decode ──────────────────────────────────────────────

@torch.no_grad()
def encode(model, images: torch.Tensor) -> dict:
    """Encode images to omega tokens. Works for both v1 and v2.

    Args:
        model: PatchSVAE or PatchSVAEv2 in eval mode
        images: (B, 3, H, W) — normalized, on same device as model

    Returns:
        dict with S, S_orig, U, Vt, M, gh, gw
        (v2 also includes conduit_packet)
    """
    model.eval()
    from geolip_svae.model import extract_patches
    patches, gh, gw = extract_patches(images, model.patch_size)
    svd = model.encode_patches(patches)
    svd['gh'] = gh
    svd['gw'] = gw
    return svd


@torch.no_grad()
def decode(model, svd: dict) -> torch.Tensor:
    """Decode omega tokens back to images. Works for both v1 and v2.

    Args:
        model: PatchSVAE or PatchSVAEv2 in eval mode
        svd: dict from encode()

    Returns:
        images: (B, 3, H, W)
    """
    model.eval()
    from geolip_svae.model import stitch_patches

    if isinstance(model, PatchSVAEv2):
        decoded = model.decode_patches(
            svd['U'], svd['S'], svd['Vt'], svd['conduit_packet'])
    else:
        decoded = model.decode_patches(svd['U'], svd['S'], svd['Vt'])

    recon = stitch_patches(decoded, svd['gh'], svd['gw'], model.patch_size)
    return model.boundary_smooth(recon)


@torch.no_grad()
def reconstruct(model, images: torch.Tensor) -> torch.Tensor:
    """Full round-trip: encode → decode. Works for both v1 and v2.

    Args:
        model: PatchSVAE or PatchSVAEv2 in eval mode
        images: (B, 3, H, W)

    Returns:
        recon: (B, 3, H, W)
    """
    model.eval()
    return model(images)['recon']


# ── Batched Inference ────────────────────────────────────────────

@torch.no_grad()
def batched_forward(model, images: torch.Tensor,
                    max_batch: int = 16) -> dict:
    """Forward pass in chunks to avoid OOM on large batches.

    Args:
        model: PatchSVAE or PatchSVAEv2 in eval mode
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
"""
SVAE Unified Trainer (v2)
==========================
Single entry point for all model variants. Replaces the previous train.py.

    Fresnel    (images):   python -m geolip_svae.train --preset fresnel_base
    Johanna    (noise):    python -m geolip_svae.train --preset johanna_base
    Alexandria (text):     python -m geolip_svae.train --preset alexandria_small
    Freckles   (D=4 noise):python -m geolip_svae.train --preset freckles_64
    Fresnel-64 (D=4 imgs): python -m geolip_svae.train --preset fresnel_64
    H2-64      (sphere):   python -m geolip_svae.train --preset h2_64_single
    BinTree    (substrate):python -m geolip_svae.train --preset bintree_proto

    Streaming continuation:  python -m geolip_svae.train_streaming \
                                 --hf-version v50_fresnel_64

What v2 adds over v1
--------------------
  * Architecture kwargs (linear_readout, svd_mode, match_params, smooth_mid)
    plumbed through preset dict to PatchSVAE — enables h2-class architectures
    that v1 could not build at all.
  * johanna_F diagnostics restored: epoch_max_grad, per-layer alpha mean/std,
    cv_in_band boolean, full per-step `history` list dumped to final_report.json.
  * Mid-epoch reporting cadence via `report_every` (not just per-epoch).
  * allowed_types filter for noise datasets — Gaussian-only foundation,
    custom subset, or all 16.
  * BinaryTreeDataset + bit-recovery metric for the substrate prototype.

Existing behavior preserved
---------------------------
  * Pretrained loading from HF
  * Curriculum (patience-based and scheduled tier unlocks)
  * HF checkpoint + TB upload
  * All five existing presets (fresnel_*, johanna_*, alexandria_*) still run.

Listed presets:
    fresnel_tiny       TinyImageNet 64x64,  300 ep
    fresnel_small      ImageNet-128 128x128, 50 ep
    fresnel_base       ImageNet-256 256x256, 20 ep
    johanna_tiny       Curriculum noise 64x64, 300 ep
    johanna_small      Omega noise 128x128, 200 ep (pretrained from Gaussian)
    johanna_base       Scheduled noise 256x256, 30 ep
    alexandria_small   Wikipedia text 128x128, 100 ep (pretrained from Johanna)
    freckles_64        Omega noise 64x64, 100 ep (D=4 noise specialist, 2.55M params)
    freckles_256       Omega noise 256x256, 1 ep (init from freckles_64)
    freckles_512       Omega noise 512x512, 1 ep (init from freckles_256)
    fresnel_64         TinyImageNet 64x64 with Freckles geometry (D=4, 2.55M params)
    h2_64_single       H2-class single battery, gaussian only — for reproducing
                       individual h2-64 banks
    bintree_proto      Binary-tree substrate prototype on h2-64 architecture

For long-running continuation of any trained model on streaming random crops
(the "sublens perspective" mode that produced v50_fresnel_64's 140M+ images),
see `geolip_svae.train_streaming`.
"""

import os
import math
import json
import time
import argparse
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from geolip_svae.model import PatchSVAE, cv_of

# ── HuggingFace auth ─────────────────────────────────────────────────

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════════

# All architecture kwargs are explicit; presets opt into h2-class via
# linear_readout=True + svd_mode='none' + match_params=True. Defaults
# match the original PatchSVAE defaults if omitted.

PRESETS: Dict[str, Dict[str, Any]] = {
    # ── Fresnel (images) ──
    'fresnel_tiny': dict(
        # Architecture
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        # Training
        dataset='tiny_imagenet', img_size=64, batch_size=256,
        lr=1e-4, epochs=300, target_cv=0.2915,
        hf_version='v19_fresnel_tiny', save_every=10,
    ),
    'fresnel_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_128', img_size=128, batch_size=128,
        lr=1e-4, epochs=50, target_cv=0.2915,
        hf_version='v12_imagenet128', save_every=1,
    ),
    'fresnel_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_256', img_size=256, batch_size=64,
        lr=1e-4, epochs=20, target_cv=0.2915,
        hf_version='v13_imagenet256', save_every=1,
    ),

    # ── Johanna (noise) ──
    'johanna_tiny': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='curriculum_noise', img_size=64, batch_size=512,
        lr=3e-4, epochs=300, target_cv=0.125,
        hf_version='v18_johanna_curriculum', save_every=25,
        curriculum='patience',
    ),
    'johanna_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='omega_noise', img_size=128, batch_size=128,
        lr=1e-4, epochs=200, target_cv=0.125,
        hf_version='v16_johanna_omega', save_every=10,
        pretrained='v14_noise/checkpoints/epoch_0200.pt',
    ),
    'johanna_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='scheduled_noise', img_size=256, batch_size=64,
        lr=1e-4, epochs=30, target_cv=0.2915,
        hf_version='v20_johanna_base', save_every=5,
        curriculum='scheduled', tier_schedule={5: 1, 8: 2, 10: 3, 12: 4},
    ),

    # ── Alexandria (text) ──
    'alexandria_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='wikipedia', img_size=128, batch_size=128,
        lr=1e-4, epochs=100, target_cv=0.2915,
        hf_version='v22_alexandria_small', save_every=10,
        pretrained='v16_johanna_omega/checkpoints/best.pt',
        ds_size=200000, val_size=5000,
    ),

    # ── Freckles (D=4 noise specialist, 2.55M params) ──
    # Resolution-invariant by construction: cross-attn weights dimensioned by
    # D=4, not N. Same weights work at any patch count. v41 inits from v40,
    # v42 inits from v41 — cumulative resolution-transfer chain.
    'freckles_64': dict(
        # Freckles architecture
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Training: 16-type omega noise at 64x64 (256 patches) for 100 ep
        dataset='omega_noise', img_size=64, batch_size=256,
        lr=1e-4, epochs=100, target_cv=0.125,  # historical value, predates 0.20-0.23 band
        hf_version='v40_freckles_noise', save_every=10,
        ds_size=500_000, val_size=10_000,
        report_every=500,
    ),
    'freckles_256': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Resolution transfer test: v40 weights, fine-tune at 256x256 (4096 patches).
        # 1 epoch is enough — spectrum stays within 0.4% of v40.
        dataset='omega_noise', img_size=256, batch_size=64,
        lr=1e-5, epochs=1, target_cv=0.125,
        hf_version='v41_freckles_256', save_every=1,
        pretrained='v40_freckles_noise/checkpoints/best.pt',
        ds_size=200_000, val_size=2_000,
        report_every=200,
    ),
    'freckles_512': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        # Continued resolution transfer: v41 weights, fine-tune at 512x512.
        dataset='omega_noise', img_size=512, batch_size=16,
        lr=1e-5, epochs=1, target_cv=0.125,
        hf_version='v42_freckles_512', save_every=1,
        pretrained='v41_freckles_256/checkpoints/best.pt',
        ds_size=80_000, val_size=1_000,
        report_every=100,
    ),

    # ── Fresnel-64 (D=4 ImageNet specialist, Freckles geometry) ──
    # Same architecture as Freckles, trained on ImageNet crops instead of noise.
    # 297M unique crops; spectrum locks at step 17,500. Identical attractor to
    # Freckles v40 → universal manifold across modalities.
    'fresnel_64': dict(
        V=48, D=4, patch_size=4, hidden=384, depth=4, n_cross=2,
        dataset='tiny_imagenet', img_size=64, batch_size=256,
        lr=1e-4, epochs=100, target_cv=0.125,
        hf_version='v50_fresnel_64', save_every=10,
        report_every=500,
    ),
    # Note: the v50_fresnel_64 model was *also* run through 140M+ random 64x64
    # crops of ImageNet-256 via train_streaming.py — see that module for the
    # continuation trainer. The "fresnel_64_256" name on HF refers to that
    # streaming continuation (sublens perspective, not a 256x256 finetune).

    # ── H2-class (sphere-solver, the architecture used by h2-64 batteries) ──
    'h2_64_single': dict(
        # H2_linear_matched architecture
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training — gaussian only by default (foundation)
        dataset='omega_noise', img_size=64, batch_size=128,
        lr=1e-3, epochs=20, target_cv=0.215,  # midpoint of CV band
        allowed_types=[0],
        hf_version='h2_64_repro_single', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Diagnostics cadence
        report_every=200,
    ),

    # ── BinaryTree substrate prototype ──
    'bintree_proto': dict(
        # H2-64 architecture exactly
        V=32, D=4, patch_size=4, hidden=64, depth=1, n_cross=1, n_heads=4,
        smooth_mid=16,
        linear_readout=True, svd_mode='none', match_params=True,
        # Training on i.i.d. depth-4 binary trees (BFS-encoded, ±1 floats)
        dataset='binary_tree', img_size=16, batch_size=256,
        lr=1e-3, epochs=20, target_cv=0.215,
        hf_version='bintree_proto_v1', save_every=5,
        ds_size=200_000, val_size=2_000,
        # Tree config
        tree_depth=4,
        # Diagnostics cadence
        report_every=200,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# NOISE DATASETS
# ═══════════════════════════════════════════════════════════════════

NOISE_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_scaled', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse',
    8: 'block', 9: 'gradient', 10: 'checkerboard', 11: 'mixed',
    12: 'structural', 13: 'cauchy', 14: 'exponential', 15: 'laplace',
}

TIERS = {
    0: [0],              # Gaussian
    1: [4, 5, 8, 9],     # Pink, Brown, Block, Gradient
    2: [1, 2, 10, 11],   # Uniform, Scaled, Checkerboard, Mixed
    3: [3, 14, 15, 7],   # Poisson, Exponential, Laplace, Sparse
    4: [13, 6, 12],      # Cauchy, Salt-pepper, Structural
}


def _pink_noise(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(
        S / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww)
    )


def _brown_noise(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(
        S / (fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww)
    )


def _generate_noise(noise_type, s, rng):
    if noise_type == 0:
        return torch.randn(3, s, s)
    elif noise_type == 1:
        return torch.rand(3, s, s) * 2 - 1
    elif noise_type == 2:
        return (torch.rand(3, s, s) - 0.5) * 4
    elif noise_type == 3:
        lam = rng.uniform(0.5, 20.0)
        return torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0
    elif noise_type == 4:
        img = _pink_noise((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 5:
        img = _brown_noise((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 6:
        img = torch.where(torch.rand(3, s, s) > 0.5,
                          torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2)
        return img + torch.randn(3, s, s) * 0.1
    elif noise_type == 7:
        return torch.randn(3, s, s) * (torch.rand(3, s, s) > 0.9).float() * 3
    elif noise_type == 8:
        b = rng.randint(2, 16)
        sm = torch.randn(3, s // b + 1, s // b + 1)
        return F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif noise_type == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        a = rng.uniform(0, 2 * math.pi)
        return ((math.cos(a) * gx + math.sin(a) * gy)
                .unsqueeze(0).expand(3, -1, -1)
                + torch.randn(3, s, s) * 0.5)
    elif noise_type == 10:
        cs = rng.randint(2, 16)
        cy = torch.arange(s) // cs; cx = torch.arange(s) // cs
        return (((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float()
                .unsqueeze(0).expand(3, -1, -1) * 2 - 1
                + torch.randn(3, s, s) * 0.3)
    elif noise_type == 11:
        alpha = rng.uniform(0.2, 0.8)
        return alpha * torch.randn(3, s, s) + (1 - alpha) * (torch.rand(3, s, s) * 2 - 1)
    elif noise_type == 12:
        img = torch.zeros(3, s, s); h2 = s // 2
        img[:, :h2, :h2] = torch.randn(3, h2, h2)
        img[:, :h2, h2:] = torch.rand(3, h2, h2) * 2 - 1
        img[:, h2:, :h2] = _pink_noise((3, h2, h2)) / 2
        img[:, h2:, h2:] = torch.where(torch.rand(3, h2, h2) > 0.5,
                                         torch.ones(3, h2, h2),
                                         -torch.ones(3, h2, h2))
        return img
    elif noise_type == 13:
        return torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3)
    elif noise_type == 14:
        return torch.empty(3, s, s).exponential_(1.0) - 1.0
    elif noise_type == 15:
        u = torch.rand(3, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    return torch.randn(3, s, s)


class CurriculumNoiseDataset(torch.utils.data.Dataset):
    """Noise with tier-based type activation for Johanna curriculum training.

    `allowed_types` overrides the curriculum entirely if provided.
    """

    def __init__(self, size=500000, img_size=64, allowed_types=None):
        self.size = size
        self.img_size = img_size
        self._rng = np.random.RandomState(42)
        self._call_count = 0
        if allowed_types is not None:
            self.active_types = list(allowed_types)
            self.current_tier = -1  # locked
        else:
            self.active_types = list(TIERS[0])
            self.current_tier = 0

    def unlock_tier(self, tier):
        if self.current_tier == -1:
            return  # locked by allowed_types
        if tier in TIERS:
            for t in TIERS[tier]:
                if t not in self.active_types:
                    self.active_types.append(t)
            self.current_tier = tier

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.RandomState(int.from_bytes(os.urandom(4), 'big'))
            torch.manual_seed(int.from_bytes(os.urandom(4), 'big'))
        noise_type = self.active_types[idx % len(self.active_types)]
        img = _generate_noise(noise_type, self.img_size, self._rng).clamp(-4, 4)
        return img.float(), noise_type


class OmegaNoiseDataset(torch.utils.data.Dataset):
    """Noise types with optional `allowed_types` filter.

    Default (allowed_types=None) is all 16 types. Gaussian-only foundation
    runs pass allowed_types=[0]. Custom subsets are passed as iterables.
    """

    def __init__(self, size=1280000, img_size=128, allowed_types=None):
        self.size = size
        self.img_size = img_size
        self._rng = np.random.RandomState(42)
        self._call_count = 0
        self.active_types = (list(allowed_types) if allowed_types is not None
                              else list(range(16)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.RandomState(int.from_bytes(os.urandom(4), 'big'))
            torch.manual_seed(int.from_bytes(os.urandom(4), 'big'))
        noise_type = self.active_types[idx % len(self.active_types)]
        img = _generate_noise(noise_type, self.img_size, self._rng).clamp(-4, 4)
        return img.float(), noise_type


# ═══════════════════════════════════════════════════════════════════
# IMAGE DATASETS
# ═══════════════════════════════════════════════════════════════════

class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), item.get('label', 0)


def get_image_loaders(dataset_name, img_size, batch_size):
    """Returns (train_loader, test_loader, mean, std)."""
    from datasets import load_dataset

    if dataset_name == 'tiny_imagenet':
        ds = load_dataset('zh-plus/tiny-imagenet')
        mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['valid'], transform)

    elif dataset_name == 'imagenet_128':
        ds = load_dataset('benjamin-paine/imagenet-1k-128x128')
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['validation'], transform)

    elif dataset_name == 'imagenet_256':
        ds = load_dataset('benjamin-paine/imagenet-1k-256x256')
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['validation'], transform)

    else:
        raise ValueError(f"Unknown image dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, test_loader, mean, std


# ═══════════════════════════════════════════════════════════════════
# TEXT DATASET
# ═══════════════════════════════════════════════════════════════════

class WikiTextAsImage(torch.utils.data.Dataset):
    """Wikipedia text packed as (3, H, W) byte tensors. [0,255] → [-1,1]."""

    def __init__(self, size=200000, img_size=128, split='train'):
        self.size = size
        self.img_size = img_size
        self.n_bytes = 3 * img_size * img_size
        from datasets import load_dataset
        ds = load_dataset('wikimedia/wikipedia', '20231101.en',
                          split=split, streaming=True)
        target_bytes = min(size * self.n_bytes, 500_000_000)
        chunks, total = [], 0
        for article in ds:
            text = article['text']
            if text.strip():
                chunks.append(text)
                total += len(text)
            if total >= target_bytes:
                break
        self.raw_bytes = '\n'.join(chunks).encode('utf-8')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        max_start = max(0, len(self.raw_bytes) - self.n_bytes)
        start = torch.randint(0, max_start + 1, (1,)).item()
        chunk = self.raw_bytes[start:start + self.n_bytes]
        if len(chunk) < self.n_bytes:
            chunk = chunk + b'\x00' * (self.n_bytes - len(chunk))
        arr = np.frombuffer(chunk, dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).float()
        tensor = (tensor / 127.5) - 1.0
        return tensor.reshape(3, self.img_size, self.img_size), 0


# ═══════════════════════════════════════════════════════════════════
# BINARY TREE DATASET — substrate prototype
# ═══════════════════════════════════════════════════════════════════

class BinaryTreeDataset(torch.utils.data.Dataset):
    """Depth-d full binary trees, BFS-encoded into 4x4 RGB patches.

    Each 4x4 RGB patch (48 floats) holds:
      - First N_NODES floats: bit values (in {-1, +1}) at BFS positions
      - Remaining floats: zero padding
    where N_NODES = 2^(d+1) - 1.

    Image layout: (img_size // 4)^2 patches per "image", each independently
    sampled. Tree bits are i.i.d. Bernoulli(0.5).
    """

    PATCH_FLOATS = 48

    def __init__(self, size=200_000, img_size=16, depth=4, seed=42):
        self.size = size
        self.img_size = img_size
        self.depth = depth
        # Total nodes in full binary tree of depth d (counting both internal
        # and leaf nodes): root (depth 0) + 2 (depth 1) + 4 + ... + 2^d
        # = 2^(d+1) - 1.
        self.n_nodes = (2 ** (depth + 1)) - 1
        if self.n_nodes > self.PATCH_FLOATS:
            raise ValueError(
                f"depth={depth} produces {self.n_nodes} nodes, "
                f"exceeds patch capacity {self.PATCH_FLOATS}"
            )
        self.n_pad = self.PATCH_FLOATS - self.n_nodes
        if img_size % 4 != 0:
            raise ValueError(f"img_size must be divisible by 4, got {img_size}")
        self.gh = self.gw = img_size // 4
        self.n_patches = self.gh * self.gw
        self._base_seed = seed
        self._call_count = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    @staticmethod
    def bfs_layout(depth: int):
        """For each BFS index 0..n_nodes-1, return (level, position_within_level)."""
        layout = []
        for level in range(depth + 1):
            for pos in range(2 ** level):
                layout.append((level, pos))
        return layout

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.default_rng(int.from_bytes(os.urandom(4), 'big'))

        # Sample n_patches independent trees
        bits = self._rng.integers(0, 2, size=(self.n_patches, self.n_nodes))
        bits = bits.astype(np.float32) * 2.0 - 1.0  # 0/1 → -1/+1

        # Pad to 48 floats per patch
        padded = np.zeros((self.n_patches, self.PATCH_FLOATS), dtype=np.float32)
        padded[:, :self.n_nodes] = bits

        # Reshape to (n_patches, 3, 4, 4)  — channels-first 3x4x4 layout
        patches = padded.reshape(self.n_patches, 3, 4, 4)

        # Stitch into image: (3, H, W) where H=W=img_size
        img = patches.reshape(self.gh, self.gw, 3, 4, 4)
        img = img.transpose(2, 0, 3, 1, 4)
        img = img.reshape(3, self.img_size, self.img_size)

        return torch.from_numpy(img), 0


def decode_image_to_trees(images: torch.Tensor, depth: int) -> torch.Tensor:
    """Inverse of the spatial layout: extract per-patch tree bits from image.

    Returns: [B, n_patches, n_nodes] tensor of values close to ±1.
    """
    B, C, H, W = images.shape
    assert C == 3 and H % 4 == 0 and W % 4 == 0
    n_nodes = (2 ** (depth + 1)) - 1
    gh = gw = H // 4
    p = images.reshape(B, 3, gh, 4, gw, 4)
    p = p.permute(0, 2, 4, 1, 3, 5)  # [B, gh, gw, 3, 4, 4]
    p = p.reshape(B, gh * gw, 48)
    return p[..., :n_nodes]  # [B, n_patches, n_nodes]


def bit_recovery_metrics(orig_trees: torch.Tensor,
                         recon_trees: torch.Tensor,
                         depth: int) -> Dict[str, Any]:
    """Compute bit-level metrics from float-valued tree reconstructions.

    Args:
        orig_trees: [B, n_patches, n_nodes] in {-1, +1}
        recon_trees: [B, n_patches, n_nodes] continuous floats
        depth: tree depth (for per-level breakdown)

    Returns dict with:
        per_bit_acc:           fraction of correctly-recovered bits
        tree_exact_rate:       fraction of trees with all bits correct
        per_position_acc:      list of n_nodes accuracies (one per BFS index)
        per_level_acc:         dict {level: accuracy} for levels 0..depth
    """
    orig_bits = (orig_trees > 0).float()
    recon_bits = (recon_trees > 0).float()
    correct = (orig_bits == recon_bits).float()  # [B, n_patches, n_nodes]

    per_bit_acc = correct.mean().item()
    tree_exact_rate = correct.all(dim=-1).float().mean().item()
    per_position_acc = correct.mean(dim=(0, 1)).tolist()  # n_nodes

    layout = BinaryTreeDataset.bfs_layout(depth)
    per_level: Dict[int, list] = {l: [] for l in range(depth + 1)}
    for idx, (lvl, _pos) in enumerate(layout):
        per_level[lvl].append(per_position_acc[idx])
    per_level_acc = {l: float(np.mean(v)) for l, v in per_level.items()}

    return {
        'per_bit_acc': per_bit_acc,
        'tree_exact_rate': tree_exact_rate,
        'per_position_acc': per_position_acc,
        'per_level_acc': per_level_acc,
    }


# ═══════════════════════════════════════════════════════════════════
# PER-TYPE EVALUATION
# ═══════════════════════════════════════════════════════════════════

def eval_per_type(model, active_types, img_size, device, n_per_type=64):
    """MSE for each active noise type."""
    rng = np.random.RandomState(99)
    model.eval()
    results = {}
    with torch.no_grad():
        for t in active_types:
            imgs = torch.stack([
                _generate_noise(t, img_size, rng).clamp(-4, 4)
                for _ in range(n_per_type)
            ]).to(device)
            out = model(imgs)
            results[t] = F.mse_loss(out['recon'], imgs).item()
    return results


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train(cfg: Dict[str, Any]):
    """Main training loop. cfg is a preset dict or custom config."""

    # ── Architecture kwargs (what v1 was missing) ──
    V              = cfg['V']
    D              = cfg['D']
    patch_size     = cfg['patch_size']
    hidden         = cfg['hidden']
    depth          = cfg['depth']
    n_cross        = cfg['n_cross']
    n_heads        = cfg.get('n_heads', None)
    smooth_mid     = cfg.get('smooth_mid', None)
    linear_readout = cfg.get('linear_readout', False)
    svd_mode       = cfg.get('svd_mode', 'default')
    match_params   = cfg.get('match_params', True)

    # ── Training ──
    dataset       = cfg['dataset']
    img_size      = cfg['img_size']
    batch_size    = cfg['batch_size']
    lr            = cfg['lr']
    epochs        = cfg['epochs']
    target_cv     = cfg['target_cv']
    hf_version    = cfg['hf_version']
    save_every    = cfg.get('save_every', 10)
    report_every  = cfg.get('report_every', 500)

    # ── Loss ──
    cv_weight     = cfg.get('cv_weight', 0.3)
    boost         = cfg.get('boost', 0.5)
    sigma         = cfg.get('sigma', 0.15)

    # ── Data filters / curriculum ──
    pretrained    = cfg.get('pretrained', None)
    curriculum    = cfg.get('curriculum', None)
    tier_schedule = cfg.get('tier_schedule', None)
    allowed_types = cfg.get('allowed_types', None)

    # ── Tree config ──
    tree_depth    = cfg.get('tree_depth', 4)

    # ── Output paths ──
    save_dir      = cfg.get('save_dir', '/content/checkpoints')
    hf_repo       = cfg.get('hf_repo', 'AbstractPhil/geolip-SVAE')
    tb_dir        = cfg.get('tb_dir', '/content/runs')
    upload        = cfg.get('upload', True)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── TensorBoard ──
    from torch.utils.tensorboard import SummaryWriter
    run_name = f"{hf_version}_{img_size}x{img_size}_h{hidden}_d{depth}_lr{lr}"
    tb_path = os.path.join(tb_dir, run_name)
    writer = SummaryWriter(tb_path)
    print(f"  TensorBoard: {tb_path}")

    # ── HuggingFace ──
    hf_enabled = False
    api = None
    if upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.whoami()
            hf_enabled = True
            hf_prefix = f"{hf_version}/checkpoints"
            print(f"  HuggingFace: {hf_repo}/{hf_prefix}")
        except Exception as e:
            print(f"  HuggingFace: disabled ({e})")

    def upload_to_hf(local_path, remote_name, prefix=None):
        if not hf_enabled:
            return
        prefix = prefix if prefix is not None else hf_prefix
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"{prefix}/{remote_name}",
                repo_id=hf_repo, repo_type="model")
            print(f"  ☁️  Uploaded: {hf_repo}/{prefix}/{remote_name}")
        except Exception as e:
            print(f"  ⚠️  HF upload: {e}")

    # ── Model ──
    model_kwargs = dict(
        V=V, D=D, ps=patch_size, hidden=hidden, depth=depth, n_cross=n_cross,
        linear_readout=linear_readout, svd_mode=svd_mode,
        match_params=match_params,
    )
    if n_heads is not None:
        model_kwargs['n_heads'] = n_heads
    if smooth_mid is not None:
        model_kwargs['smooth_mid'] = smooth_mid
    model = PatchSVAE(**model_kwargs).to(device)

    # ── Pretrained weights ──
    if pretrained:
        from huggingface_hub import hf_hub_download
        print(f"\n  Loading pretrained: {pretrained}")
        try:
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=pretrained,
                                         repo_type='model')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            print(f"  Loaded ep{ckpt['epoch']}, MSE={ckpt['test_mse']:.6f}")
        except Exception as e:
            print(f"  ⚠️  Pretrained load failed: {e} — training from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Data ──
    is_noise = is_text = is_image = is_tree = False

    if dataset in ('tiny_imagenet', 'imagenet_128', 'imagenet_256'):
        train_loader, test_loader, _, _ = get_image_loaders(
            dataset, img_size, batch_size)
        is_image = True

    elif dataset == 'curriculum_noise':
        ds_size = cfg.get('ds_size', 500_000)
        val_size = cfg.get('val_size', 10_000)
        train_ds = CurriculumNoiseDataset(size=ds_size, img_size=img_size,
                                           allowed_types=allowed_types)
        val_ds = CurriculumNoiseDataset(size=val_size, img_size=img_size,
                                         allowed_types=allowed_types)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_noise = True

    elif dataset in ('omega_noise', 'scheduled_noise'):
        ds_size = cfg.get('ds_size', 1_280_000)
        val_size = cfg.get('val_size', 10_000)
        if dataset == 'scheduled_noise':
            train_ds = CurriculumNoiseDataset(size=ds_size, img_size=img_size,
                                               allowed_types=allowed_types)
            val_ds = CurriculumNoiseDataset(size=val_size, img_size=img_size,
                                             allowed_types=allowed_types)
        else:
            train_ds = OmegaNoiseDataset(size=ds_size, img_size=img_size,
                                          allowed_types=allowed_types)
            val_ds = OmegaNoiseDataset(size=val_size, img_size=img_size,
                                        allowed_types=allowed_types)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_noise = True

    elif dataset == 'wikipedia':
        ds_size = cfg.get('ds_size', 200_000)
        val_size = cfg.get('val_size', 5_000)
        print(f"\n  Loading Wikipedia corpus...")
        train_ds = WikiTextAsImage(size=ds_size, img_size=img_size, split='train')
        val_ds = WikiTextAsImage(size=val_size, img_size=img_size, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_text = True

    elif dataset == 'binary_tree':
        ds_size = cfg.get('ds_size', 200_000)
        val_size = cfg.get('val_size', 2_000)
        train_ds = BinaryTreeDataset(size=ds_size, img_size=img_size,
                                      depth=tree_depth)
        val_ds = BinaryTreeDataset(size=val_size, img_size=img_size,
                                    depth=tree_depth, seed=999)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_tree = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── Print config ──
    n_patches = (img_size // patch_size) ** 2
    arch_tags = []
    if linear_readout:
        arch_tags.append('linear_readout')
    if svd_mode != 'default':
        arch_tags.append(f"svd={svd_mode}")
    arch_str = f" [{'+'.join(arch_tags)}]" if arch_tags else ""

    print(f"\n  SVAE TRAINER (v2) — {hf_version}{arch_str}")
    print(f"  {img_size}×{img_size}, {n_patches} patches, V={V}, D={D}, "
          f"{total_params:,} params")
    print(f"  Dataset: {dataset}, batch={batch_size}, lr={lr}, epochs={epochs}")
    print(f"  Target CV: {target_cv}, soft hand: boost={1+boost:.1f}x, "
          f"penalty={cv_weight}")
    if allowed_types is not None:
        print(f"  Allowed types: {allowed_types}")
    if curriculum:
        print(f"  Curriculum: {curriculum}")
    if tier_schedule:
        print(f"  Tier schedule: {tier_schedule}")
    if is_tree:
        print(f"  Tree depth: {tree_depth}, "
              f"n_nodes: {train_ds.n_nodes}, n_pad: {train_ds.n_pad}")
    print("=" * 100)

    # ── Helpers ──
    best_recon = float('inf')
    history: List[Dict[str, Any]] = []

    def save_checkpoint(path, epoch_, test_mse_, extra=None, do_upload=True):
        ckpt_out = {
            'epoch': epoch_, 'test_mse': test_mse_,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'config': {
                'V': V, 'D': D, 'patch_size': patch_size,
                'hidden': hidden, 'depth': depth, 'n_cross_layers': n_cross,
                'linear_readout': linear_readout, 'svd_mode': svd_mode,
                'match_params': match_params,
                'target_cv': target_cv, 'dataset': dataset,
                'img_size': img_size, 'lr': lr,
            },
        }
        if extra:
            ckpt_out.update(extra)
        torch.save(ckpt_out, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  💾 Saved: {path} ({size_mb:.1f}MB, ep{epoch_}, "
              f"MSE={test_mse_:.6f})")
        if do_upload:
            upload_to_hf(path, os.path.basename(path))

    def per_layer_alphas():
        """Return (alpha_mean, alpha_std) averaged across cross-attn layers."""
        if n_cross <= 0 or len(model.cross_attn) == 0:
            return 0.0, 0.0
        alphas = [layer.alpha.detach() for layer in model.cross_attn]
        a_mean = torch.stack([a.mean() for a in alphas]).mean().item()
        a_std = torch.stack([a.std() for a in alphas]).mean().item()
        return a_mean, a_std

    # ── Patience promotion state (for curriculum) ──
    tier_best_mse = float('inf')
    stale_epochs = 0

    # ── Training ──
    last_cv = target_cv
    last_prox = 1.0
    global_batch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, n_seen = 0.0, 0.0, 0
        epoch_max_grad = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}",
                    bar_format='{l_bar}{bar:20}{r_bar}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(images)
            recon_loss = F.mse_loss(out['recon'], images)

            # Soft-hand proximity (measure CV every 50 batches)
            with torch.no_grad():
                if batch_idx % 50 == 0:
                    current_cv = cv_of(out['svd']['M'][0, 0])
                    if current_cv > 0:
                        last_cv = current_cv
                    delta = last_cv - target_cv
                    last_prox = math.exp(-delta ** 2 / (2 * sigma ** 2))

            recon_w = 1.0 + boost * last_prox
            cv_pen = cv_weight * (1.0 - last_prox)
            loss = recon_w * recon_loss + cv_pen * (last_cv - target_cv) ** 2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.cross_attn.parameters(), max_norm=0.5
            )

            # Track total grad norm for stability
            total_grad = sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            epoch_max_grad = max(epoch_max_grad, total_grad)

            opt.step()

            total_loss += loss.item() * len(images)
            total_recon += recon_loss.item() * len(images)
            n_seen += len(images)
            global_batch += 1
            pbar.set_postfix_str(f"mse={recon_loss.item():.4f} cv={last_cv:.3f}")

            # Mid-epoch report
            if global_batch % report_every == 0:
                model.eval()
                with torch.no_grad():
                    test_imgs, _ = next(iter(test_loader))
                    test_imgs = test_imgs.to(device)
                    t_out = model(test_imgs)
                    test_mse = F.mse_loss(t_out['recon'], test_imgs).item()

                    S_batch = t_out['svd']['S']
                    S_orig = t_out['svd']['S_orig']
                    S_mean = S_batch.mean(dim=(0, 1))
                    S0 = S_mean[0].item()
                    SD = S_mean[-1].item()
                    ratio = S0 / (SD + 1e-8)
                    erank = model.effective_rank(
                        S_batch.reshape(-1, D)
                    ).mean().item()
                    s_delta = (S_batch - S_orig).abs().mean().item()
                    a_mean, a_std = per_layer_alphas()
                    cv_in_band = 0.13 <= last_cv <= 0.30

                # TB scalars
                writer.add_scalar('train/loss', total_loss / n_seen, global_batch)
                writer.add_scalar('train/recon', total_recon / n_seen, global_batch)
                writer.add_scalar('test/mse', test_mse, global_batch)
                writer.add_scalar('geo/S0', S0, global_batch)
                writer.add_scalar('geo/SD', SD, global_batch)
                writer.add_scalar('geo/ratio', ratio, global_batch)
                writer.add_scalar('geo/erank', erank, global_batch)
                writer.add_scalar('geo/row_cv', last_cv, global_batch)
                writer.add_scalar('geo/cv_in_band', float(cv_in_band), global_batch)
                writer.add_scalar('geo/s_delta', s_delta, global_batch)
                writer.add_scalar('cross_attn/alpha_mean', a_mean, global_batch)
                writer.add_scalar('cross_attn/alpha_std', a_std, global_batch)
                writer.add_scalar('stab/prox', last_prox, global_batch)
                writer.add_scalar('stab/recon_w', recon_w, global_batch)
                writer.add_scalar('stab/epoch_max_grad', epoch_max_grad, global_batch)
                writer.add_scalar('stab/lr', opt.param_groups[0]['lr'], global_batch)

                history.append({
                    'epoch': epoch, 'global_batch': global_batch,
                    'train_recon': total_recon / n_seen,
                    'test_mse': test_mse,
                    'S0': S0, 'SD': SD, 'ratio': ratio, 'erank': erank,
                    'row_cv': last_cv, 'cv_in_band': cv_in_band,
                    's_delta': s_delta,
                    'alpha_mean': a_mean, 'alpha_std': a_std,
                    'epoch_max_grad': epoch_max_grad,
                })
                model.train()

        pbar.close()
        sched.step()
        epoch_time = time.time() - t0

        # ── Full epoch eval ──
        model.eval()
        test_mse_total, test_n = 0.0, 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                test_mse_total += F.mse_loss(out['recon'], imgs).item() * len(imgs)
                test_n += len(imgs)
        test_mse = test_mse_total / test_n

        # Geometry snapshot
        with torch.no_grad():
            sample, _ = next(iter(test_loader))
            sample = sample[:min(64, len(sample))].to(device)
            out = model(sample)
            S_mean = out['svd']['S'].mean(dim=(0, 1))
            S_orig = out['svd']['S_orig'].mean(dim=(0, 1))
            ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
            erank = model.effective_rank(out['svd']['S'].reshape(-1, D)).mean().item()
            s_delta = (S_mean - S_orig).abs().mean().item()
            a_mean, a_std = per_layer_alphas()
            cv_in_band = 0.13 <= last_cv <= 0.30

        # Per-type MSE for noise variants
        type_str = ""
        if is_noise:
            active = list(range(16))
            ds_obj = train_loader.dataset
            if hasattr(ds_obj, 'active_types'):
                active = ds_obj.active_types
            type_mse = eval_per_type(model, active, img_size, device, n_per_type=32)
            type_str = " ".join(f"{NOISE_NAMES[t][:4]}={v:.3f}"
                                  for t, v in sorted(type_mse.items()))

        # Byte accuracy for text
        byte_str = ""
        if is_text:
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:32].to(device)
                sample_out = model(sample_imgs)
                orig_b = ((sample_imgs.cpu().flatten(1) + 1.0) * 127.5)\
                    .round().clamp(0, 255).long()
                recon_b = ((sample_out['recon'].cpu().flatten(1) + 1.0) * 127.5)\
                    .round().clamp(0, 255).long()
                byte_acc = (orig_b == recon_b).float().mean().item()
            byte_str = f"bytes={byte_acc * 100:.1f}%"

        # Bit-recovery for binary tree
        tree_str = ""
        tree_metrics = None
        if is_tree:
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:64].to(device)
                sample_out = model(sample_imgs)
                orig_trees = decode_image_to_trees(sample_imgs, tree_depth)
                recon_trees = decode_image_to_trees(sample_out['recon'], tree_depth)
                tree_metrics = bit_recovery_metrics(orig_trees, recon_trees,
                                                    tree_depth)
            tree_str = (f"bits={tree_metrics['per_bit_acc']*100:.1f}% "
                        f"trees={tree_metrics['tree_exact_rate']*100:.1f}%")
            for lvl, acc in tree_metrics['per_level_acc'].items():
                writer.add_scalar(f'tree/level_{lvl}_acc', acc, epoch)
            writer.add_scalar('tree/per_bit_acc',
                              tree_metrics['per_bit_acc'], epoch)
            writer.add_scalar('tree/exact_rate',
                              tree_metrics['tree_exact_rate'], epoch)

        print(f" {epoch:3d} | {total_loss/n_seen:.4f} {total_recon/n_seen:.4f} "
              f"{epoch_time:.0f}s | test={test_mse:.6f} | "
              f"S0={S_mean[0]:.3f} SD={S_mean[-1]:.3f} r={ratio:.2f} er={erank:.2f}"
              f" | cv={last_cv:.3f} band={'Y' if cv_in_band else 'N'} "
              f"Sd={s_delta:.5f} a={a_mean:.3f} g={epoch_max_grad:.1f} "
              f"{byte_str} {tree_str} {type_str}")

        # Per-epoch TB
        writer.add_scalar('epoch/test_mse', test_mse, epoch)
        writer.add_scalar('epoch/train_recon', total_recon / n_seen, epoch)
        writer.add_scalar('epoch/cv', last_cv, epoch)
        writer.add_scalar('epoch/cv_in_band', float(cv_in_band), epoch)
        writer.add_scalar('epoch/S0', S_mean[0].item(), epoch)
        writer.add_scalar('epoch/erank', erank, epoch)
        writer.add_scalar('epoch/max_grad', epoch_max_grad, epoch)
        writer.add_scalar('epoch/time_s', epoch_time, epoch)
        writer.add_scalar('epoch/alpha_mean', a_mean, epoch)
        writer.add_scalar('epoch/alpha_std', a_std, epoch)

        # End-of-epoch history record
        history.append({
            'epoch': epoch, 'global_batch': global_batch,
            'epoch_test_mse': test_mse,
            'train_recon': total_recon / n_seen,
            'S0': S_mean[0].item(), 'SD': S_mean[-1].item(),
            'ratio': ratio, 'erank': erank,
            'row_cv': last_cv, 'cv_in_band': cv_in_band,
            's_delta': s_delta,
            'alpha_mean': a_mean, 'alpha_std': a_std,
            'epoch_max_grad': epoch_max_grad,
            'epoch_time_s': epoch_time,
            'tree_metrics': tree_metrics,
        })

        # ── Curriculum: scheduled tier unlocks ──
        if curriculum == 'scheduled' and tier_schedule and epoch in tier_schedule:
            next_tier = tier_schedule[epoch]
            train_loader.dataset.unlock_tier(next_tier)
            test_loader.dataset.unlock_tier(next_tier)
            new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
            print(f"\n  ★ TIER {next_tier} UNLOCKED (epoch {epoch}): "
                  f"+{', '.join(new_names)}")
            active_now = [NOISE_NAMES[t] for t in train_loader.dataset.active_types]
            print(f"    Active: {active_now}\n")
            save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                            epoch, test_mse, do_upload=True)

        # ── Curriculum: patience-based promotion ──
        if curriculum == 'patience' and hasattr(train_loader.dataset, 'unlock_tier'):
            improvement = (tier_best_mse - test_mse) / (tier_best_mse + 1e-8)
            if test_mse < tier_best_mse:
                tier_best_mse = test_mse
            if improvement < 0.01:
                stale_epochs += 1
            else:
                stale_epochs = 0
            if (stale_epochs >= 10
                    and train_loader.dataset.current_tier >= 0
                    and train_loader.dataset.current_tier < max(TIERS.keys())):
                next_tier = train_loader.dataset.current_tier + 1
                train_loader.dataset.unlock_tier(next_tier)
                test_loader.dataset.unlock_tier(next_tier)
                new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
                print(f"\n  ★ PROMOTED TO TIER {next_tier}: +{', '.join(new_names)}")
                active_now = [NOISE_NAMES[t] for t in train_loader.dataset.active_types]
                print(f"    Active: {active_now}\n")
                tier_best_mse = test_mse
                stale_epochs = 0
                save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                                epoch, test_mse, do_upload=True)

        # ── Checkpointing ──
        if test_mse < best_recon:
            best_recon = test_mse
            save_checkpoint(os.path.join(save_dir, 'best.pt'),
                            epoch, test_mse, do_upload=False)

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(os.path.join(save_dir, f'epoch_{epoch:04d}.pt'),
                            epoch, test_mse)
            best_path = os.path.join(save_dir, 'best.pt')
            if os.path.exists(best_path):
                upload_to_hf(best_path, 'best.pt')
            writer.flush()
            if hf_enabled:
                try:
                    api.upload_folder(folder_path=tb_path,
                                      path_in_repo=f"{hf_version}/tensorboard/{run_name}",
                                      repo_id=hf_repo, repo_type="model")
                    print(f"  ☁️  TB synced")
                except Exception:
                    pass

    writer.close()

    # ── Final report ──
    final_report = {
        'run_name': run_name,
        'config': {
            'V': V, 'D': D, 'patch_size': patch_size,
            'hidden': hidden, 'depth': depth, 'n_cross_layers': n_cross,
            'n_heads': n_heads, 'smooth_mid': smooth_mid,
            'linear_readout': linear_readout, 'svd_mode': svd_mode,
            'match_params': match_params,
            'dataset': dataset, 'img_size': img_size, 'batch_size': batch_size,
            'lr': lr, 'epochs': epochs, 'target_cv': target_cv,
            'allowed_types': allowed_types, 'curriculum': curriculum,
            'tier_schedule': tier_schedule, 'tree_depth': tree_depth,
        },
        'n_params': total_params,
        'n_patches': n_patches,
        'best_test_mse': best_recon,
        'history': history,
    }
    report_path = os.path.join(save_dir, 'final_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    print(f"\n  Final report: {report_path}")
    if hf_enabled:
        upload_to_hf(report_path, 'final_report.json',
                      prefix=hf_version)

    print(f"\n  TRAINING COMPLETE — {hf_version}")
    print(f"  Best MSE: {best_recon:.6f}")
    print(f"  Checkpoints: {save_dir}/")
    return final_report


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVAE Unified Trainer (v2)')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Named preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from preset')
    parser.add_argument('--no-upload', action='store_true',
                        help='Disable HF upload')
    args = parser.parse_args()

    if args.list_presets:
        for name, cfg in PRESETS.items():
            ds = cfg['dataset']
            sz = cfg['img_size']
            ep = cfg['epochs']
            arch_tags = []
            if cfg.get('linear_readout'):
                arch_tags.append('lin_readout')
            if cfg.get('svd_mode', 'default') != 'default':
                arch_tags.append(f"svd={cfg['svd_mode']}")
            arch = f" [{'+'.join(arch_tags)}]" if arch_tags else ""
            pre = cfg.get('pretrained', 'scratch')
            print(f"  {name:<22s} {ds:<20s} {sz}×{sz}  {ep:>3d} ep"
                  f"  V={cfg['V']:<3d} D={cfg['D']:<3d}{arch}  from={pre}")
        exit()

    if not args.preset:
        parser.print_help()
        print("\nPresets:")
        for name in PRESETS:
            print(f"  {name}")
        exit()

    cfg = dict(PRESETS[args.preset])
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.no_upload:
        cfg['upload'] = False

    torch.set_float32_matmul_precision('high')
    train(cfg)
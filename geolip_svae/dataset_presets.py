"""
geolip_svae.dataset_presets
============================
Dataset classes, helpers, and factory registry for the SVAE trainer.

Decoupled from train.py so the trainer is a pure orchestrator and
datasets can be reused independently (codebook calibration, evaluation
harnesses, downstream task probes, etc.).

Public surface:

    Dataset classes (instantiate directly when you need raw access):
        OmegaNoiseDataset, CurriculumNoiseDataset
        HFImageDataset                (use get_image_loaders for ImageNet variants)
        WikiTextAsImage
        BinaryTreeDataset             (+ decode_image_to_trees, bit_recovery_metrics)
        SentencePieceBitDataset       (+ decode_image_to_tokens, token_bit_recovery_metrics)
        ByteTrigramDataset            (+ byte_recovery_metrics)

    Constants:
        NOISE_NAMES, TIERS

    Free functions:
        _generate_noise(noise_type, s, rng, channels=3)
        _pink_noise(shape), _brown_noise(shape)
        eval_per_type(model, types, img_size, device, n_per_type=64, channels=3)

    Factory layer (used by the trainer; consume via get_dataset_bundle):
        @dataclass DatasetBundle
        DATASET_FACTORIES (registry mapping cfg['dataset'] → factory fn)
        get_dataset_bundle(cfg, channels=3) → DatasetBundle

The factory layer is the single dispatch surface the trainer uses:

    bundle = get_dataset_bundle(cfg, channels=channels)
    bundle.train_loader   bundle.test_loader
    bundle.is_noise / is_text / is_image / is_tree / is_sentencepiece /
        is_byte_trigram   — kind flags consumed by the trainer's
                            per-kind logging / eval branches
"""

from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


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


def _generate_noise(noise_type, s, rng, channels: int = 3):
    """Generate one noise image of shape ``(channels, s, s)``.

    The 16 noise types preserve their statistical character regardless of
    channel count; ``channels`` only changes how many parallel channels
    are produced. Default 3 reproduces the original RGB-shaped noise.
    """
    C = channels
    if noise_type == 0:
        return torch.randn(C, s, s)
    elif noise_type == 1:
        return torch.rand(C, s, s) * 2 - 1
    elif noise_type == 2:
        return (torch.rand(C, s, s) - 0.5) * 4
    elif noise_type == 3:
        lam = rng.uniform(0.5, 20.0)
        return torch.poisson(torch.full((C, s, s), lam)) / lam - 1.0
    elif noise_type == 4:
        img = _pink_noise((C, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 5:
        img = _brown_noise((C, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 6:
        img = torch.where(torch.rand(C, s, s) > 0.5,
                          torch.ones(C, s, s) * 2, -torch.ones(C, s, s) * 2)
        return img + torch.randn(C, s, s) * 0.1
    elif noise_type == 7:
        return torch.randn(C, s, s) * (torch.rand(C, s, s) > 0.9).float() * 3
    elif noise_type == 8:
        b = rng.randint(2, 16)
        sm = torch.randn(C, s // b + 1, s // b + 1)
        return F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif noise_type == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        a = rng.uniform(0, 2 * math.pi)
        return ((math.cos(a) * gx + math.sin(a) * gy)
                .unsqueeze(0).expand(C, -1, -1)
                + torch.randn(C, s, s) * 0.5)
    elif noise_type == 10:
        cs = rng.randint(2, 16)
        cy = torch.arange(s) // cs; cx = torch.arange(s) // cs
        return (((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float()
                .unsqueeze(0).expand(C, -1, -1) * 2 - 1
                + torch.randn(C, s, s) * 0.3)
    elif noise_type == 11:
        alpha = rng.uniform(0.2, 0.8)
        return alpha * torch.randn(C, s, s) + (1 - alpha) * (torch.rand(C, s, s) * 2 - 1)
    elif noise_type == 12:
        img = torch.zeros(C, s, s); h2 = s // 2
        img[:, :h2, :h2] = torch.randn(C, h2, h2)
        img[:, :h2, h2:] = torch.rand(C, h2, h2) * 2 - 1
        img[:, h2:, :h2] = _pink_noise((C, h2, h2)) / 2
        img[:, h2:, h2:] = torch.where(torch.rand(C, h2, h2) > 0.5,
                                         torch.ones(C, h2, h2),
                                         -torch.ones(C, h2, h2))
        return img
    elif noise_type == 13:
        return torch.tan(math.pi * (torch.rand(C, s, s) - 0.5)).clamp(-3, 3)
    elif noise_type == 14:
        return torch.empty(C, s, s).exponential_(1.0) - 1.0
    elif noise_type == 15:
        u = torch.rand(C, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    return torch.randn(C, s, s)


class CurriculumNoiseDataset(torch.utils.data.Dataset):
    """Noise with tier-based type activation for Johanna curriculum training.

    `allowed_types` overrides the curriculum entirely if provided.
    """

    def __init__(self, size=500000, img_size=64, allowed_types=None,
                 channels: int = 3):
        self.size = size
        self.img_size = img_size
        self.channels = channels
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
        img = _generate_noise(
            noise_type, self.img_size, self._rng, self.channels,
        ).clamp(-4, 4)
        return img.float(), noise_type


class OmegaNoiseDataset(torch.utils.data.Dataset):
    """Noise types with optional `allowed_types` filter.

    Default (allowed_types=None) is all 16 types. Gaussian-only foundation
    runs pass allowed_types=[0]. Custom subsets are passed as iterables.
    """

    def __init__(self, size=1280000, img_size=128, allowed_types=None,
                 channels: int = 3):
        self.size = size
        self.img_size = img_size
        self.channels = channels
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
        img = _generate_noise(
            noise_type, self.img_size, self._rng, self.channels,
        ).clamp(-4, 4)
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
# SENTENCEPIECE-BIT DATASET — first real-data substrate prototype
# ═══════════════════════════════════════════════════════════════════

class SentencePieceBitDataset(torch.utils.data.Dataset):
    """T5-base SentencePiece token bits packed into (3, H, W) images.

    Each 4x4 RGB patch (48 floats) holds:
      - First n_bits floats: ±1 bit values of one token's ID (LSB-first)
      - Remaining floats: zero padding

    Image layout: (img_size // 4)^2 patches per image, each holding ONE
    token. Tokens within an image come from a contiguous corpus excerpt,
    so the substrate sees natural token-sequence locality.

    Vocab: t5-base = 32128 < 2^15 = 32768. Default n_bits=16 leaves a
    1-bit buffer above vocab range (clean byte alignment for downstream
    decoders).
    """

    PATCH_FLOATS = 48

    def __init__(self, size=200_000, img_size=16,
                 tokenizer_id='google-t5/t5-base',
                 corpus_id='wikitext-2-raw-v1',
                 n_bits=16, split='train', seed=42,
                 max_corpus_chars=20_000_000):
        self.size = size
        self.img_size = img_size
        self.n_bits = n_bits
        if n_bits > self.PATCH_FLOATS:
            raise ValueError(
                f"n_bits={n_bits} exceeds patch capacity {self.PATCH_FLOATS}")
        self.n_pad = self.PATCH_FLOATS - n_bits
        if img_size % 4 != 0:
            raise ValueError(f"img_size must be divisible by 4, got {img_size}")
        self.gh = self.gw = img_size // 4
        self.n_patches = self.gh * self.gw  # tokens per image
        self._base_seed = seed
        self._call_count = 0
        self._rng = np.random.default_rng(seed)

        # Promote bare aliases to canonical org-prefixed paths. The bare names
        # ('t5-base', 't5-small') now redirect to 'google-t5/...' on HF and
        # the redirect is unauthenticated-restricted in some access tiers.
        if tokenizer_id in ('t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'):
            tokenizer_id = f'google-t5/{tokenizer_id}'

        # ── Load tokenizer ──
        # Two modes: tokenizer_id can be either:
        #   (a) an HF repo id like 'google-t5/t5-base' → downloads spiece.model
        #   (b) a local path ending in .model or .spm → loads directly
        # We use the `sentencepiece` package directly (no transformers dep
        # required just for tokenization).
        import sentencepiece as spm
        if os.path.isfile(tokenizer_id) and tokenizer_id.endswith(('.model', '.spm')):
            spm_path = tokenizer_id
            print(f"  [SentencePieceBitDataset] Loading local tokenizer "
                  f"{tokenizer_id}...")
        else:
            from huggingface_hub import hf_hub_download
            print(f"  [SentencePieceBitDataset] Loading tokenizer "
                  f"{tokenizer_id}...")
            spm_path = hf_hub_download(repo_id=tokenizer_id,
                                        filename='spiece.model')
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.vocab_size = self.sp.GetPieceSize()
        max_token_id = self.vocab_size - 1
        if max_token_id >= (1 << n_bits):
            raise ValueError(
                f"vocab_size={self.vocab_size} requires more than n_bits={n_bits}; "
                f"max token id {max_token_id} >= 2^{n_bits} = {1 << n_bits}")
        print(f"  [SentencePieceBitDataset] vocab={self.vocab_size}, "
              f"max_id={max_token_id}, encoded as {n_bits} bits")

        # ── Load and tokenize corpus (cached after first call) ──
        # Two modes: corpus_id can be either:
        #   (a) an HF datasets identifier like 'wikitext-2-raw-v1' or
        #       'wikitext/wikitext-103-v1' → loaded via load_dataset
        #   (b) a local path to a .txt file → loaded directly
        from datasets import load_dataset
        if os.path.isfile(corpus_id) and corpus_id.endswith(('.txt', '.text')):
            print(f"  [SentencePieceBitDataset] Loading local corpus "
                  f"{corpus_id}...")
            with open(corpus_id, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            print(f"  [SentencePieceBitDataset] Loading corpus "
                  f"{corpus_id}...")
            if corpus_id.startswith('wikitext'):
                ds = load_dataset('wikitext', corpus_id, split=split)
            else:
                ds = load_dataset(corpus_id, split=split)
            text_parts, total = [], 0
            for record in ds:
                txt = record.get('text', '')
                if not txt or not txt.strip():
                    continue
                text_parts.append(txt)
                total += len(txt)
                if total >= max_corpus_chars:
                    break
            full_text = '\n'.join(text_parts)
        # Cap text by char count if loaded from local
        if len(full_text) > max_corpus_chars:
            full_text = full_text[:max_corpus_chars]
        print(f"  [SentencePieceBitDataset] Tokenizing {len(full_text):,} chars...")
        self.token_ids = np.asarray(
            self.sp.EncodeAsIds(full_text), dtype=np.int32)
        if len(self.token_ids) < self.n_patches + 1:
            raise RuntimeError(
                f"Corpus too small: only {len(self.token_ids)} tokens, "
                f"need at least {self.n_patches + 1}")
        print(f"  [SentencePieceBitDataset] {len(self.token_ids):,} tokens. "
              f"n_patches/img = {self.n_patches}")

    def __len__(self):
        return self.size

    @staticmethod
    def ids_to_bits(ids: np.ndarray, n_bits: int) -> np.ndarray:
        """[N] int → [N, n_bits] ±1 floats, LSB-first."""
        bit_indices = np.arange(n_bits, dtype=np.int32)
        # Broadcast: (N, 1) >> (n_bits,) → (N, n_bits)
        bits = ((ids[:, None] >> bit_indices[None, :]) & 1).astype(np.float32)
        return bits * 2.0 - 1.0  # 0/1 → -1/+1

    @staticmethod
    def bits_to_ids(bits: np.ndarray) -> np.ndarray:
        """[..., n_bits] sign-thresholded floats → [...] int ids, LSB-first."""
        n_bits = bits.shape[-1]
        bin_bits = (bits > 0).astype(np.int64)
        powers = (1 << np.arange(n_bits, dtype=np.int64))
        return (bin_bits * powers).sum(axis=-1)

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.default_rng(int.from_bytes(os.urandom(4), 'big'))

        # Random contiguous window of n_patches tokens
        max_start = len(self.token_ids) - self.n_patches
        start = int(self._rng.integers(0, max_start + 1))
        ids = self.token_ids[start:start + self.n_patches]

        # Encode each id to n_bits ±1 values
        bits = self.ids_to_bits(ids, self.n_bits)  # [n_patches, n_bits]

        # Pad to 48 floats per patch
        padded = np.zeros((self.n_patches, self.PATCH_FLOATS), dtype=np.float32)
        padded[:, :self.n_bits] = bits

        # Reshape to (n_patches, 3, 4, 4) channels-first
        patches = padded.reshape(self.n_patches, 3, 4, 4)

        # Stitch into image (3, H, W)
        img = patches.reshape(self.gh, self.gw, 3, 4, 4)
        img = img.transpose(2, 0, 3, 1, 4)
        img = img.reshape(3, self.img_size, self.img_size)

        return torch.from_numpy(img), 0


def decode_image_to_tokens(images: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Inverse of the SentencePiece-bit spatial layout.

    Returns: [B, n_patches, n_bits] tensor of values close to ±1.
    """
    B, C, H, W = images.shape
    assert C == 3 and H % 4 == 0 and W % 4 == 0
    gh = gw = H // 4
    p = images.reshape(B, 3, gh, 4, gw, 4)
    p = p.permute(0, 2, 4, 1, 3, 5)  # [B, gh, gw, 3, 4, 4]
    p = p.reshape(B, gh * gw, 48)
    return p[..., :n_bits]  # [B, n_patches, n_bits]


def token_bit_recovery_metrics(orig_bits: torch.Tensor,
                                recon_bits: torch.Tensor) -> Dict[str, Any]:
    """Per-bit and per-token recovery metrics for SentencePiece-bit content.

    Args:
        orig_bits:  [B, n_patches, n_bits] in {-1, +1}
        recon_bits: [B, n_patches, n_bits] continuous floats

    Returns dict with:
        per_bit_acc:           overall fraction of correctly-recovered bits
        token_exact_rate:      fraction of tokens with all bits correct
        per_bit_position_acc:  list of n_bits accuracies (which bit positions
                               are most/least reliable — LSB to MSB)
        per_seq_position_acc:  list of n_patches accuracies (does the model
                               handle position-1-in-sequence as well as
                               position-N-in-sequence)
    """
    orig_signs = (orig_bits > 0).float()
    recon_signs = (recon_bits > 0).float()
    correct = (orig_signs == recon_signs).float()  # [B, n_patches, n_bits]

    per_bit_acc = correct.mean().item()
    # All bits correct in a token = exact match
    token_exact_rate = correct.all(dim=-1).float().mean().item()
    # Accuracy averaged over [B, n_patches] for each bit position
    per_bit_position_acc = correct.mean(dim=(0, 1)).tolist()
    # Accuracy averaged over [B, n_bits] for each sequence position
    per_seq_position_acc = correct.mean(dim=(0, 2)).tolist()

    return {
        'per_bit_acc': per_bit_acc,
        'token_exact_rate': token_exact_rate,
        'per_bit_position_acc': per_bit_position_acc,
        'per_seq_position_acc': per_seq_position_acc,
    }


# ═══════════════════════════════════════════════════════════════════
# BYTE-TRIGRAM DATASET (every cell is a pixel-equivalent RGB byte triple)
# ═══════════════════════════════════════════════════════════════════

class ByteTrigramDataset(torch.utils.data.Dataset):
    """UTF-8 byte n-grams packed as image channels.

    Each spatial cell (row, col) of each patch_size×patch_size patch holds
    one byte n-gram from the corpus stream as a ``channels``-tuple:

        cell[r, c]: (c0, c1, ..., c{C-1}) = bytes[Ci], ..., bytes[Ci + C - 1]

    where i is the cell's linear index in the image (row-major across
    patches, row-major within each patch) and C = ``channels``. Bytes
    are normalized [0, 255] → [-1, 1] via (b - 127.5) / 127.5.

    Default ``channels=3`` is the canonical RGB-trigram encoding (3 bytes
    per pixel as R, G, B). Other values give:
        channels=1  monogram   — one byte per pixel; greyscale-equivalent
        channels=2  bigram     — two bytes per pixel
        channels=3  trigram    — three bytes (the default; matches RGB)
        channels=4  quadgram   — four bytes per pixel (fits RGBA-shaped tensors)

    Capacity (per spatial cell): 256^channels distinguishable n-grams
    (e.g. 16.7M for trigrams, 4.3B for quadgrams).
    Capacity (per ps×ps patch): ps² · channels bytes per patch.
    Capacity (per H×W image): H·W·channels bytes per training sample.

    No padding — every float carries signal. The class name is preserved
    for back-compat; ``channels=3`` reproduces the trigram encoding exactly.

    Args:
        size: dataset length (number of training samples to yield)
        img_size: image dimension (must be divisible by patch_size)
        patch_size: patch dimension (default 4)
        channels: bytes-per-cell n-gram size (default 3 = RGB trigram).
            Must match the channel count of the model that consumes
            this dataset (PatchSVAE.channels).
        corpus_id: HF dataset name (default 'wikitext-103-raw-v1') OR a
            local .txt path. Loaded as a single byte stream.
        split: dataset split for HF datasets (default 'train')
        seed: RNG seed for window sampling
        max_corpus_bytes: cap on corpus bytes loaded into memory.
            Default None = load entire corpus.
    """

    def __init__(self, size=200_000, img_size=256, patch_size=4,
                 channels=3,
                 corpus_id='wikitext-103-raw-v1', split='train',
                 seed=42, max_corpus_bytes=None):
        self.size = size
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by "
                f"patch_size={patch_size}")
        self.gh = self.gw = img_size // patch_size
        self.cells_per_patch = patch_size * patch_size  # 16 at ps=4
        self.n_patches = self.gh * self.gw               # 4096 at 256/4
        self.bytes_per_image = self.n_patches * self.cells_per_patch * channels
        self._base_seed = seed
        self._call_count = 0
        self._rng = np.random.default_rng(seed)

        # ── Load corpus as raw UTF-8 byte stream ──
        if os.path.isfile(corpus_id) and corpus_id.endswith(('.txt', '.text')):
            print(f"  [ByteTrigramDataset] Loading local corpus "
                  f"{corpus_id}...")
            with open(corpus_id, 'rb') as f:
                if max_corpus_bytes is not None:
                    full_bytes = f.read(max_corpus_bytes)
                else:
                    full_bytes = f.read()
        else:
            print(f"  [ByteTrigramDataset] Loading corpus {corpus_id}...")
            from datasets import load_dataset
            if corpus_id.startswith('wikitext'):
                ds = load_dataset('wikitext', corpus_id, split=split)
            else:
                ds = load_dataset(corpus_id, split=split)
            chunks = []
            total = 0
            for record in ds:
                txt = record.get('text', '')
                if not txt or not txt.strip():
                    continue
                b = txt.encode('utf-8')
                chunks.append(b)
                total += len(b)
                if max_corpus_bytes is not None and total >= max_corpus_bytes:
                    break
            full_bytes = b'\n'.join(chunks)

        if len(full_bytes) > 10_000_000_000:
            gb = len(full_bytes) / 1e9
            print(f"  [ByteTrigramDataset] WARNING: corpus is {gb:.1f} GB. "
                  f"This loads entirely into system RAM as a uint8 array. "
                  f"Set max_corpus_bytes=N to subsample if RAM-constrained.")

        self.corpus = np.frombuffer(full_bytes, dtype=np.uint8)
        if len(self.corpus) < self.bytes_per_image + 16:
            raise ValueError(
                f"Corpus too small: {len(self.corpus):,} bytes < "
                f"required {self.bytes_per_image:,} per image.")
        print(f"  [ByteTrigramDataset] Corpus: {len(self.corpus):,} bytes "
              f"({len(self.corpus) / 1e6:.1f} MB), "
              f"{self.bytes_per_image:,} bytes/image, "
              f"{len(self.corpus) // self.bytes_per_image:,} non-overlapping "
              f"images available "
              f"({len(self.corpus) - self.bytes_per_image:,} valid window starts)")

    def __len__(self):
        return self.size

    @staticmethod
    def bytes_to_image(byte_chunk: np.ndarray, img_size: int,
                       patch_size: int = 4,
                       channels: int = 3) -> np.ndarray:
        """``[bytes_per_image]`` uint8 → ``[channels, img_size, img_size]`` float32 in [-1, 1].

        Layout: byte stream packs into cells in row-major-across-patches,
        row-major-within-patch order. Cell ``i`` holds bytes
        ``byte_chunk[C*i : C*i + C]`` as the C-tuple of channel values,
        where C = ``channels``.
        """
        gh = gw = img_size // patch_size
        cells_per_patch = patch_size * patch_size
        n_patches = gh * gw
        # Reshape: [n_patches, cells_per_patch, channels]
        rgb = byte_chunk.reshape(n_patches, cells_per_patch, channels).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5  # → [-1, 1]
        per_patch = rgb.reshape(n_patches, patch_size, patch_size, channels)
        grid = per_patch.reshape(gh, gw, patch_size, patch_size, channels)
        # Permute (gh, gw, ps_r, ps_c, channel) → (channel, gh, ps_r, gw, ps_c)
        img = grid.transpose(4, 0, 2, 1, 3)
        img = img.reshape(channels, img_size, img_size)
        return img

    @staticmethod
    def image_to_bytes(images: torch.Tensor, patch_size: int = 4,
                        channels: int = 3) -> torch.Tensor:
        """``[B, C, H, W]`` float → ``[B, n_cells_total, C]`` in {0..255}.

        Inverse of ``bytes_to_image`` for the same ``patch_size`` and
        ``channels``. Maps continuous [-1, 1] back to rounded uint8 byte
        values. C = ``channels``.
        """
        B, C, H, W = images.shape
        assert C == channels and H == W and H % patch_size == 0, (
            f"Need square C={channels}-ch image div by ps; got "
            f"{tuple(images.shape)}, ps={patch_size}, channels={channels}"
        )
        gh = gw = H // patch_size
        ps = patch_size
        # (B, C, gh, ps, gw, ps) → (B, gh, gw, ps_r, ps_c, channel)
        x = images.reshape(B, channels, gh, ps, gw, ps)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, gh * gw * ps * ps, channels)
        # Recover bytes: float in [-1, 1] → byte in [0, 255]
        bytes_f = x * 127.5 + 127.5
        return bytes_f.clamp(0, 255).round().to(torch.uint8)

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.default_rng(int.from_bytes(os.urandom(4), 'big'))

        # Sample a random window of bytes_per_image consecutive bytes
        max_start = len(self.corpus) - self.bytes_per_image
        start = int(self._rng.integers(0, max_start + 1))
        chunk = self.corpus[start:start + self.bytes_per_image]
        img = self.bytes_to_image(
            chunk, self.img_size, self.patch_size, self.channels,
        )
        return torch.from_numpy(img), 0


def byte_recovery_metrics(orig_bytes: torch.Tensor,
                           recon_bytes: torch.Tensor) -> Dict[str, Any]:
    """Per-byte and per-n-gram recovery metrics.

    Args:
        orig_bytes:  ``[B, n_cells, C]`` uint8 (output of image_to_bytes on input)
        recon_bytes: ``[B, n_cells, C]`` uint8 (output of image_to_bytes on recon)

    The last dim ``C`` is the n-gram size (3 for the canonical RGB trigram
    encoding, parameterizable per ByteTrigramDataset.channels). The function
    infers C from the input shape — no kwarg needed.

    Returns:
        per_byte_acc:        fraction of bytes recovered exactly
        per_byte_l1:         mean |orig - recon| in byte units (0..255)
        trigram_exact_rate:  fraction of n-grams (cells) with ALL bytes exact.
                             Name kept for back-compat; semantics generalize
                             to whatever C is.
        per_channel_acc:     ``[C]`` per-channel exact-recovery rates separately
    """
    orig = orig_bytes.to(torch.int32)
    recon = recon_bytes.to(torch.int32)
    diff = (orig - recon).abs()
    correct = (diff == 0).float()  # [B, n_cells, C]

    per_byte_acc = correct.mean().item()
    per_byte_l1 = diff.float().mean().item()
    # Trigram exact: all 3 bytes per cell match
    # All bytes in the n-gram cell match (channel-count-agnostic).
    trigram_exact = correct.all(dim=-1).float()                 # [B, n_cells]
    trigram_exact_rate = trigram_exact.mean().item()
    per_channel_acc = correct.mean(dim=(0, 1)).tolist()         # [C]

    return {
        'per_byte_acc': per_byte_acc,
        'per_byte_l1': per_byte_l1,
        'trigram_exact_rate': trigram_exact_rate,
        'per_channel_acc': per_channel_acc,
    }


# ═══════════════════════════════════════════════════════════════════
# PER-TYPE EVALUATION
# ═══════════════════════════════════════════════════════════════════

def eval_per_type(model, active_types, img_size, device, n_per_type=64,
                   channels: int = 3):
    """MSE for each active noise type."""
    rng = np.random.RandomState(99)
    model.eval()
    results = {}
    with torch.no_grad():
        for t in active_types:
            imgs = torch.stack([
                _generate_noise(t, img_size, rng, channels).clamp(-4, 4)
                for _ in range(n_per_type)
            ]).to(device)
            out = model(imgs)
            results[t] = F.mse_loss(out['recon'], imgs).item()
    return results


# ═══════════════════════════════════════════════════════════════════
# DATASET FACTORY LAYER
# ═══════════════════════════════════════════════════════════════════
# Single dispatch surface used by the trainer. Each factory function
# takes (cfg: dict, channels: int) and returns a DatasetBundle with
# train_loader, test_loader, and the appropriate is_* kind flags set.
#
# Adding a new dataset: write a factory function below, register it
# in DATASET_FACTORIES, done. The trainer's only contact with dataset
# code is `bundle = get_dataset_bundle(cfg, channels)`.


@dataclass
class DatasetBundle:
    """Train + test loaders plus kind-flags consumed by the trainer's
    per-kind logging and evaluation branches.
    """
    train_loader: torch.utils.data.DataLoader
    test_loader:  torch.utils.data.DataLoader
    is_noise:         bool = False
    is_text:          bool = False
    is_image:         bool = False
    is_tree:          bool = False
    is_sentencepiece: bool = False
    is_byte_trigram:  bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Factory functions ──────────────────────────────────────────────


def _imagenet_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    name = cfg['dataset']
    train_loader, test_loader, _, _ = get_image_loaders(
        name, cfg['img_size'], cfg['batch_size'],
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_image=True,
    )


def _curriculum_noise_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    ds_size = cfg.get('ds_size', 500_000)
    val_size = cfg.get('val_size', 10_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    allowed_types = cfg.get('allowed_types', None)

    train_ds = CurriculumNoiseDataset(
        size=ds_size, img_size=img_size,
        allowed_types=allowed_types, channels=channels,
    )
    val_ds = CurriculumNoiseDataset(
        size=val_size, img_size=img_size,
        allowed_types=allowed_types, channels=channels,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_noise=True,
    )


def _omega_noise_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    """Handles both 'omega_noise' (always-on 16 types or allowed_types
    subset) and 'scheduled_noise' (curriculum-on-noise — uses
    CurriculumNoiseDataset under the hood)."""
    ds_size = cfg.get('ds_size', 1_280_000)
    val_size = cfg.get('val_size', 10_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    allowed_types = cfg.get('allowed_types', None)

    if cfg['dataset'] == 'scheduled_noise':
        cls = CurriculumNoiseDataset
    else:
        cls = OmegaNoiseDataset

    train_ds = cls(
        size=ds_size, img_size=img_size,
        allowed_types=allowed_types, channels=channels,
    )
    val_ds = cls(
        size=val_size, img_size=img_size,
        allowed_types=allowed_types, channels=channels,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_noise=True,
    )


def _wikipedia_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    ds_size = cfg.get('ds_size', 200_000)
    val_size = cfg.get('val_size', 5_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    print(f"\n  Loading Wikipedia corpus...")
    train_ds = WikiTextAsImage(size=ds_size, img_size=img_size, split='train')
    val_ds = WikiTextAsImage(size=val_size, img_size=img_size, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_text=True,
    )


def _binary_tree_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    ds_size = cfg.get('ds_size', 200_000)
    val_size = cfg.get('val_size', 2_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    tree_depth = cfg.get('tree_depth', 4)

    train_ds = BinaryTreeDataset(size=ds_size, img_size=img_size, depth=tree_depth)
    val_ds = BinaryTreeDataset(size=val_size, img_size=img_size, depth=tree_depth, seed=999)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_tree=True,
    )


def _sentencepiece_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    ds_size = cfg.get('ds_size', 200_000)
    val_size = cfg.get('val_size', 2_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    sp_tokenizer = cfg.get('sp_tokenizer', 't5-base')
    sp_corpus = cfg.get('sp_corpus', 'wikitext-2-raw-v1')
    sp_n_bits = cfg.get('sp_n_bits', 16)

    train_ds = SentencePieceBitDataset(
        size=ds_size, img_size=img_size,
        tokenizer_id=sp_tokenizer, corpus_id=sp_corpus,
        n_bits=sp_n_bits, seed=42,
    )
    val_ds = SentencePieceBitDataset(
        size=val_size, img_size=img_size,
        tokenizer_id=sp_tokenizer, corpus_id=sp_corpus,
        n_bits=sp_n_bits, seed=999,
    )
    # num_workers=0 because the tokenized corpus + sp model are too large to
    # copy into worker processes cheaply, and __getitem__ is fast.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_sentencepiece=True,
    )


def _byte_trigram_factory(cfg: Dict[str, Any], channels: int) -> DatasetBundle:
    ds_size = cfg.get('ds_size', 100_000)
    val_size = cfg.get('val_size', 1_000)
    img_size = cfg['img_size']
    bs = cfg['batch_size']
    patch_size = cfg['patch_size']
    bt_corpus = cfg.get('bt_corpus', 'wikitext-103-raw-v1')
    bt_max_corpus_bytes = cfg.get('bt_max_corpus_bytes', None)

    train_ds = ByteTrigramDataset(
        size=ds_size, img_size=img_size, patch_size=patch_size,
        channels=channels, corpus_id=bt_corpus, seed=42,
        max_corpus_bytes=bt_max_corpus_bytes,
    )
    val_ds = ByteTrigramDataset(
        size=val_size, img_size=img_size, patch_size=patch_size,
        channels=channels, corpus_id=bt_corpus, seed=999,
        max_corpus_bytes=bt_max_corpus_bytes,
    )
    # num_workers=4: ByteTrigramDataset just indexes into a uint8 numpy
    # array — workers fork via copy-on-write without ballooning RAM, and
    # the batch-level data-load wallclock at large batch can match or
    # exceed GPU compute time.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return DatasetBundle(
        train_loader=train_loader, test_loader=test_loader, is_byte_trigram=True,
    )


# ── Registry ────────────────────────────────────────────────────────

DATASET_FACTORIES: Dict[str, Callable[[Dict[str, Any], int], DatasetBundle]] = {
    # Image
    'tiny_imagenet':     _imagenet_factory,
    'imagenet_128':      _imagenet_factory,
    'imagenet_256':      _imagenet_factory,
    # Noise
    'curriculum_noise':  _curriculum_noise_factory,
    'omega_noise':       _omega_noise_factory,
    'scheduled_noise':   _omega_noise_factory,
    # Text / substrate
    'wikipedia':         _wikipedia_factory,
    'binary_tree':       _binary_tree_factory,
    'sentencepiece_bits':_sentencepiece_factory,
    'byte_trigram':      _byte_trigram_factory,
}


def get_dataset_bundle(cfg: Dict[str, Any], channels: int = 3) -> DatasetBundle:
    """Single dispatch into ``DATASET_FACTORIES``.

    Replaces the if/elif dataset-dispatch block the trainer used to carry
    inline. Channel count is threaded through to every factory so
    channel-aware datasets (noise, byte_trigram) emit C-channel tensors;
    channel-agnostic ones (image / text / tree / sentencepiece) ignore
    the kwarg.
    """
    name = cfg.get('dataset')
    if name not in DATASET_FACTORIES:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Known: {sorted(DATASET_FACTORIES)}"
        )
    return DATASET_FACTORIES[name](cfg, channels=channels)


__all__ = [
    # Constants
    'NOISE_NAMES', 'TIERS',
    # Noise machinery
    '_pink_noise', '_brown_noise', '_generate_noise',
    'CurriculumNoiseDataset', 'OmegaNoiseDataset',
    # Image
    'HFImageDataset', 'get_image_loaders',
    # Text
    'WikiTextAsImage',
    # Substrate prototypes
    'BinaryTreeDataset', 'decode_image_to_trees', 'bit_recovery_metrics',
    'SentencePieceBitDataset', 'decode_image_to_tokens',
    'token_bit_recovery_metrics',
    'ByteTrigramDataset', 'byte_recovery_metrics',
    # Eval
    'eval_per_type',
    # Factory layer
    'DatasetBundle', 'DATASET_FACTORIES', 'get_dataset_bundle',
]

"""
Freckles Stress Test — Extreme Resolution & Out-of-Scope Noise
================================================================
Push Freckles to resolutions she's never seen, noise types she's
never trained on, and spatial structures that mix multiple types.

Tests:
  1. Extreme Resolution: 256, 512, 1024, 2048, 4096, 8192
     Plus weird sizes: 36, 52, 76, 100, 140, 172, 204, 300, 444, 600
  2. Out-of-Scope Noise: 16 novel types not in training set
  3. Spatial Matte: large images with regional noise zones
  4. Noise Triangulation: can omega tokens identify WHICH noise
     is WHERE without any classification training?
  5. Multi-noise Composites: layered noise at varying intensities

Usage:
    python freckles_stress_test.py --checkpoint /path/to/best.pt

Colab:
    !python -m geolip_svae.noise_stress_test --model v41_freckles_256
"""

import os
import math
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# KNOWN NOISE (the 16 training types for reference fingerprints)
# ═══════════════════════════════════════════════════════════════

KNOWN_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_sc', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse',
    8: 'block', 9: 'gradient', 10: 'checker', 11: 'mixed',
    12: 'structural', 13: 'cauchy', 14: 'exponential', 15: 'laplace',
}


def _pink(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / torch.sqrt(fx**2 + fy**2).clamp(min=1e-8), s=(h, ww))


def _brown(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / (fx**2 + fy**2).clamp(min=1e-8), s=(h, ww))


def gen_known_noise(t, s, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    if t == 0: return torch.randn(3, s, s)
    elif t == 1: return torch.rand(3, s, s) * 2 - 1
    elif t == 2: return (torch.rand(3, s, s) - 0.5) * 4
    elif t == 3:
        lam = rng.uniform(0.5, 20.0)
        return torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0
    elif t == 4:
        img = _pink((3, s, s)); return img / (img.std() + 1e-8)
    elif t == 5:
        img = _brown((3, s, s)); return img / (img.std() + 1e-8)
    elif t == 6:
        return torch.where(torch.rand(3, s, s) > 0.5,
                          torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2) + torch.randn(3, s, s) * 0.1
    elif t == 7:
        return torch.randn(3, s, s) * (torch.rand(3, s, s) > 0.9).float() * 3
    elif t == 8:
        b = rng.randint(2, max(3, s // 4))
        sm = torch.randn(3, s // b + 1, s // b + 1)
        return F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif t == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        a = rng.uniform(0, 2 * math.pi)
        return (math.cos(a) * gx + math.sin(a) * gy).unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.5
    elif t == 10:
        cs = rng.randint(2, max(3, s // 4))
        cy = torch.arange(s) // cs; cx = torch.arange(s) // cs
        return ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float().unsqueeze(0).expand(3, -1, -1) * 2 - 1 + torch.randn(3, s, s) * 0.3
    elif t == 11:
        alpha = rng.uniform(0.2, 0.8)
        return alpha * torch.randn(3, s, s) + (1 - alpha) * (torch.rand(3, s, s) * 2 - 1)
    elif t == 12:
        img = torch.zeros(3, s, s); h2 = s // 2; w2 = s // 2
        img[:, :h2, :w2] = torch.randn(3, h2, w2)
        img[:, :h2, w2:s] = (torch.rand(3, h2, s - w2) * 2 - 1)
        img[:, h2:s, :w2] = _pink((3, s - h2, w2)) / 2
        img[:, h2:s, w2:s] = torch.where(torch.rand(3, s - h2, s - w2) > 0.5,
                                           torch.ones(3, s - h2, s - w2), -torch.ones(3, s - h2, s - w2))
        return img
    elif t == 13:
        return torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3)
    elif t == 14:
        return torch.empty(3, s, s).exponential_(1.0) - 1.0
    elif t == 15:
        u = torch.rand(3, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    return torch.randn(3, s, s)


# ═══════════════════════════════════════════════════════════════
# OUT-OF-SCOPE NOISE TYPES (never trained on)
# ═══════════════════════════════════════════════════════════════

OOD_NAMES = {
    100: 'log_normal',
    101: 'beta',
    102: 'weibull',
    103: 'gumbel',
    104: 'rayleigh',
    105: 'perlin_approx',
    106: 'wavelet_noise',
    107: 'fractal_fbm',
    108: 'gabor',
    109: 'sine_composite',
    110: 'voronoi_approx',
    111: 'shot_noise',
    112: 'quantize_noise',
    113: 'jpeg_artifact',
    114: 'ring_noise',
    115: 'spiral_noise',
}


def gen_ood_noise(t, s, rng=None):
    """Generate out-of-distribution noise types."""
    if rng is None: rng = np.random.RandomState(42)

    if t == 100:  # log-normal
        return torch.exp(torch.randn(3, s, s) * 0.5) - 1.5

    elif t == 101:  # beta
        a = torch.distributions.Beta(0.5, 0.5)
        return a.sample((3, s, s)) * 4 - 2

    elif t == 102:  # weibull
        u = torch.rand(3, s, s).clamp(min=1e-8)
        k = 1.5
        return (-torch.log(u)).pow(1.0 / k) - 1.0

    elif t == 103:  # gumbel
        u = torch.rand(3, s, s).clamp(1e-8, 1 - 1e-8)
        return -(torch.log(-torch.log(u))) * 0.5

    elif t == 104:  # rayleigh
        return torch.sqrt(-2 * torch.log(torch.rand(3, s, s).clamp(min=1e-8))) - 1.0

    elif t == 105:  # perlin approximation (multi-octave interpolated noise)
        img = torch.zeros(3, s, s)
        for octave in range(5):
            freq = 2 ** octave
            amp = 0.5 ** octave
            small = torch.randn(3, max(2, s // (freq * 4) + 1), max(2, s // (freq * 4) + 1))
            up = F.interpolate(small.unsqueeze(0), size=s, mode='bilinear', align_corners=False).squeeze(0)
            img = img + amp * up
        return img / (img.std() + 1e-8)

    elif t == 106:  # wavelet-like (high frequency bands)
        img = torch.randn(3, s, s)
        # Kill low frequencies
        S = torch.fft.rfft2(img)
        h, w = s, s // 2 + 1
        mask = torch.ones(h, w)
        cutoff = max(1, s // 16)
        mask[:cutoff, :cutoff] = 0
        return torch.fft.irfft2(S * mask.unsqueeze(0), s=(s, s))

    elif t == 107:  # fractional Brownian motion approximation
        img = torch.zeros(3, s, s)
        H = 0.7  # Hurst exponent
        for octave in range(8):
            freq = 2 ** octave
            amp = freq ** (-H)
            small = torch.randn(3, max(2, s // freq + 1), max(2, s // freq + 1))
            up = F.interpolate(small.unsqueeze(0), size=s, mode='bilinear', align_corners=False).squeeze(0)
            img = img + amp * up
        return img / (img.std() + 1e-8)

    elif t == 108:  # gabor-like (oriented sinusoidal + gaussian envelope)
        y = torch.linspace(-3, 3, s).unsqueeze(1).expand(s, s)
        x = torch.linspace(-3, 3, s).unsqueeze(0).expand(s, s)
        angle = rng.uniform(0, math.pi)
        freq = rng.uniform(2, 10)
        xr = x * math.cos(angle) + y * math.sin(angle)
        envelope = torch.exp(-(x**2 + y**2) / 2)
        gabor = envelope * torch.cos(2 * math.pi * freq * xr)
        return gabor.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.2

    elif t == 109:  # sine composite (multiple frequencies)
        y = torch.linspace(0, 1, s).unsqueeze(1).expand(s, s)
        x = torch.linspace(0, 1, s).unsqueeze(0).expand(s, s)
        img = torch.zeros(3, s, s)
        for c in range(3):
            for _ in range(5):
                fx = rng.uniform(1, 20)
                fy = rng.uniform(1, 20)
                phase = rng.uniform(0, 2 * math.pi)
                img[c] += torch.sin(2 * math.pi * (fx * x + fy * y) + phase)
        return img / (img.std() + 1e-8)

    elif t == 110:  # voronoi approximation (nearest-seed distance)
        n_seeds = rng.randint(10, 50)
        seeds = torch.rand(n_seeds, 2) * s
        y = torch.arange(s).float().unsqueeze(1).expand(s, s)
        x = torch.arange(s).float().unsqueeze(0).expand(s, s)
        coords = torch.stack([y, x], dim=-1).reshape(-1, 2)
        dists = torch.cdist(coords, seeds)
        min_dist = dists.min(dim=1).values.reshape(s, s)
        img = min_dist / (min_dist.max() + 1e-8) * 4 - 2
        return img.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.1

    elif t == 111:  # shot noise (poisson with very low lambda)
        lam = rng.uniform(0.01, 0.2)
        return torch.poisson(torch.full((3, s, s), lam)) * 3.0 - lam * 3

    elif t == 112:  # quantization noise
        levels = rng.randint(2, 8)
        base = torch.randn(3, s, s)
        quantized = torch.round(base * levels) / levels
        return quantized + torch.randn(3, s, s) * 0.05

    elif t == 113:  # jpeg-like block artifact (8×8 blocks with discontinuities)
        bs = 8
        img = torch.randn(3, s, s)
        small = F.avg_pool2d(img.unsqueeze(0), bs, bs)
        img = F.interpolate(small, size=s, mode='nearest').squeeze(0)
        return img + torch.randn(3, s, s) * 0.2

    elif t == 114:  # ring/radial noise
        y = torch.linspace(-1, 1, s).unsqueeze(1).expand(s, s)
        x = torch.linspace(-1, 1, s).unsqueeze(0).expand(s, s)
        r = torch.sqrt(x**2 + y**2)
        freq = rng.uniform(3, 15)
        rings = torch.sin(2 * math.pi * freq * r)
        return rings.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.3

    elif t == 115:  # spiral noise
        y = torch.linspace(-1, 1, s).unsqueeze(1).expand(s, s)
        x = torch.linspace(-1, 1, s).unsqueeze(0).expand(s, s)
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        freq = rng.uniform(2, 8)
        spiral = torch.sin(2 * math.pi * freq * r + 3 * theta)
        return spiral.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.2

    return torch.randn(3, s, s)


# ═══════════════════════════════════════════════════════════════
# SPATIAL MATTE GENERATORS
# ═══════════════════════════════════════════════════════════════

def gen_zone_matte(s, n_zones=4, rng=None):
    """Generate image with distinct noise zones and a ground-truth zone map.

    Returns:
        image: (3, s, s) — composite noise image
        zone_map: (s, s) — integer map of which zone each pixel belongs to
        zone_types: list of noise type IDs per zone
    """
    if rng is None: rng = np.random.RandomState(42)

    image = torch.zeros(3, s, s)
    zone_map = torch.zeros(s, s, dtype=torch.long)

    if n_zones == 4:
        # Quadrant split
        h2 = s // 2
        types = rng.choice(16, size=4, replace=False).tolist()
        image[:, :h2, :h2] = gen_known_noise(types[0], h2, rng).clamp(-4, 4)
        image[:, :h2, h2:] = gen_known_noise(types[1], h2, rng).clamp(-4, 4)
        image[:, h2:, :h2] = gen_known_noise(types[2], h2, rng).clamp(-4, 4)
        image[:, h2:, h2:] = gen_known_noise(types[3], h2, rng).clamp(-4, 4)
        zone_map[:h2, :h2] = 0
        zone_map[:h2, h2:] = 1
        zone_map[h2:, :h2] = 2
        zone_map[h2:, h2:] = 3
        return image, zone_map, types

    elif n_zones == 9:
        # 3×3 grid
        h3 = s // 3
        types = rng.choice(16, size=9, replace=False).tolist()
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                r0, r1 = i * h3, (i + 1) * h3 if i < 2 else s
                c0, c1 = j * h3, (j + 1) * h3 if j < 2 else s
                h = r1 - r0; w = c1 - c0
                noise = gen_known_noise(types[idx], max(h, w), rng).clamp(-4, 4)
                image[:, r0:r1, c0:c1] = noise[:, :h, :w]
                zone_map[r0:r1, c0:c1] = idx
        return image, zone_map, types

    elif n_zones == 16:
        # 4×4 grid — all 16 types
        h4 = s // 4
        types = list(range(16))
        rng.shuffle(types)
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                r0, r1 = i * h4, (i + 1) * h4 if i < 3 else s
                c0, c1 = j * h4, (j + 1) * h4 if j < 3 else s
                h = r1 - r0; w = c1 - c0
                noise = gen_known_noise(types[idx], max(h, w), rng).clamp(-4, 4)
                image[:, r0:r1, c0:c1] = noise[:, :h, :w]
                zone_map[r0:r1, c0:c1] = idx
        return image, zone_map, types

    # Random blobs
    types = rng.choice(16, size=n_zones, replace=True).tolist()
    centers = torch.rand(n_zones, 2) * s
    y = torch.arange(s).float().unsqueeze(1).expand(s, s)
    x = torch.arange(s).float().unsqueeze(0).expand(s, s)
    coords = torch.stack([y, x], dim=-1)
    dists = torch.cdist(coords.reshape(-1, 2), centers).reshape(s, s, n_zones)
    zone_map = dists.argmin(dim=-1)
    for z in range(n_zones):
        mask = (zone_map == z)
        region_size = int(mask.sum().sqrt().item())
        if region_size < 4:
            region_size = 4
        noise = gen_known_noise(types[z], region_size, rng).clamp(-4, 4)
        noise_full = F.interpolate(noise.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        for c in range(3):
            image[c][mask] = noise_full[c][mask]
    return image, zone_map, types


# ═══════════════════════════════════════════════════════════════
# PATCH / TILE UTILITIES
# ═══════════════════════════════════════════════════════════════

def extract_patches(images, ps=4):
    B, C, H, W = images.shape
    gh, gw = H // ps, W // ps
    p = images.reshape(B, C, gh, ps, gw, ps)
    return p.permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, C * ps * ps), gh, gw


def stitch_patches(patches, gh, gw, ps=4):
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, 3, ps, ps)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * ps, gw * ps)


@torch.no_grad()
def tile_encode_full(model, image, tile_size=64, ps=4, device='cuda'):
    """Tile-encode any resolution image. Returns omega grid + SVD cache."""
    C, H, W = image.shape
    # Pad to tile_size multiple
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    C, H, W = image.shape

    tiles_h, tiles_w = H // tile_size, W // tile_size
    gh_tile, gw_tile = tile_size // ps, tile_size // ps

    all_S, all_S_orig = [], []
    for th in range(tiles_h):
        for tw in range(tiles_w):
            tile = image[:, th * tile_size:(th + 1) * tile_size,
                            tw * tile_size:(tw + 1) * tile_size]
            tile = tile.unsqueeze(0).to(device)
            out = model(tile)
            all_S.append(out['svd']['S'].cpu())
            all_S_orig.append(out['svd']['S_orig'].cpu())

    # Reshape into full grid
    S_tiles = torch.cat(all_S, dim=1)           # (1, total_patches, D)
    S_orig = torch.cat(all_S_orig, dim=1)
    gh_full = tiles_h * gh_tile
    gw_full = tiles_w * gw_tile
    return S_tiles, S_orig, gh_full, gw_full


# ═══════════════════════════════════════════════════════════════
# REFERENCE FINGERPRINTS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_reference_fingerprints(model, device, ps=4, n_samples=64):
    """Compute mean omega token profile for each of the 16 known noise types."""
    rng = np.random.RandomState(42)
    refs = {}
    for t in range(16):
        all_S = []
        for _ in range(n_samples):
            img = gen_known_noise(t, 64, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out = model(img)
            # Mean omega token across all patches
            all_S.append(out['svd']['S'].mean(dim=1))
        refs[t] = torch.cat(all_S, dim=0).mean(0)  # (D,)
    return refs


def classify_omega(omega, refs):
    """Classify an omega token against reference fingerprints.

    Args:
        omega: (D,) single omega token
        refs: dict {type_id: (D,) reference}

    Returns:
        best_type, confidence (cosine similarity)
    """
    best_sim, best_t = -1, 0
    for t, ref in refs.items():
        sim = F.cosine_similarity(omega.unsqueeze(0), ref.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_t = t
    return best_t, best_sim


# ═══════════════════════════════════════════════════════════════
# TEST 1: EXTREME RESOLUTIONS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_extreme_resolution(model, device, ps=4):
    """Tile-encode at extreme resolutions, measure reconstruction and timing."""
    print("\n" + "=" * 70)
    print("TEST 1: Extreme Resolution Scaling")
    print("=" * 70)

    resolutions = [36, 52, 64, 76, 100, 128, 140, 172, 204, 256, 300,
                   444, 512, 600, 1024, 2048, 4096]
    rng = np.random.RandomState(42)
    results = {}

    for res in resolutions:
        # Skip if resolution not divisible by ps
        if res % ps != 0:
            print(f"  {res}×{res} — skipped (not divisible by {ps})")
            continue

        n_patches = (res // ps) ** 2
        # Use smaller tile for very large images
        tile = min(64, res)
        if res % tile != 0:
            tile = res  # fall back to full if not tileable

        try:
            t0 = time.time()
            # Test with gaussian
            img = gen_known_noise(0, res, rng).clamp(-4, 4)

            if res <= 256:
                # Direct encoding
                img_batch = img.unsqueeze(0).to(device)
                out = model(img_batch)
                mse = F.mse_loss(out['recon'], img_batch).item()
            else:
                # Tile encoding
                S, S_orig, gh, gw = tile_encode_full(model, img, tile_size=tile, ps=ps, device=device)
                mse = -1  # can't easily compute pixel MSE for tiled without full decode

            elapsed = time.time() - t0
            mem_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

            results[res] = {
                'patches': n_patches,
                'mse': mse,
                'time_s': elapsed,
                'mem_mb': mem_mb,
            }
            mse_str = f"{mse:.6f}" if mse >= 0 else "tile-only"
            print(f"  {res:>5d}×{res:<5d} {n_patches:>8d} patches | "
                  f"MSE={mse_str:>12s} | {elapsed:.2f}s | {mem_mb:.0f}MB")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        except Exception as e:
            print(f"  {res:>5d}×{res:<5d} — FAILED: {e}")
            results[res] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════
# TEST 2: OUT-OF-SCOPE NOISE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_ood_noise(model, device, ps=4, n_samples=8):
    """Test reconstruction of noise types never seen during training."""
    print("\n" + "=" * 70)
    print("TEST 2: Out-of-Distribution Noise Types")
    print("=" * 70)

    rng = np.random.RandomState(77)
    results = {}

    # Known baseline
    known_mses = {}
    for t in range(16):
        imgs = torch.stack([gen_known_noise(t, 64, rng).clamp(-4, 4)
                            for _ in range(n_samples)]).to(device)
        out = model(imgs)
        known_mses[KNOWN_NAMES[t]] = F.mse_loss(out['recon'], imgs).item()

    known_avg = np.mean(list(known_mses.values()))
    print(f"\n  Known noise avg MSE: {known_avg:.6f}")

    # OOD types
    for t in sorted(OOD_NAMES.keys()):
        imgs = torch.stack([gen_ood_noise(t, 64, rng).clamp(-4, 4)
                            for _ in range(n_samples)]).to(device)
        out = model(imgs)
        mse = F.mse_loss(out['recon'], imgs).item()

        # Effective rank of OOD
        erank = model.effective_rank(out['svd']['S'].reshape(-1, model.D)).mean().item()

        ratio = mse / (known_avg + 1e-10)
        results[OOD_NAMES[t]] = {'mse': mse, 'erank': erank, 'ratio_vs_known': ratio}
        status = "✓ handles" if ratio < 10 else "△ degrades" if ratio < 100 else "✗ fails"
        print(f"  {OOD_NAMES[t]:<18s} MSE={mse:.6f} er={erank:.2f} "
              f"ratio={ratio:.1f}x {status}")

    return results


# ═══════════════════════════════════════════════════════════════
# TEST 3: SPATIAL MATTE — NOISE TRIANGULATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_noise_triangulation(model, device, ps=4):
    """Can omega tokens identify which noise is where?

    Generate zoned images, encode, classify each patch against
    reference fingerprints. Measure classification accuracy WITHOUT
    any classification training — purely spectral similarity.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Noise Triangulation (zero-shot spatial classification)")
    print("=" * 70)

    # Build reference fingerprints
    refs = compute_reference_fingerprints(model, device, ps=ps)
    print(f"  Reference fingerprints computed for {len(refs)} types")

    rng = np.random.RandomState(999)
    results = {}

    for n_zones, res in [(4, 128), (4, 256), (4, 512),
                          (9, 128), (9, 256),
                          (16, 256), (16, 512)]:
        if res % ps != 0:
            continue

        accuracies = []
        for trial in range(5):
            image, zone_map, zone_types = gen_zone_matte(res, n_zones, rng)
            image = image.clamp(-4, 4)

            # Encode
            if res <= 256:
                img_batch = image.unsqueeze(0).to(device)
                out = model(img_batch)
                S = out['svd']['S']  # (1, N, D)
            else:
                S, _, gh, gw = tile_encode_full(model, image, tile_size=64, ps=ps, device=device)

            # Classify each patch
            gh_full = res // ps
            gw_full = res // ps
            correct, total = 0, 0

            for pi in range(S.shape[1]):
                # Patch grid position
                row = (pi // gw_full) * ps + ps // 2
                col = (pi % gw_full) * ps + ps // 2
                row = min(row, res - 1)
                col = min(col, res - 1)

                # Ground truth zone
                gt_zone = zone_map[row, col].item()
                gt_type = zone_types[gt_zone]

                # Classify by omega fingerprint
                omega = S[0, pi].to(device)
                pred_type, conf = classify_omega(omega, refs)

                if pred_type == gt_type:
                    correct += 1
                total += 1

            acc = correct / total
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        key = f"{n_zones}zones_{res}px"
        results[key] = {'accuracy': mean_acc, 'std': std_acc, 'n_zones': n_zones, 'res': res}
        print(f"  {key:<20s} acc={mean_acc:.1%} ± {std_acc:.1%} "
              f"({res // ps}×{res // ps} grid, {n_zones} zones)")

    return results


# ═══════════════════════════════════════════════════════════════
# TEST 4: MULTI-NOISE COMPOSITES
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_composites(model, device, ps=4, n_samples=8):
    """Test reconstruction of layered noise composites."""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Noise Composites")
    print("=" * 70)

    rng = np.random.RandomState(555)
    results = {}

    composites = [
        ('gauss+pink', [(0, 0.5), (4, 0.5)]),
        ('gauss+salt', [(0, 0.7), (6, 0.3)]),
        ('pink+brown', [(4, 0.5), (5, 0.5)]),
        ('cauchy+laplace', [(13, 0.5), (15, 0.5)]),
        ('checker+gradient', [(10, 0.6), (9, 0.4)]),
        ('3-way: gauss+pink+block', [(0, 0.4), (4, 0.3), (8, 0.3)]),
        ('4-way: gauss+unif+pink+expo', [(0, 0.25), (1, 0.25), (4, 0.25), (14, 0.25)]),
        ('heavy: cauchy+salt+sparse', [(13, 0.4), (6, 0.3), (7, 0.3)]),
        ('gentle: pink+brown+gradient', [(4, 0.4), (5, 0.3), (9, 0.3)]),
        ('all_16_equal', [(t, 1/16) for t in range(16)]),
    ]

    for name, components in composites:
        mses = []
        for _ in range(n_samples):
            img = torch.zeros(3, 64, 64)
            for noise_type, weight in components:
                img = img + weight * gen_known_noise(noise_type, 64, rng).clamp(-4, 4)
            img = img.clamp(-4, 4).unsqueeze(0).to(device)
            out = model(img)
            mses.append(F.mse_loss(out['recon'], img).item())

        avg_mse = np.mean(mses)
        results[name] = {'mse': avg_mse, 'n_components': len(components)}
        print(f"  {name:<35s} MSE={avg_mse:.6f} ({len(components)} layers)")

    return results


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL (same as noise_diagnostic.py)
# ═══════════════════════════════════════════════════════════════

def load_freckles(model_path=None, hf_version=None, device='cuda'):
    if hf_version:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id='AbstractPhil/geolip-SVAE',
            filename=f'{hf_version}/checkpoints/best.pt',
            repo_type='model')
    else:
        ckpt_path = model_path

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    # Inline Freckles
    from geolip_core.linalg.eigh import FLEigh, _FL_MAX_N

    def gram_eigh_svd(A):
        B, M, N = A.shape
        orig_dtype = A.dtype
        if N <= _FL_MAX_N and A.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                A_d = A.double()
                G = torch.bmm(A_d.transpose(1, 2), A_d)
                eigenvalues, V = FLEigh()(G.float())
                eigenvalues = eigenvalues.double().flip(-1)
                V = V.double().flip(-1)
                S = torch.sqrt(eigenvalues.clamp(min=1e-24))
                U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
                Vh = V.transpose(-2, -1).contiguous()
            return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
        with torch.amp.autocast('cuda', enabled=False):
            A_d = A.double()
            G = torch.bmm(A_d.transpose(1, 2), A_d)
            G.diagonal(dim1=-2, dim2=-1).add_(1e-12)
            eigenvalues, V = torch.linalg.eigh(G)
            eigenvalues = eigenvalues.flip(-1); V = V.flip(-1)
            S = torch.sqrt(eigenvalues.clamp(min=1e-24))
            U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
            Vh = V.transpose(-2, -1).contiguous()
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

    class SpectralCrossAttention(nn.Module):
        def __init__(self, D, n_heads=2, max_alpha=0.2, alpha_init=-2.0):
            super().__init__()
            self.n_heads = n_heads; self.head_dim = D // n_heads
            self.max_alpha = max_alpha; assert D % n_heads == 0
            self.qkv = nn.Linear(D, 3 * D); self.out_proj = nn.Linear(D, D)
            self.norm = nn.LayerNorm(D); self.scale = self.head_dim ** -0.5
            self.alpha_logits = nn.Parameter(torch.full((D,), alpha_init))
        @property
        def alpha(self): return self.max_alpha * torch.sigmoid(self.alpha_logits)
        def forward(self, S):
            B, N, D = S.shape; S_n = self.norm(S)
            qkv = self.qkv(S_n).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            out = (((q @ k.transpose(-2, -1)) * self.scale).softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
            return S * (1.0 + self.alpha.unsqueeze(0).unsqueeze(0) * torch.tanh(self.out_proj(out)))

    class BoundarySmooth(nn.Module):
        def __init__(self, channels=3, mid=8):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(channels, mid, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(mid, channels, 3, padding=1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        def forward(self, x): return x + self.net(x)

    class Freckles(nn.Module):
        def __init__(self, V=48, D=4, ps=4, hidden=384, depth=4, n_cross=2):
            super().__init__()
            self.V, self.D, self.patch_size = V, D, ps
            self.patch_dim = 3 * ps * ps; self.mat_dim = V * D
            self.enc_in = nn.Linear(self.patch_dim, hidden)
            self.enc_blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
            self.enc_out = nn.Linear(hidden, self.mat_dim)
            nn.init.orthogonal_(self.enc_out.weight)
            self.dec_in = nn.Linear(self.mat_dim, hidden)
            self.dec_blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
            self.dec_out = nn.Linear(hidden, self.patch_dim)
            self.cross_attn = nn.ModuleList([
                SpectralCrossAttention(D, n_heads=min(2, D)) for _ in range(n_cross)])
            self.boundary_smooth = BoundarySmooth(channels=3, mid=8)

        def encode_patches(self, patches):
            B, N, _ = patches.shape
            h = F.gelu(self.enc_in(patches.reshape(B * N, -1)))
            for block in self.enc_blocks: h = h + block(h)
            M = F.normalize(self.enc_out(h).reshape(B * N, self.V, self.D), dim=-1)
            U, S, Vt = gram_eigh_svd(M)
            U = U.reshape(B, N, self.V, self.D); S = S.reshape(B, N, self.D)
            Vt = Vt.reshape(B, N, self.D, self.D); M = M.reshape(B, N, self.V, self.D)
            S_c = S
            for layer in self.cross_attn: S_c = layer(S_c)
            return {'U': U, 'S_orig': S, 'S': S_c, 'Vt': Vt, 'M': M}

        def decode_patches(self, U, S, Vt):
            B, N, V, D = U.shape
            M_hat = torch.bmm(U.reshape(B*N, V, D) * S.reshape(B*N, D).unsqueeze(1),
                              Vt.reshape(B*N, D, D))
            h = F.gelu(self.dec_in(M_hat.reshape(B * N, -1)))
            for block in self.dec_blocks: h = h + block(h)
            return self.dec_out(h).reshape(B, N, -1)

        def forward(self, images):
            patches, gh, gw = extract_patches(images, self.patch_size)
            svd = self.encode_patches(patches)
            decoded = self.decode_patches(svd['U'], svd['S'], svd['Vt'])
            recon = stitch_patches(decoded, gh, gw, self.patch_size)
            return {'recon': self.boundary_smooth(recon), 'svd': svd}

        @staticmethod
        def effective_rank(S):
            p = S / (S.sum(-1, keepdim=True) + 1e-8); p = p.clamp(min=1e-8)
            return (-(p * p.log()).sum(-1)).exp()

    model = Freckles(V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
                     hidden=cfg['hidden'], depth=cfg['depth'],
                     n_cross=cfg['n_cross_layers']).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    print(f"  Loaded Freckles: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Config: V={cfg['V']}, D={cfg['D']}, ps={cfg['patch_size']}")
    return model, cfg


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_stress_test(model, device, ps=4, save_path=None):
    print("\n" + "=" * 70)
    print("FRECKLES STRESS TEST — Extreme Resolution & OOD Noise")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    all_results['extreme_resolution'] = test_extreme_resolution(model, device, ps=ps)
    all_results['ood_noise'] = test_ood_noise(model, device, ps=ps)
    all_results['noise_triangulation'] = test_noise_triangulation(model, device, ps=ps)
    all_results['composites'] = test_composites(model, device, ps=ps)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"STRESS TEST COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    if save_path:
        def to_json(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): to_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json(v) for v in obj]
            return obj
        with open(save_path, 'w') as f:
            json.dump(to_json(all_results), f, indent=2)
        print(f"  Results saved: {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Freckles Stress Test')
    parser.add_argument('--model', type=str, default='v40_freckles_noise')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default='freckles_stress_test.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_freckles(model_path=args.checkpoint, hf_version=args.model, device=device)
    run_stress_test(model, device, ps=cfg['patch_size'], save_path=args.output)
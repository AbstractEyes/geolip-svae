"""
Freckles Piecemeal Noise Diagnostic
======================================
Tests whether 4×4 omega tokens are resolution-independent.

If a 4×4 patch produces the same spectral fingerprint regardless of
which image it came from, then:
  1. Encode at any resolution → downsample omega grid → decode at smaller res
  2. Encode at small res → upsample omega grid → decode at larger res
  3. Encode tiles of a large image → stitch omega grids → decode seamlessly

This diagnostic runs all permutations across noise types and resolutions.

Usage:
    python -m geolip_svae.noise_diagnostic --model v40_freckles_noise
    python -m geolip_svae.noise_diagnostic --checkpoint /path/to/best.pt
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


# ═══════════════════════════════════════════════════════════════
# NOISE GENERATORS (all 16 types)
# ═══════════════════════════════════════════════════════════════

NOISE_NAMES = {
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


def gen_noise(noise_type, s, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    if noise_type == 0: return torch.randn(3, s, s)
    elif noise_type == 1: return torch.rand(3, s, s) * 2 - 1
    elif noise_type == 2: return (torch.rand(3, s, s) - 0.5) * 4
    elif noise_type == 3:
        lam = rng.uniform(0.5, 20.0)
        return torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0
    elif noise_type == 4:
        img = _pink((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 5:
        img = _brown((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 6:
        return torch.where(torch.rand(3, s, s) > 0.5,
                          torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2) + torch.randn(3, s, s) * 0.1
    elif noise_type == 7:
        return torch.randn(3, s, s) * (torch.rand(3, s, s) > 0.9).float() * 3
    elif noise_type == 8:
        b = rng.randint(2, max(3, s // 2))
        sm = torch.randn(3, s // b + 1, s // b + 1)
        return F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif noise_type == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        a = rng.uniform(0, 2 * math.pi)
        return (math.cos(a) * gx + math.sin(a) * gy).unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.5
    elif noise_type == 10:
        cs = rng.randint(2, max(3, s // 2))
        cy = torch.arange(s) // cs; cx = torch.arange(s) // cs
        return ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float().unsqueeze(0).expand(3, -1, -1) * 2 - 1 + torch.randn(3, s, s) * 0.3
    elif noise_type == 11:
        alpha = rng.uniform(0.2, 0.8)
        return alpha * torch.randn(3, s, s) + (1 - alpha) * (torch.rand(3, s, s) * 2 - 1)
    elif noise_type == 12:
        img = torch.zeros(3, s, s); h2 = s // 2
        img[:, :h2, :h2] = torch.randn(3, h2, h2)
        img[:, :h2, h2:] = torch.rand(3, h2, h2) * 2 - 1
        img[:, h2:, :h2] = _pink((3, h2, h2)) / 2
        img[:, h2:, h2:] = torch.where(torch.rand(3, h2, h2) > 0.5,
                                         torch.ones(3, h2, h2), -torch.ones(3, h2, h2))
        return img
    elif noise_type == 13:
        return torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3)
    elif noise_type == 14:
        return torch.empty(3, s, s).exponential_(1.0) - 1.0
    elif noise_type == 15:
        u = torch.rand(3, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    return torch.randn(3, s, s)


# ═══════════════════════════════════════════════════════════════
# PATCH UTILITIES
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


# ═══════════════════════════════════════════════════════════════
# OMEGA GRID OPERATIONS
# ═══════════════════════════════════════════════════════════════

def omega_grid_to_2d(S, gh, gw):
    """Reshape (B, N, D) → (B, D, gh, gw) for spatial operations."""
    B, N, D = S.shape
    return S.permute(0, 2, 1).reshape(B, D, gh, gw)


def omega_2d_to_grid(S_2d, gh, gw):
    """Reshape (B, D, gh, gw) → (B, N, D)."""
    B, D = S_2d.shape[:2]
    return S_2d.reshape(B, D, gh * gw).permute(0, 2, 1)


def upsample_omega(S, gh_src, gw_src, gh_dst, gw_dst, mode='bilinear'):
    """Spatially upsample omega grid."""
    s2d = omega_grid_to_2d(S, gh_src, gw_src)
    up = F.interpolate(s2d, size=(gh_dst, gw_dst), mode=mode, align_corners=False)
    return omega_2d_to_grid(up, gh_dst, gw_dst)


def downsample_omega(S, gh_src, gw_src, gh_dst, gw_dst, mode='bilinear'):
    """Spatially downsample omega grid."""
    s2d = omega_grid_to_2d(S, gh_src, gw_src)
    down = F.interpolate(s2d, size=(gh_dst, gw_dst), mode=mode, align_corners=False)
    return omega_2d_to_grid(down, gh_dst, gw_dst)


def tile_encode(model, large_image, tile_size, ps=4, device='cuda'):
    """Encode a large image by tiling into tile_size chunks.

    Args:
        model: Freckles model
        large_image: (3, H, W) single image, H,W > tile_size
        tile_size: encode tile size (e.g., 64)
        ps: patch size (4)

    Returns:
        S_full: (1, N_total, D) stitched omega grid
        gh_full, gw_full: full grid dims
    """
    C, H, W = large_image.shape
    assert H % tile_size == 0 and W % tile_size == 0
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    gh_tile = tile_size // ps
    gw_tile = tile_size // ps

    all_S = []
    with torch.no_grad():
        for th in range(tiles_h):
            row_S = []
            for tw in range(tiles_w):
                tile = large_image[:, th * tile_size:(th + 1) * tile_size,
                                      tw * tile_size:(tw + 1) * tile_size]
                tile = tile.unsqueeze(0).to(device)
                out = model(tile)
                S = out['svd']['S']  # (1, gh_tile*gw_tile, D)
                # Reshape to 2D grid for spatial stitching
                S_2d = omega_grid_to_2d(S, gh_tile, gw_tile)  # (1, D, gh_tile, gw_tile)
                row_S.append(S_2d)
            # Concatenate along width
            row_cat = torch.cat(row_S, dim=3)  # (1, D, gh_tile, gw_tile * tiles_w)
            all_S.append(row_cat)
        # Concatenate along height
        full_S_2d = torch.cat(all_S, dim=2)  # (1, D, gh_full, gw_full)

    gh_full = gh_tile * tiles_h
    gw_full = gw_tile * tiles_w
    S_full = omega_2d_to_grid(full_S_2d, gh_full, gw_full)
    return S_full, gh_full, gw_full


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC TESTS
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def test_native_recon(model, device, resolutions, n_per_type=8, ps=4):
    """Test 1: Native encode/decode at each resolution."""
    print("\n" + "=" * 70)
    print("TEST 1: Native Reconstruction at Each Resolution")
    print("=" * 70)

    results = {}
    rng = np.random.RandomState(42)

    for res in resolutions:
        res_results = {}
        for t in range(16):
            imgs = torch.stack([gen_noise(t, res, rng).clamp(-4, 4)
                                for _ in range(n_per_type)]).to(device)
            out = model(imgs)
            mse = F.mse_loss(out['recon'], imgs).item()
            res_results[NOISE_NAMES[t]] = mse

        avg = np.mean(list(res_results.values()))
        worst = max(res_results.values())
        worst_name = max(res_results, key=res_results.get)
        n_patches = (res // ps) ** 2
        results[res] = {'per_type': res_results, 'avg': avg, 'worst': worst}

        print(f"\n  {res}×{res} ({n_patches} patches):")
        print(f"    avg={avg:.6f}  worst={worst:.6f} ({worst_name})")
        for name, mse in sorted(res_results.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {name}: {mse:.6f}")

    return results


@torch.no_grad()
def test_piecemeal_downsample(model, device, ps=4, n_per_type=4):
    """Test 2: Encode large → downsample omega grid → decode small.

    Encode 128×128 (1024 patches) → downsample omega to 256 patches → decode 64×64.
    Compare with native 64×64 encoding of the same noise (different realization).
    """
    print("\n" + "=" * 70)
    print("TEST 2: Piecemeal Downsample (128→64 via omega grid)")
    print("=" * 70)

    rng = np.random.RandomState(99)
    results = {}

    for t in range(16):
        mse_native, mse_piecemeal, mse_cross = [], [], []

        for _ in range(n_per_type):
            # Generate same-seed noise at both resolutions
            seed = rng.randint(0, 2**31)

            # Native 64×64
            torch.manual_seed(seed)
            img_64 = gen_noise(t, 64, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out_64 = model(img_64)
            mse_n = F.mse_loss(out_64['recon'], img_64).item()
            mse_native.append(mse_n)

            # Encode at 128×128
            torch.manual_seed(seed)
            img_128 = gen_noise(t, 128, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out_128 = model(img_128)
            S_128 = out_128['svd']['S']  # (1, 1024, 4)

            # Downsample omega grid: 32×32 → 16×16
            S_down = downsample_omega(S_128, 32, 32, 16, 16)

            # Decode at 64×64 using downsampled omegas
            # We need U and Vt at 64×64 scale — use native encoding for basis
            decoded = model.decode_patches(
                out_64['svd']['U'], S_down, out_64['svd']['Vt'])
            recon_pm = stitch_patches(decoded, 16, 16, ps)
            recon_pm = model.boundary_smooth(recon_pm)
            mse_pm = F.mse_loss(recon_pm, img_64).item()
            mse_piecemeal.append(mse_pm)

            # Omega similarity: how close are downsampled 128 omegas to native 64 omegas?
            S_native = out_64['svd']['S']
            omega_sim = F.mse_loss(S_down, S_native).item()
            mse_cross.append(omega_sim)

        results[NOISE_NAMES[t]] = {
            'native_mse': np.mean(mse_native),
            'piecemeal_mse': np.mean(mse_piecemeal),
            'omega_distance': np.mean(mse_cross),
        }

    print(f"\n  {'type':<14s} {'native':>10s} {'piecemeal':>10s} {'ω distance':>10s} {'ratio':>8s}")
    print(f"  {'-'*56}")
    for name, r in sorted(results.items()):
        ratio = r['piecemeal_mse'] / (r['native_mse'] + 1e-10)
        print(f"  {name:<14s} {r['native_mse']:10.6f} {r['piecemeal_mse']:10.6f} "
              f"{r['omega_distance']:10.6f} {ratio:8.2f}x")

    return results


@torch.no_grad()
def test_piecemeal_upsample(model, device, ps=4, n_per_type=4):
    """Test 3: Encode small → upsample omega grid → decode large.

    Encode 64×64 (256 patches) → upsample omega to 1024 patches → decode 128×128.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Piecemeal Upsample (64→128 via omega grid)")
    print("=" * 70)

    rng = np.random.RandomState(123)
    results = {}

    for t in range(16):
        mse_native_128, mse_upsampled = [], []

        for _ in range(n_per_type):
            seed = rng.randint(0, 2**31)

            # Native 128×128
            torch.manual_seed(seed)
            img_128 = gen_noise(t, 128, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out_128 = model(img_128)
            mse_n = F.mse_loss(out_128['recon'], img_128).item()
            mse_native_128.append(mse_n)

            # Encode at 64×64
            torch.manual_seed(seed)
            img_64 = gen_noise(t, 64, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out_64 = model(img_64)
            S_64 = out_64['svd']['S']  # (1, 256, 4)

            # Upsample omega grid: 16×16 → 32×32
            S_up = upsample_omega(S_64, 16, 16, 32, 32)

            # Decode at 128×128 using 128-scale basis
            decoded = model.decode_patches(
                out_128['svd']['U'], S_up, out_128['svd']['Vt'])
            recon_up = stitch_patches(decoded, 32, 32, ps)
            recon_up = model.boundary_smooth(recon_up)
            mse_u = F.mse_loss(recon_up, img_128).item()
            mse_upsampled.append(mse_u)

        results[NOISE_NAMES[t]] = {
            'native_128_mse': np.mean(mse_native_128),
            'upsampled_mse': np.mean(mse_upsampled),
        }

    print(f"\n  {'type':<14s} {'native 128':>12s} {'upsampled':>12s} {'ratio':>8s}")
    print(f"  {'-'*50}")
    for name, r in sorted(results.items()):
        ratio = r['upsampled_mse'] / (r['native_128_mse'] + 1e-10)
        print(f"  {name:<14s} {r['native_128_mse']:12.6f} {r['upsampled_mse']:12.6f} {ratio:8.2f}x")

    return results


@torch.no_grad()
def test_tile_stitch(model, device, ps=4, n_per_type=4):
    """Test 4: Tile-encode large image → stitch omega grids → decode.

    Generate 128×128 noise. Encode as 4 tiles of 64×64.
    Stitch the omega grids. Compare with native 128×128 encoding.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Tile Encode + Stitch (4×64×64 tiles → 128×128)")
    print("=" * 70)

    rng = np.random.RandomState(456)
    results = {}

    for t in range(16):
        mse_native, mse_tiled, omega_dist = [], [], []

        for _ in range(n_per_type):
            img = gen_noise(t, 128, rng).clamp(-4, 4)

            # Native 128×128 encoding
            img_batch = img.unsqueeze(0).to(device)
            out_native = model(img_batch)
            mse_n = F.mse_loss(out_native['recon'], img_batch).item()
            mse_native.append(mse_n)

            # Tile encoding: 4 tiles of 64×64
            S_tiled, gh, gw = tile_encode(model, img, tile_size=64, ps=ps, device=device)

            # Compare omega grids
            S_native = out_native['svd']['S']
            od = F.mse_loss(S_tiled, S_native).item()
            omega_dist.append(od)

            # Decode tiled omegas using native basis
            decoded = model.decode_patches(
                out_native['svd']['U'], S_tiled, out_native['svd']['Vt'])
            recon_tiled = stitch_patches(decoded, gh, gw, ps)
            recon_tiled = model.boundary_smooth(recon_tiled)
            mse_t = F.mse_loss(recon_tiled, img_batch).item()
            mse_tiled.append(mse_t)

        results[NOISE_NAMES[t]] = {
            'native_mse': np.mean(mse_native),
            'tiled_mse': np.mean(mse_tiled),
            'omega_distance': np.mean(omega_dist),
        }

    print(f"\n  {'type':<14s} {'native':>10s} {'tiled':>10s} {'ω distance':>10s} {'match':>8s}")
    print(f"  {'-'*56}")
    for name, r in sorted(results.items()):
        ratio = r['tiled_mse'] / (r['native_mse'] + 1e-10)
        print(f"  {name:<14s} {r['native_mse']:10.6f} {r['tiled_mse']:10.6f} "
              f"{r['omega_distance']:10.6f} {ratio:8.2f}x")

    return results


@torch.no_grad()
def test_omega_consistency(model, device, ps=4, n_samples=32):
    """Test 5: Do identical 4×4 patches produce identical omega tokens?

    Extract the same patch from different spatial locations in the same image.
    Compare their omega tokens. Perfect consistency = 0 distance.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Omega Token Consistency (same patch, different location)")
    print("=" * 70)

    rng = np.random.RandomState(789)
    results = {}

    for t in range(16):
        distances = []

        for _ in range(n_samples):
            # Generate 128×128 noise
            img = gen_noise(t, 128, rng).clamp(-4, 4).unsqueeze(0).to(device)
            out = model(img)
            S = out['svd']['S']  # (1, 1024, 4)

            # Reshape to spatial grid
            S_2d = omega_grid_to_2d(S, 32, 32)  # (1, 4, 32, 32)

            # Compare random patch pairs
            for _ in range(10):
                r1, c1 = rng.randint(0, 32), rng.randint(0, 32)
                r2, c2 = rng.randint(0, 32), rng.randint(0, 32)
                if r1 == r2 and c1 == c2:
                    continue
                s1 = S_2d[0, :, r1, c1]
                s2 = S_2d[0, :, r2, c2]
                # For uniform noise, all patches should have similar omega tokens
                # For structured noise, they'll differ by location
                distances.append((s1 - s2).abs().mean().item())

        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        results[NOISE_NAMES[t]] = {'mean_dist': avg_dist, 'std_dist': std_dist}

    print(f"\n  {'type':<14s} {'mean ω dist':>12s} {'std':>10s} {'structure':>10s}")
    print(f"  {'-'*50}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean_dist']):
        structure = 'uniform' if r['mean_dist'] < 0.1 else 'spatial' if r['mean_dist'] < 0.5 else 'strong'
        print(f"  {name:<14s} {r['mean_dist']:12.6f} {r['std_dist']:10.6f} {structure:>10s}")

    return results


@torch.no_grad()
def test_geometric_scaling(model, device, ps=4, n_per_type=8):
    """Test 6: How do erank and S profile change with resolution?

    If the geometry is resolution-invariant, erank and S0/SD ratio
    should be constant across resolutions.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Geometric Constants vs Resolution")
    print("=" * 70)

    resolutions = [32, 48, 64, 80, 96, 128]
    rng = np.random.RandomState(321)
    D = 4

    results = {}
    for res in resolutions:
        if res % ps != 0:
            continue
        all_S = []
        for t in range(16):
            for _ in range(n_per_type):
                img = gen_noise(t, res, rng).clamp(-4, 4).unsqueeze(0).to(device)
                out = model(img)
                all_S.append(out['svd']['S'])

        S_cat = torch.cat(all_S, dim=0)  # (N, patches, D)
        S_flat = S_cat.reshape(-1, D)
        S_mean = S_flat.mean(0)

        p = S_flat / (S_flat.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        erank = (-(p * p.log()).sum(-1)).exp().mean().item()

        results[res] = {
            'erank': erank,
            'S0': S_mean[0].item(),
            'SD': S_mean[-1].item(),
            'ratio': (S_mean[0] / (S_mean[-1] + 1e-8)).item(),
            'n_patches': (res // ps) ** 2,
        }

    print(f"\n  {'res':>5s} {'patches':>8s} {'erank':>7s} {'S0':>7s} {'SD':>7s} {'S0/SD':>7s}")
    print(f"  {'-'*44}")
    for res, r in sorted(results.items()):
        print(f"  {res:>5d} {r['n_patches']:>8d} {r['erank']:7.3f} "
              f"{r['S0']:7.3f} {r['SD']:7.3f} {r['ratio']:7.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════

def load_freckles(model_path=None, hf_version=None, device='cuda'):
    """Load Freckles model from checkpoint or HuggingFace."""

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

    # Import Freckles — try from the module first, fall back to inline
    try:
        from freckles import Freckles
    except ImportError:
        # Inline minimal Freckles for standalone use
        print("  (Using inline Freckles definition)")
        from geolip_svae.model import (
            BoundarySmooth as BS,
            SpectralCrossAttention as SCA,
        )

        class Freckles(nn.Module):
            def __init__(self, V=48, D=4, ps=4, hidden=384, depth=4, n_cross=2):
                super().__init__()
                self.V, self.D, self.patch_size = V, D, ps
                self.patch_dim = 3 * ps * ps
                self.mat_dim = V * D
                self.enc_in = nn.Linear(self.patch_dim, hidden)
                self.enc_blocks = nn.ModuleList([
                    nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, hidden))
                    for _ in range(depth)])
                self.enc_out = nn.Linear(hidden, self.mat_dim)
                nn.init.orthogonal_(self.enc_out.weight)
                self.dec_in = nn.Linear(self.mat_dim, hidden)
                self.dec_blocks = nn.ModuleList([
                    nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, hidden))
                    for _ in range(depth)])
                self.dec_out = nn.Linear(hidden, self.patch_dim)
                self.cross_attn = nn.ModuleList([
                    SCA(D, n_heads=min(2, D)) for _ in range(n_cross)])
                self.boundary_smooth = BS(channels=3, mid=8)

            def encode_patches(self, patches):
                B, N, _ = patches.shape
                from geolip_svae.model import gram_eigh_svd
                h = F.gelu(self.enc_in(patches.reshape(B * N, -1)))
                for block in self.enc_blocks: h = h + block(h)
                M = F.normalize(self.enc_out(h).reshape(B * N, self.V, self.D), dim=-1)
                U, S, Vt = gram_eigh_svd(M)
                U = U.reshape(B, N, self.V, self.D)
                S = S.reshape(B, N, self.D)
                Vt = Vt.reshape(B, N, self.D, self.D)
                M = M.reshape(B, N, self.V, self.D)
                S_c = S
                for layer in self.cross_attn: S_c = layer(S_c)
                return {'U': U, 'S_orig': S, 'S': S_c, 'Vt': Vt, 'M': M}

            def decode_patches(self, U, S, Vt):
                B, N, V, D = U.shape
                M_hat = torch.bmm(
                    U.reshape(B * N, V, D) * S.reshape(B * N, D).unsqueeze(1),
                    Vt.reshape(B * N, D, D))
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

    model = Freckles(
        V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
        hidden=cfg['hidden'], depth=cfg['depth'],
        n_cross=cfg['n_cross_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded Freckles: {n_params:,} params, V={cfg['V']}, D={cfg['D']}, ps={cfg['patch_size']}")
    print(f"  Trained at: {cfg.get('img_size', '?')}×{cfg.get('img_size', '?')}")
    print(f"  Val MSE: {ckpt.get('val_mse', '?')}")

    return model, cfg


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_full_diagnostic(model, device, ps=4, save_path=None):
    """Run all 6 tests and optionally save results."""

    print("\n" + "=" * 70)
    print("FRECKLES PIECEMEAL NOISE DIAGNOSTIC")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    # Test 1: Native reconstruction at multiple resolutions
    all_results['native_recon'] = test_native_recon(
        model, device, resolutions=[32, 48, 64, 96, 128], ps=ps)

    # Test 2: Downsample piecemeal (128→64)
    all_results['piecemeal_downsample'] = test_piecemeal_downsample(
        model, device, ps=ps)

    # Test 3: Upsample piecemeal (64→128)
    all_results['piecemeal_upsample'] = test_piecemeal_upsample(
        model, device, ps=ps)

    # Test 4: Tile encode + stitch
    all_results['tile_stitch'] = test_tile_stitch(model, device, ps=ps)

    # Test 5: Omega consistency
    all_results['omega_consistency'] = test_omega_consistency(
        model, device, ps=ps)

    # Test 6: Geometric scaling
    all_results['geometric_scaling'] = test_geometric_scaling(
        model, device, ps=ps)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DIAGNOSTIC COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")

    if save_path:
        # Convert tensors for JSON
        def to_json(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: to_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json(v) for v in obj]
            return obj

        with open(save_path, 'w') as f:
            json.dump(to_json(all_results), f, indent=2)
        print(f"  Results saved: {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Freckles Piecemeal Noise Diagnostic')
    parser.add_argument('--model', type=str, default=None,
                        help='HuggingFace version (e.g., v40_freckles_noise)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Local checkpoint path')
    parser.add_argument('--output', type=str, default='freckles_diagnostic.json',
                        help='Output JSON path')
    args = parser.parse_args()

    if not args.model and not args.checkpoint:
        args.model = 'v40_freckles_noise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_freckles(
        model_path=args.checkpoint,
        hf_version=args.model,
        device=device)

    run_full_diagnostic(model, device, ps=cfg['patch_size'], save_path=args.output)
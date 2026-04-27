"""
geolip_svae.inference.calibration
==================================
Calibration data generators for codebook extraction and inference probes.

A calibration set is a small batch of inputs used to evoke representative
M tensors from a trained sphere-solver. Different models were trained
on different distributions; calibrating with a matched distribution
gives the model its best foot forward, while calibrating with a
mismatched distribution exposes how the codebook responds to
out-of-distribution inputs (a Phase U-style probe).

All generators return ``(N, 3, H, W)`` float32 tensors clamped to
``[-4, 4]``. Resolution and N are caller-specified.

Built-in distributions
----------------------
gaussian        — N(0, 1) per pixel. The canonical h2-64 calibration.
uniform         — U(-1, 1) per pixel.
sixteen_noise   — Johanna/Freckles training mix: 16 noise types in
                  equal proportion, drawn from the canonical h2-64
                  noise generator family.

Registry pattern
----------------
``CALIBRATION_REGISTRY`` maps name → callable. Users can register
new generators via ``register_calibration(name, fn)`` to add custom
distributions without modifying this file.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
# Generator type
# ════════════════════════════════════════════════════════════════════
# Calibration generator signature:
#     fn(n: int, size: int, seed: int) -> Tensor of shape (n, 3, size, size)

CalibrationFn = Callable[[int, int, int], torch.Tensor]


# ════════════════════════════════════════════════════════════════════
# Single-distribution generators
# ════════════════════════════════════════════════════════════════════

def gen_gaussian(n: int, size: int, seed: int = 0) -> torch.Tensor:
    """Pure N(0, 1) gaussian noise. Shape (n, 3, size, size)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, 3, size, size, generator=g).clamp(-4, 4)


def gen_uniform(n: int, size: int, seed: int = 0) -> torch.Tensor:
    """Uniform U(-1, 1) noise."""
    g = torch.Generator().manual_seed(int(seed))
    return (torch.rand(n, 3, size, size, generator=g) * 2 - 1).clamp(-4, 4)


# ════════════════════════════════════════════════════════════════════
# 16-noise mix (Johanna / Freckles / h2-64 training distribution)
# ════════════════════════════════════════════════════════════════════
# Reproduces the noise family used to train the canonical h2-64 array
# and the Johanna / Freckles HF checkpoints. Indices 0..15 map to:
#   0 gaussian          1 uniform           2 uniform_scaled    3 poisson
#   4 pink              5 brown             6 salt_pepper       7 sparse_impulses
#   8 block_upsampled   9 gradient_gaussian 10 checker          11 gauss_uniform_mix
#   12 four_quadrant    13 cauchy           14 exponential      15 laplace

NOISE_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_scaled', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse_impulses',
    8: 'block_upsampled', 9: 'gradient_gaussian', 10: 'checker',
    11: 'gauss_uniform_mix', 12: 'four_quadrant',
    13: 'cauchy', 14: 'exponential', 15: 'laplace',
}


def _pink_noise(shape, rng_torch):
    w = torch.randn(shape, generator=rng_torch)
    s = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(s / torch.sqrt(fx**2 + fy**2).clamp(min=1e-8),
                              s=(h, ww))


def _brown_noise(shape, rng_torch):
    w = torch.randn(shape, generator=rng_torch)
    s = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(s / (fx**2 + fy**2).clamp(min=1e-8), s=(h, ww))


def _gen_one_noise(noise_type: int, size: int, seed: int) -> torch.Tensor:
    """Generate a single (3, size, size) sample of the given noise type.

    Mirrors the canonical noise generator from h2-64 training. ``size``
    must be even for some types (block_upsampled, four_quadrant); the
    caller is responsible for supplying an even size.
    """
    rng_t = torch.Generator().manual_seed(int(seed))
    rng_n = np.random.RandomState(int(seed))
    s = size

    if noise_type == 0:
        img = torch.randn(3, s, s, generator=rng_t)
    elif noise_type == 1:
        img = torch.rand(3, s, s, generator=rng_t) * 2 - 1
    elif noise_type == 2:
        img = (torch.rand(3, s, s, generator=rng_t) - 0.5) * 4
    elif noise_type == 3:
        lam = rng_n.uniform(0.5, 20.0)
        img = (torch.poisson(torch.full((3, s, s), lam), generator=rng_t)
                / lam - 1.0)
    elif noise_type == 4:
        img = _pink_noise((3, s, s), rng_t)
        img = img / (img.std() + 1e-8)
    elif noise_type == 5:
        img = _brown_noise((3, s, s), rng_t)
        img = img / (img.std() + 1e-8)
    elif noise_type == 6:
        mask = torch.rand(3, s, s, generator=rng_t) > 0.5
        img = torch.where(mask, torch.ones(3, s, s) * 2,
                                  torch.ones(3, s, s) * -2)
        img = img + torch.randn(3, s, s, generator=rng_t) * 0.1
    elif noise_type == 7:
        mask = torch.rand(3, s, s, generator=rng_t) > 0.9
        img = torch.randn(3, s, s, generator=rng_t) * mask.float() * 3
    elif noise_type == 8:
        block = int(rng_n.randint(2, 16))
        small = torch.randn(3, s // block + 1, s // block + 1, generator=rng_t)
        img = F.interpolate(small.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif noise_type == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        angle = float(rng_n.uniform(0, 2 * math.pi))
        grad = math.cos(angle) * gx + math.sin(angle) * gy
        img = (grad.unsqueeze(0).expand(3, -1, -1)
                + torch.randn(3, s, s, generator=rng_t) * 0.5)
    elif noise_type == 10:
        cs = int(rng_n.randint(2, 16))
        cy = torch.arange(s) // cs
        cx = torch.arange(s) // cs
        checker = ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float() * 2 - 1
        img = (checker.unsqueeze(0).expand(3, -1, -1)
                + torch.randn(3, s, s, generator=rng_t) * 0.3)
    elif noise_type == 11:
        a = torch.randn(3, s, s, generator=rng_t)
        b = torch.rand(3, s, s, generator=rng_t) * 2 - 1
        alpha = float(rng_n.uniform(0.2, 0.8))
        img = alpha * a + (1 - alpha) * b
    elif noise_type == 12:
        if s % 2:
            raise ValueError(
                f"four_quadrant requires even size, got {s}"
            )
        img = torch.zeros(3, s, s)
        h2 = s // 2
        img[:, :h2, :h2] = torch.randn(3, h2, h2, generator=rng_t)
        img[:, :h2, h2:] = torch.rand(3, h2, h2, generator=rng_t) * 2 - 1
        img[:, h2:, :h2] = _pink_noise((3, h2, h2), rng_t) / 2
        sp = torch.where(
            torch.rand(3, h2, h2, generator=rng_t) > 0.5,
            torch.ones(3, h2, h2),
            -torch.ones(3, h2, h2),
        )
        img[:, h2:, h2:] = sp
    elif noise_type == 13:
        u = torch.rand(3, s, s, generator=rng_t)
        img = torch.tan(math.pi * (u - 0.5)).clamp(-3, 3)
    elif noise_type == 14:
        img = torch.empty(3, s, s).exponential_(1.0, generator=rng_t) - 1.0
    elif noise_type == 15:
        u = torch.rand(3, s, s, generator=rng_t) - 0.5
        img = -torch.sign(u) * torch.log1p(-2 * u.abs())
    else:
        raise ValueError(f"Unknown noise_type {noise_type}")
    return img.clamp(-4, 4).float()


def gen_sixteen_noise(
    n: int,
    size: int,
    seed: int = 0,
    samples_per_type: Optional[int] = None,
) -> torch.Tensor:
    """16-noise mix: equal samples per noise type.

    By default produces ``ceil(n / 16)`` samples of each type, then
    truncates to ``n`` total. Useful as a calibration distribution
    matching Johanna/Freckles training data.

    Args:
        n: total number of calibration images requested
        size: spatial dimension (must be even — required by some types)
        seed: master seed; per-type seeds derive from ``seed * 1000 + type * 100 + i``
        samples_per_type: override the per-type count. If set, returns
            ``samples_per_type * 16`` images; ``n`` is ignored.

    Returns:
        (n_actual, 3, size, size) tensor where n_actual = n or
        samples_per_type * 16.
    """
    if size % 2:
        raise ValueError(
            f"gen_sixteen_noise requires even size (four_quadrant uses s/2); "
            f"got {size}"
        )

    if samples_per_type is None:
        samples_per_type = math.ceil(n / 16)

    images = []
    for t in range(16):
        for i in range(samples_per_type):
            sub_seed = seed * 1000 + t * 100 + i
            images.append(_gen_one_noise(t, size, sub_seed))
    stack = torch.stack(images, dim=0)

    if samples_per_type is not None and (samples_per_type * 16 != n):
        # Caller specified samples_per_type explicitly — return all
        return stack
    return stack[:n]


# ════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════

CALIBRATION_REGISTRY: Dict[str, CalibrationFn] = {
    'gaussian': gen_gaussian,
    'uniform': gen_uniform,
    'sixteen_noise': gen_sixteen_noise,
    '16noise': gen_sixteen_noise,  # alias
}


def register_calibration(name: str, fn: CalibrationFn) -> None:
    """Register a custom calibration generator under ``name``."""
    if name in CALIBRATION_REGISTRY:
        raise KeyError(
            f"Calibration '{name}' is already registered. "
            f"Choose a different name or unregister first."
        )
    CALIBRATION_REGISTRY[name] = fn


def get_calibration(name: str) -> CalibrationFn:
    """Look up a calibration generator by name."""
    if name not in CALIBRATION_REGISTRY:
        raise KeyError(
            f"Unknown calibration '{name}'. "
            f"Registered: {sorted(CALIBRATION_REGISTRY)}"
        )
    return CALIBRATION_REGISTRY[name]


def make_calibration(
    name: str,
    n: int,
    size: int,
    seed: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Build a calibration tensor by registry name.

    Convenience wrapper over the registry. Forwards ``**kwargs`` to
    the underlying generator (e.g. ``samples_per_type=`` for sixteen_noise).
    """
    fn = get_calibration(name)
    return fn(n, size, seed, **kwargs) if kwargs else fn(n, size, seed)


__all__ = [
    'CalibrationFn',
    'NOISE_NAMES',
    'gen_gaussian',
    'gen_uniform',
    'gen_sixteen_noise',
    'CALIBRATION_REGISTRY',
    'register_calibration',
    'get_calibration',
    'make_calibration',
]
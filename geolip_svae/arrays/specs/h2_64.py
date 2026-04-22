"""
geolip_svae.arrays.specs.h2_64
================================
Specification for the h2-64 battery array.

64 sphere-solver batteries, each an H2_linear_matched PatchSVAE at HIGH
band (D=4, V=32, ps=4, hidden=64, 57,215 params each). Bundled with 3
epoch phases (epoch_1, best, final) → 192 banks total.

Training composition:
    Group 1 — 16 single-noise experts (one per noise type 0..15)
    Group 2 — 15 gaussian+one pairs + 1 all-16 generalist
    Group 3 — 16 (gaussian, easy, medium, hard) balanced covering
    Group 4 — 16 (easy, medium, hard, hard) no-gaussian covering

Source repo: AbstractPhil/geolip-svae-h2-64
"""

from itertools import combinations, product
from typing import Any, Dict, List, Tuple


NAME = "h2_64"

# ── Architecture: H2_linear_matched at HIGH band ─────────────────────

BATTERY_CLASS = "PatchSVAE"
BATTERY_MODULE = "geolip_svae.model"

BATTERY_KWARGS = {
    # Base dims
    "V": 32,
    "D": 4,
    "ps": 4,
    "hidden": 64,
    "depth": 1,
    "n_cross": 1,
    "n_heads": 4,
    # smooth_mid must be explicit — default in PatchSVAE is ps-dependent
    # but the h2-64 training used unconditional mid=16, matching what's
    # in johanna_F_trainer.py
    "smooth_mid": 16,
    # H-group ablation: linear readout replaces SVD
    "svd_mode": "none",
    "linear_readout": True,
    "match_params": True,
}

N_BATTERIES = 64
EPOCH_PHASE_NAMES = ["epoch_1", "best", "final"]

SOURCE_REPO = "AbstractPhil/geolip-svae-h2-64"


# ── Noise taxonomy (replicated from h2_64_configs.py in training repo) ──

NOISE_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_scaled', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse_impulses',
    8: 'block_upsampled', 9: 'gradient_gaussian', 10: 'checker',
    11: 'gauss_uniform_mix', 12: 'four_quadrant',
    13: 'cauchy', 14: 'exponential', 15: 'laplace',
}

GAUSSIAN = 0
EASY_NOISES = [1, 2, 13, 14, 15]
MEDIUM_NOISES = [3, 6, 7, 11]
HARD_NOISES = [4, 5, 8, 9, 10, 12]


# ── Balanced covering designs (deterministic, reproducible) ──────────

def _build_gaussian_quads() -> List[Tuple[int, int, int, int]]:
    """16 (gaussian, easy, medium, hard) via stride-7 covering."""
    all_triples = list(product(EASY_NOISES, MEDIUM_NOISES, HARD_NOISES))
    indices = [(k * 7) % len(all_triples) for k in range(16)]
    seen, picked = set(), []
    for idx in indices:
        while idx in seen:
            idx = (idx + 1) % len(all_triples)
        seen.add(idx)
        picked.append(all_triples[idx])
    return [(GAUSSIAN, e, m, h) for (e, m, h) in picked]


def _build_no_gaussian_quads() -> List[Tuple[int, int, int, int]]:
    """16 (easy, medium, hard, hard) via stride-19 covering. No gaussian."""
    hard_pairs = list(combinations(HARD_NOISES, 2))
    all_quads = list(product(EASY_NOISES, MEDIUM_NOISES, hard_pairs))
    indices = [(k * 19) % len(all_quads) for k in range(16)]
    seen, picked = set(), []
    for idx in indices:
        while idx in seen:
            idx = (idx + 1) % len(all_quads)
        seen.add(idx)
        picked.append(all_quads[idx])
    return [(e, m, h_pair[0], h_pair[1]) for (e, m, h_pair) in picked]


# ── Full 64-config assembly ──────────────────────────────────────────

def get_configs() -> List[Dict[str, Any]]:
    """Return the 64 battery configs in deterministic order.

    Each config is a dict with keys:
        battery_id: int (0..63)
        subgroup:   str
        variant:    str
        noise_types: List[int]
    The hf_path (subgroup/variant) is also where final_report.json and
    checkpoints live in the source HF repo.
    """
    configs = []

    # Group 1: 16 single-noise experts
    for n in range(16):
        configs.append({
            'battery_id': len(configs),
            'subgroup': 'single',
            'variant': f'noise_{n:02d}_{NOISE_NAMES[n]}',
            'noise_types': [n],
            'noise_names': [NOISE_NAMES[n]],
        })

    # Group 2: 15 gaussian-pair + 1 all-16
    for n in range(1, 16):
        configs.append({
            'battery_id': len(configs),
            'subgroup': 'gaussian_plus_one',
            'variant': f'pair_{n:02d}_{NOISE_NAMES[n]}',
            'noise_types': [GAUSSIAN, n],
            'noise_names': [NOISE_NAMES[GAUSSIAN], NOISE_NAMES[n]],
        })
    configs.append({
        'battery_id': len(configs),
        'subgroup': 'gaussian_plus_one',
        'variant': 'all_16',
        'noise_types': list(range(16)),
        'noise_names': [NOISE_NAMES[i] for i in range(16)],
    })

    # Group 3: 16 gaussian-quads
    for i, (g, e, m, h) in enumerate(_build_gaussian_quads(), 1):
        configs.append({
            'battery_id': len(configs),
            'subgroup': 'gaussian_quad',
            'variant': f'quad_{i:02d}_E{e:02d}_M{m:02d}_H{h:02d}',
            'noise_types': [g, e, m, h],
            'noise_names': [NOISE_NAMES[g], NOISE_NAMES[e],
                             NOISE_NAMES[m], NOISE_NAMES[h]],
        })

    # Group 4: 16 no-gaussian quads
    for i, (e, m, h1, h2) in enumerate(_build_no_gaussian_quads(), 1):
        configs.append({
            'battery_id': len(configs),
            'subgroup': 'no_gaussian_quad',
            'variant': f'ngq_{i:02d}_E{e:02d}_M{m:02d}_H{h1:02d}_H{h2:02d}',
            'noise_types': [e, m, h1, h2],
            'noise_names': [NOISE_NAMES[e], NOISE_NAMES[m],
                             NOISE_NAMES[h1], NOISE_NAMES[h2]],
        })

    assert len(configs) == N_BATTERIES
    return configs


def checkpoint_path(config: Dict[str, Any], epoch: int) -> str:
    """Derive the HF path to a specific checkpoint."""
    return f"{config['subgroup']}/{config['variant']}/epoch_{epoch}_checkpoint.pt"


def report_path(config: Dict[str, Any]) -> str:
    """Derive the HF path to final_report.json."""
    return f"{config['subgroup']}/{config['variant']}/final_report.json"
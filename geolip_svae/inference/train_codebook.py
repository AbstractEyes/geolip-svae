"""
geolip_svae.inference.train_codebook
=====================================
Codebook creation pipeline + post-extraction topological analysis.

This module is the *orchestrator* layer over ``codebook.extract_codebook``.
It auto-resolves a sensible calibration based on the model's training
config, runs the extraction, saves the artifact, optionally uploads to
HuggingFace, and (optionally) runs the three topology probes from the
research:

  Probe A — kNN-graph connectivity sweep over angular thresholds θ
  Probe B — Local intrinsic dimension via PCA on k-neighbor offsets
  Probe C — Persistent homology (ripser, H0/H1/H2)

These match the math in ``tests/experiment_codebook_topological_analysis.py``
generalized to operate on any axis cloud (single codebook OR pooled
multi-bank). ripser is optional; the kNN and PCA probes always run.

Trainer integration (see ``geolip_svae.train``): ``create_codebook`` is
called at end-of-training when ``cfg.get('build_codebook', True)``,
producing an artifact alongside the final checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from geolip_svae.inference.calibration import make_calibration
from geolip_svae.inference.codebook import (
    Codebook,
    extract_codebook,
)
from geolip_svae.inference.engine import InferenceEngine
from geolip_svae.inference.loading import HF_REPO


# ── Optional ripser import (persistent homology probe) ──────────────

try:
    from ripser import ripser as _ripser
    HAVE_RIPSER = True
except ImportError:
    HAVE_RIPSER = False
    _ripser = None


# ── Default calibration registry ────────────────────────────────────

# Keyed by inferred model class. Each entry is what to feed
# ``make_calibration`` for the canonical extraction.
#
# Sizes match the empirical defaults used in the U5 codebook capacity
# tests and the byte_trigram_proto sessions (000115, 000118).

DEFAULT_CALIBRATIONS: Dict[str, Dict[str, Any]] = {
    'h2-class':      {'name': 'sixteen_noise', 'n': 64, 'size': 64},
    'a-class':       {'name': 'sixteen_noise', 'n': 64, 'size': 256},
    's-class':       {'name': 'sixteen_noise', 'n': 64, 'size': 64},
    'p-class':       {'name': 'gaussian',      'n': 64, 'size': 64},
    'byte_trigram':  {'name': 'sixteen_noise', 'n': 64, 'size': 64},
    'sentencepiece': {'name': 'sixteen_noise', 'n': 64, 'size': 16},
    'binary_tree':   {'name': 'gaussian',      'n': 64, 'size': 16},
    'image':         {'name': 'gaussian',      'n': 64, 'size': 64},
    'text':          {'name': 'sixteen_noise', 'n': 64, 'size': 128},
    'unknown':       {'name': 'sixteen_noise', 'n': 64, 'size': 64},
}


def infer_class_from_cfg(cfg: Dict[str, Any]) -> str:
    """Heuristic: look at architecture flags + dataset to pick a class.

    Returns one of the keys in ``DEFAULT_CALIBRATIONS``.
    """
    V        = cfg.get('V')
    D        = cfg.get('D')
    dataset  = cfg.get('dataset', '')
    linear   = cfg.get('linear_readout', False)
    svd_mode = cfg.get('svd_mode', 'default')

    # Dataset-driven cases first (highest specificity)
    if dataset == 'byte_trigram':
        return 'byte_trigram'
    if dataset == 'sentencepiece_bits':
        return 'sentencepiece'
    if dataset == 'binary_tree':
        return 'binary_tree'
    if dataset in ('tiny_imagenet', 'imagenet_128', 'imagenet_256'):
        return 'image'
    if dataset == 'wikipedia':
        return 'text'

    # Architecture-driven cases (sphere-solver lineage)
    if linear and svd_mode == 'none':
        if V == 32 and D == 4:
            return 'h2-class'
        if V == 32 and D == 3:
            return 'p-class'
    if V == 256 and D == 16:
        return 'a-class'
    if V == 48 and D == 4:
        return 's-class'

    return 'unknown'


# ── Topology data classes ───────────────────────────────────────────

@dataclass
class TopologyReport:
    """Persistent-homology + connectivity + intrinsic-dim summary
    for a codebook (or any unit-vector axis cloud).
    """
    n_axes: int
    D: int

    # Pairwise angular distance summary stats (degrees)
    angular_dist_p25_deg: float
    angular_dist_p50_deg: float
    angular_dist_p75_deg: float
    angular_dist_p95_deg: float

    # Probe A: kNN-graph
    knn_components_at_thresh:  Dict[float, int]            # θ_deg → component count
    knn_largest_pct_at_thresh: Dict[float, float]          # θ_deg → largest comp %
    percolation_thresh_deg: Optional[float]                # θ where giant ≥ 50%

    # Probe B: local intrinsic dimension (k-neighbor PCA)
    local_dim_pr_p25:  float
    local_dim_pr_p50:  float
    local_dim_pr_p75:  float
    local_dim_count_mode: int

    # Probe C: persistent homology (ripser)
    persistence_diagrams:    Dict[str, List[List[float]]]  # 'H0','H1','H2' → [[birth, death], ...]
    persistence_n_finite:    Dict[str, int]
    persistence_n_infinite:  Dict[str, int]
    top_persistent_features: Dict[str, List[List[float]]]  # H1/H2 → [[birth_deg, death_deg, persist_deg], ...]
    ripser_thresh_deg: float
    ripser_compute_seconds: float
    ripser_available: bool

    notes: str = ''
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> Path:
        """Save as JSON. ``path`` is the stem; the file gets ``.json``."""
        path = Path(path)
        if path.suffix == '.json':
            stem = path.with_suffix('')
        else:
            stem = path
        stem.parent.mkdir(parents=True, exist_ok=True)
        json_path = stem.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return stem

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TopologyReport':
        path = Path(path)
        if path.suffix != '.json':
            path = path.with_suffix('.json')
        with open(path) as f:
            d = json.load(f)
        # JSON serialization stringifies dict keys; coerce θ keys back to float
        for k in ('knn_components_at_thresh', 'knn_largest_pct_at_thresh'):
            d[k] = {float(t): v for t, v in d[k].items()}
        return cls(**d)


@dataclass
class ArrayTopologyReport:
    """Multi-bank topology atlas — pooled axis cloud + per-bank attribution.

    Wraps a single ``TopologyReport`` over the pooled cloud plus extra
    fields that only make sense across multiple banks (per-axis bank
    provenance, rare/common classification, per-bank rare-axis counts).
    """
    pooled: TopologyReport
    n_banks: int
    bank_labels: List[str]
    n_axes_per_bank: List[int]

    # Probe: per-axis cross-bank kNN density
    cross_bank_d_knn1_deg: List[float]                     # length = n_axes
    cross_bank_d_knn1_p10_p50_p85_deg: Tuple[float, float, float]
    rare_axis_threshold_deg: float
    n_common_axes: int                                     # d_kNN1 ≤ p15
    n_typical_axes: int
    n_rare_axes: int                                       # d_kNN1 ≥ p85

    # Per-bank diversity ranking
    rare_axes_per_bank: Dict[str, int]                     # bank_label → count

    notes: str = ''
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.suffix == '.json':
            stem = path.with_suffix('')
        else:
            stem = path
        stem.parent.mkdir(parents=True, exist_ok=True)
        with open(stem.with_suffix('.json'), 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return stem

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ArrayTopologyReport':
        path = Path(path)
        if path.suffix != '.json':
            path = path.with_suffix('.json')
        with open(path) as f:
            d = json.load(f)
        d['pooled'] = TopologyReport(**d['pooled'])
        d['cross_bank_d_knn1_p10_p50_p85_deg'] = tuple(
            d['cross_bank_d_knn1_p10_p50_p85_deg']
        )
        return cls(**d)


# ── Probe primitives (numpy, no torch) ──────────────────────────────

def _normalize_axes(axes: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return unit-vector axes as float64 ``[n, D]`` numpy."""
    if isinstance(axes, torch.Tensor):
        axes = axes.detach().cpu().numpy()
    axes = np.asarray(axes, dtype=np.float64)
    norms = np.linalg.norm(axes, axis=1, keepdims=True).clip(min=1e-12)
    return axes / norms


def _pairwise_angular_dist(axes_unit: np.ndarray) -> np.ndarray:
    """``[n, n]`` symmetric angular distance matrix (radians)."""
    dot = np.clip(axes_unit @ axes_unit.T, -1.0, 1.0)
    ang = np.arccos(dot)
    np.fill_diagonal(ang, 0.0)
    return ang


def _knn_graph_components_sweep(
    ang_dist: np.ndarray,
    theta_grid_deg: Sequence[float],
    percolation_threshold: float = 0.5,
) -> Tuple[Dict[float, int], Dict[float, float], Optional[float]]:
    """Probe A — at each θ, build adjacency, count components.

    Returns:
        components_at_thresh: θ_deg → n_components
        largest_pct_at_thresh: θ_deg → largest_component %
        percolation_theta_deg: smallest θ where largest ≥ percolation_threshold
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = ang_dist.shape[0]
    comps: Dict[float, int] = {}
    largest_pct: Dict[float, float] = {}
    percolation: Optional[float] = None
    for theta_deg in theta_grid_deg:
        theta_rad = np.radians(theta_deg)
        adj = (ang_dist <= theta_rad) & (ang_dist > 0)
        sparse_adj = csr_matrix(adj.astype(np.int8))
        n_comp, labels = connected_components(sparse_adj, directed=False)
        sizes = np.bincount(labels)
        largest = sizes.max()
        comps[float(theta_deg)] = int(n_comp)
        largest_pct[float(theta_deg)] = float(100 * largest / n)
        if percolation is None and (largest / n) >= percolation_threshold:
            percolation = float(theta_deg)
    return comps, largest_pct, percolation


def _local_intrinsic_dim(
    axes_unit: np.ndarray,
    ang_dist: np.ndarray,
    k: int = 10,
    rel_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Probe B — for each axis, take k nearest neighbors, run PCA on
    centered offsets, return (count-based dim, participation-ratio dim).

    Returns:
        local_dim_count: ``[n]`` integer count of eigvals > rel_threshold * top
        local_dim_pr:    ``[n]`` float participation ratio (Σλ)² / Σλ²
    """
    n, D = axes_unit.shape
    sorted_neighbors = np.argsort(ang_dist, axis=1)  # ascending; idx 0 is self
    local_dim_count = np.zeros(n, dtype=np.int32)
    local_dim_pr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nb_idx = sorted_neighbors[i, 1:k + 1]
        offsets = axes_unit[nb_idx] - axes_unit[i]
        offsets -= offsets.mean(axis=0)
        _U, sing, _Vt = np.linalg.svd(offsets, full_matrices=False)
        eigvals = np.sort((sing ** 2) / k)[::-1]
        rel = eigvals / max(eigvals[0], 1e-12)
        local_dim_count[i] = int((rel > rel_threshold).sum())
        s1 = eigvals.sum()
        s2 = (eigvals ** 2).sum()
        local_dim_pr[i] = float((s1 ** 2) / max(s2, 1e-20)) if s2 > 0 else 0.0
    return local_dim_count, local_dim_pr


def _persistent_homology(
    ang_dist: np.ndarray,
    maxdim: int = 2,
    thresh_deg: float = 20.0,
    do_cocycles: bool = False,
) -> Tuple[Optional[Dict[str, Dict]], float]:
    """Probe C — ripser persistent homology on the angular distance matrix.

    Returns (summary_dict, compute_seconds). ``summary_dict`` is None if
    ripser is unavailable. Otherwise has keys ``H0``, ``H1``, ..., each
    mapping to ``{'finite': [[birth, death], ...], 'n_finite': int,
    'n_infinite': int, 'cocycles': [...] (optional)}``.
    """
    if not HAVE_RIPSER:
        return None, 0.0
    t0 = time.time()
    thresh_rad = np.radians(thresh_deg)
    result = _ripser(
        ang_dist,
        distance_matrix=True,
        maxdim=maxdim,
        thresh=thresh_rad,
        do_cocycles=do_cocycles,
    )
    diagrams = result['dgms']
    summary: Dict[str, Dict] = {}
    for h_dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        summary[f'H{h_dim}'] = {
            'finite':     finite.tolist(),
            'n_finite':   int(len(finite)),
            'n_infinite': int(len(dgm) - len(finite)),
        }
    if do_cocycles and 'cocycles' in result:
        # ripser returns cocycles as a list per H_dim; serialize lazily.
        for h_dim, cc in enumerate(result['cocycles']):
            summary[f'H{h_dim}']['cocycles'] = [c.tolist() for c in cc]
    return summary, time.time() - t0


# ── Public topology entry points ────────────────────────────────────

def run_topology_analysis(
    codebook: Union[Codebook, torch.Tensor, np.ndarray],
    *,
    knn_threshold_grid_deg: Sequence[float] = (
        0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0,
    ),
    local_pca_k: int = 10,
    ripser_thresh_deg: float = 20.0,
    ripser_maxdim: int = 2,
    do_cocycles: bool = False,
    notes: str = '',
) -> TopologyReport:
    """Run all three probes (kNN-graph / local PCA / ripser PH) on a
    single axis cloud and return a ``TopologyReport``.

    Accepts a ``Codebook`` (uses ``.axes``), a torch tensor, or a numpy
    array of shape ``[n_axes, D]``. Axes are normalized to unit-norm
    internally; sign canonicalization is the caller's responsibility
    (codebooks emitted by ``extract_codebook`` are already canonicalized).
    """
    if isinstance(codebook, Codebook):
        axes = codebook.axes
    else:
        axes = codebook
    axes_unit = _normalize_axes(axes)
    n, D = axes_unit.shape

    ang_dist = _pairwise_angular_dist(axes_unit)
    triu = np.triu_indices(n, k=1)
    ang_off_diag = ang_dist[triu]

    # Probe A
    comps, largest_pct, percolation = _knn_graph_components_sweep(
        ang_dist, knn_threshold_grid_deg,
    )
    # Probe B
    local_count, local_pr = _local_intrinsic_dim(
        axes_unit, ang_dist, k=local_pca_k,
    )
    # Probe C
    persistence, ripser_seconds = _persistent_homology(
        ang_dist, maxdim=ripser_maxdim,
        thresh_deg=ripser_thresh_deg, do_cocycles=do_cocycles,
    )

    # Top persistent features per H_k (degrees)
    top_features: Dict[str, List[List[float]]] = {}
    persistence_diagrams: Dict[str, List[List[float]]] = {}
    persistence_n_finite: Dict[str, int] = {}
    persistence_n_infinite: Dict[str, int] = {}
    if persistence is not None:
        for h_key, info in persistence.items():
            finite = np.asarray(info['finite']) if info['finite'] else np.zeros((0, 2))
            persistence_diagrams[h_key] = finite.tolist()
            persistence_n_finite[h_key] = info['n_finite']
            persistence_n_infinite[h_key] = info['n_infinite']
            if len(finite) > 0:
                persistences = finite[:, 1] - finite[:, 0]
                top_idx = np.argsort(persistences)[::-1][:10]
                top_features[h_key] = [
                    [float(np.degrees(finite[i, 0])),
                     float(np.degrees(finite[i, 1])),
                     float(np.degrees(persistences[i]))]
                    for i in top_idx
                ]
            else:
                top_features[h_key] = []

    return TopologyReport(
        n_axes=int(n),
        D=int(D),
        angular_dist_p25_deg=float(np.degrees(np.percentile(ang_off_diag, 25))),
        angular_dist_p50_deg=float(np.degrees(np.percentile(ang_off_diag, 50))),
        angular_dist_p75_deg=float(np.degrees(np.percentile(ang_off_diag, 75))),
        angular_dist_p95_deg=float(np.degrees(np.percentile(ang_off_diag, 95))),
        knn_components_at_thresh=comps,
        knn_largest_pct_at_thresh=largest_pct,
        percolation_thresh_deg=percolation,
        local_dim_pr_p25=float(np.percentile(local_pr, 25)),
        local_dim_pr_p50=float(np.percentile(local_pr, 50)),
        local_dim_pr_p75=float(np.percentile(local_pr, 75)),
        local_dim_count_mode=int(np.bincount(local_count).argmax()) if len(local_count) else 0,
        persistence_diagrams=persistence_diagrams,
        persistence_n_finite=persistence_n_finite,
        persistence_n_infinite=persistence_n_infinite,
        top_persistent_features=top_features,
        ripser_thresh_deg=float(ripser_thresh_deg),
        ripser_compute_seconds=float(ripser_seconds),
        ripser_available=bool(HAVE_RIPSER),
        notes=notes,
    )


def run_array_topology_analysis(
    codebooks: Sequence[Codebook],
    bank_labels: Optional[Sequence[str]] = None,
    *,
    rare_quantile_pct: float = 85.0,
    common_quantile_pct: float = 15.0,
    do_cocycles: bool = True,
    **topology_kwargs,
) -> ArrayTopologyReport:
    """Pool axes across multiple codebooks; run single-cloud topology
    via ``run_topology_analysis``; add per-axis cross-bank kNN density
    and per-bank rare-axis counts.

    Use this for multi-bank arrays (e.g. h2-64's 192 banks).
    """
    if bank_labels is None:
        bank_labels = [f'bank_{i}' for i in range(len(codebooks))]
    if len(bank_labels) != len(codebooks):
        raise ValueError(
            f"bank_labels length {len(bank_labels)} != codebooks length {len(codebooks)}"
        )

    pooled_axes = []
    axis_bank = []
    n_axes_per_bank = []
    for label, cb in zip(bank_labels, codebooks):
        axes_np = cb.axes.detach().cpu().numpy() if isinstance(cb.axes, torch.Tensor) else np.asarray(cb.axes)
        n_axes_per_bank.append(int(len(axes_np)))
        for k in range(len(axes_np)):
            pooled_axes.append(axes_np[k])
            axis_bank.append(label)
    pooled_axes = np.asarray(pooled_axes, dtype=np.float64)
    pooled_axes /= np.linalg.norm(pooled_axes, axis=1, keepdims=True).clip(min=1e-12)
    axis_bank_arr = np.asarray(axis_bank)

    # Single-cloud topology on the pooled axes
    pooled_report = run_topology_analysis(
        pooled_axes, do_cocycles=do_cocycles, **topology_kwargs,
    )

    # Per-axis cross-bank kNN-1: for each axis, find nearest neighbor from
    # a DIFFERENT bank
    ang_dist = _pairwise_angular_dist(pooled_axes)
    n = ang_dist.shape[0]
    cross_bank_d_knn1 = np.full(n, np.inf)
    for i in range(n):
        same_bank = (axis_bank_arr == axis_bank_arr[i])
        candidates = ang_dist[i].copy()
        candidates[same_bank] = np.inf
        cross_bank_d_knn1[i] = candidates.min()

    cross_bank_d_knn1_deg = np.degrees(cross_bank_d_knn1)
    p10 = float(np.percentile(cross_bank_d_knn1_deg, 10))
    p50 = float(np.percentile(cross_bank_d_knn1_deg, 50))
    p85 = float(np.percentile(cross_bank_d_knn1_deg, rare_quantile_pct))
    rare_thresh = p85
    common_thresh = float(np.percentile(cross_bank_d_knn1_deg, common_quantile_pct))
    n_common = int((cross_bank_d_knn1_deg <= common_thresh).sum())
    n_rare = int((cross_bank_d_knn1_deg >= rare_thresh).sum())
    n_typical = int(n - n_common - n_rare)

    # Per-bank rare-axis counts (diversity ranking)
    rare_per_bank: Dict[str, int] = {label: 0 for label in bank_labels}
    is_rare = (cross_bank_d_knn1_deg >= rare_thresh)
    for i in range(n):
        if is_rare[i]:
            rare_per_bank[axis_bank_arr[i]] += 1

    return ArrayTopologyReport(
        pooled=pooled_report,
        n_banks=len(codebooks),
        bank_labels=list(bank_labels),
        n_axes_per_bank=n_axes_per_bank,
        cross_bank_d_knn1_deg=cross_bank_d_knn1_deg.tolist(),
        cross_bank_d_knn1_p10_p50_p85_deg=(p10, p50, p85),
        rare_axis_threshold_deg=rare_thresh,
        n_common_axes=n_common,
        n_typical_axes=n_typical,
        n_rare_axes=n_rare,
        rare_axes_per_bank=rare_per_bank,
    )


# ── End-to-end orchestrator ─────────────────────────────────────────

@torch.no_grad()
def create_codebook(
    model,
    cfg: Dict[str, Any],
    *,
    calibration_name: Optional[str] = None,
    calibration_n: Optional[int] = None,
    calibration_size: Optional[int] = None,
    sample_agg: str = 'mean',
    patch_agg: str = 'mean',
    patch_idx: Optional[int] = None,
    threshold: float = -0.9,
    model_id: str = '',
    out_dir: Union[str, Path] = './codebooks',
    upload_to_hf: bool = False,
    hf_repo: str = HF_REPO,
    hf_version: Optional[str] = None,
    run_topology: bool = True,
    topology_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Codebook, Optional[TopologyReport]]:
    """End-to-end codebook creation.

    Workflow:
        1. Infer model class from cfg if calibration not specified;
           pick default from ``DEFAULT_CALIBRATIONS``.
        2. Generate calibration via ``make_calibration``.
        3. Run ``extract_codebook`` with the chosen aggregations.
        4. Save Codebook to ``{out_dir}/{model_id}__{calibration_name}.{safetensors,json}``.
        5. (optional) Run ``run_topology_analysis`` and save alongside.
        6. (optional) Upload codebook + topology JSON to HF under
           ``{hf_version}/codebooks/{calibration_name}.*``.

    Returns:
        (codebook, topology_report). ``topology_report`` is None when
        ``run_topology=False`` or no axes were produced.
    """
    # 1. Resolve calibration
    cls = infer_class_from_cfg(cfg)
    defaults = DEFAULT_CALIBRATIONS.get(cls, DEFAULT_CALIBRATIONS['unknown'])
    cal_name = calibration_name or defaults['name']
    cal_n = calibration_n or defaults['n']
    cal_size = calibration_size or cfg.get('img_size') or defaults['size']

    print(f"  [create_codebook] class={cls!r}, "
          f"calibration={cal_name!r} (n={cal_n}, size={cal_size})")

    # 2. Generate calibration data
    calib = make_calibration(cal_name, n=cal_n, size=cal_size)
    if not isinstance(calib, torch.Tensor):
        calib = torch.as_tensor(calib)
    # If model expects a non-3-channel input, the calibration generator
    # produces 3-channel by default — repeat or slice to match.
    target_channels = int(cfg.get('channels', 3))
    if calib.shape[1] != target_channels:
        if target_channels < calib.shape[1]:
            calib = calib[:, :target_channels]
        else:
            reps = (target_channels + calib.shape[1] - 1) // calib.shape[1]
            calib = calib.repeat(1, reps, 1, 1)[:, :target_channels]

    # 3. Extract codebook
    cb = extract_codebook(
        model, calib,
        sample_agg=sample_agg,
        patch_agg=patch_agg,
        patch_idx=patch_idx,
        threshold=threshold,
        model_id=model_id,
        model_class=cls,
        calibration_name=cal_name,
    )
    print(f"  [create_codebook] {cb}")

    # 4. Save artifact
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = f"{model_id}__{cal_name}" if model_id else cal_name
    cb_stem = out_dir / stem_name
    cb.save(cb_stem)
    print(f"  [create_codebook] saved → {cb_stem}.{{safetensors,json}}")

    # 5. Topology analysis
    topology = None
    if run_topology and cb.n_axes >= 4:
        topo_kwargs = dict(topology_kwargs or {})
        topology = run_topology_analysis(cb, **topo_kwargs)
        topology.save(cb_stem.with_name(cb_stem.name + '__topology'))
        h_summary = ', '.join(
            f"{k}={v}" for k, v in topology.persistence_n_finite.items()
        )
        print(f"  [create_codebook] topology: {h_summary} "
              f"(ripser={'yes' if topology.ripser_available else 'NO'}, "
              f"{topology.ripser_compute_seconds:.1f}s)")
    elif run_topology:
        print(f"  [create_codebook] topology skipped (n_axes={cb.n_axes} < 4)")

    # 6. HF upload
    if upload_to_hf and hf_version:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            for ext in ('.safetensors', '.json'):
                api.upload_file(
                    path_or_fileobj=str(cb_stem) + ext,
                    path_in_repo=f"{hf_version}/codebooks/{cal_name}{ext}",
                    repo_id=hf_repo, repo_type='model',
                    commit_message=f"codebook: {model_id} on {cal_name}",
                )
            if topology is not None:
                topo_path = cb_stem.with_name(cb_stem.name + '__topology.json')
                api.upload_file(
                    path_or_fileobj=str(topo_path),
                    path_in_repo=f"{hf_version}/codebooks/{cal_name}__topology.json",
                    repo_id=hf_repo, repo_type='model',
                    commit_message=f"topology: {model_id} on {cal_name}",
                )
            print(f"  [create_codebook] uploaded to {hf_repo}/{hf_version}/codebooks/")
        except Exception as e:
            print(f"  [create_codebook] HF upload failed: {type(e).__name__}: {e}")

    return cb, topology


__all__ = [
    'DEFAULT_CALIBRATIONS',
    'infer_class_from_cfg',
    'TopologyReport',
    'ArrayTopologyReport',
    'run_topology_analysis',
    'run_array_topology_analysis',
    'create_codebook',
    'HAVE_RIPSER',
]

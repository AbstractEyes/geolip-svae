"""
geolip_svae.inference.codebook
================================
Projective-axis codebook as a first-class artifact.

Empirical foundation (scratchpad entry 000101):
Every trained sphere-solver tested at D=3 and D=4 (17 distinct models)
produces an M tensor whose rows, when antipodal pairs are collapsed via
mutual-strongest matching, form a uniformly-distributed codebook on
ℝP^(D-1). The collapse method is a deterministic tensor operation —
not a learned property, not a clustering result.

This module owns:
    - The four collapse helpers (``identify_antipodal_pairs``,
      ``collapse_to_axes``, ``_canonicalize_sign``, ``_aggregate_M``).
      These were previously module-level free functions in
      ``geolip_svae.arrays.model``; they live here now and that module
      re-imports them for backward compatibility.
    - The ``Codebook`` dataclass — axes + metadata + save/load.
    - ``extract_codebook(model, calibration_images, ...)`` — the
      model-agnostic extraction entry point.

Codebooks serialize as a ``.safetensors`` + ``.json`` sidecar pair so
they fit cleanly into HuggingFace artifact upload conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# ════════════════════════════════════════════════════════════════════
# Antipodal-collapse helpers (canonical home)
# ════════════════════════════════════════════════════════════════════
# Relocated from geolip_svae.arrays.model in the inference framework
# rebuild (scratchpad 000107). The four functions are pure tensor
# operations on M; they do not depend on any model class.

# This will be expanded as needed, but for now these are the only directly tested
# aggregation methods. The 'cat' method is a passthrough that leaves it to the caller to handle the extra dimension(s).
# This is less than ideal, but it's a research variable - so we can add more methods as needed without changing the API.
SUPPORTED_AGG = ('mean', 'median', 'first', 'cat')


def identify_antipodal_pairs(
    M: torch.Tensor,
    threshold: float = -0.9,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Find rows of M that form antipodal pairs (mutual-strongest matching).

    Args:
        M: ``[V, D]`` sphere-norm row vectors.
        threshold: cosine threshold for "antipodal", default -0.9.

    Returns:
        pairs:    list of ``(i, j)`` tuples with i < j
        unpaired: row indices with no antipodal partner
    """
    M = M.detach().cpu()
    norms = M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = M / norms
    cosines = unit @ unit.T
    cosines.fill_diagonal_(1.0)

    V = M.shape[0]
    claimed = [False] * V
    pairs: List[Tuple[int, int]] = []

    candidates = []
    for i in range(V):
        best_j = int(cosines[i].argmin())
        best_cos = float(cosines[i, best_j])
        if best_cos < threshold:
            candidates.append((best_cos, i, best_j))
    candidates.sort()

    for _cos, i, j in candidates:
        if claimed[i] or claimed[j]:
            continue
        if int(cosines[j].argmin()) == i or float(cosines[j, i]) < threshold:
            pairs.append((min(i, j), max(i, j)))
            claimed[i] = True
            claimed[j] = True

    unpaired = [i for i in range(V) if not claimed[i]]
    return pairs, unpaired


def _canonicalize_sign(v: torch.Tensor) -> torch.Tensor:
    """Flip ``v`` so its first nonzero coordinate is positive."""
    for k in range(v.shape[0]):
        if v[k].abs() > 1e-6:
            return -v if v[k] < 0 else v
    return v


def collapse_to_axes(
    M: torch.Tensor,
    pairs: List[Tuple[int, int]],
    unpaired: List[int],
) -> torch.Tensor:
    """Collapse antipodal pairs into single-axis representatives.

    Each pair ``(i, j)`` becomes one axis ``(unit_i - unit_j) / ||...||``,
    sign-canonicalized. Each unpaired row stays as itself, sign-canonicalized.

    Returns:
        axes: ``[n_axes, D]`` where ``n_axes = len(pairs) + len(unpaired)``.
    """
    M = M.detach().cpu()
    norms = M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = M / norms

    representatives = []
    for i, j in pairs:
        merged = unit[i] - unit[j]
        merged = merged / merged.norm().clamp_min(1e-12)
        representatives.append(_canonicalize_sign(merged))
    for i in unpaired:
        representatives.append(_canonicalize_sign(unit[i].clone()))

    if not representatives:
        return torch.empty(0, M.shape[1], dtype=M.dtype)
    return torch.stack(representatives, dim=0)


def _aggregate_M(
    M_stack: torch.Tensor,
    method: str,
    axis_label: str = '',
) -> torch.Tensor:
    """Aggregate M tensors stacked along dim 0.

    Args:
        M_stack: ``[N, V, D]`` tensor of M matrices.
        method: one of ``SUPPORTED_AGG``.
        axis_label: descriptive name for error messages ('sample', 'patch').

    Returns:
        - 'mean'   → ``[V, D]``
        - 'median' → ``[V, D]``
        - 'first'  → ``[V, D]``  (just M_stack[0])
        - 'cat'    → unchanged ``[N, V, D]`` — caller's job to handle
    """
    if method not in SUPPORTED_AGG:
        raise ValueError(
            f"Unknown {axis_label} aggregation '{method}'. "
            f"Supported: {SUPPORTED_AGG}"
        )
    if method == 'mean':
        return M_stack.mean(dim=0)
    if method == 'median':
        return M_stack.median(dim=0).values
    if method == 'first':
        return M_stack[0]
    return M_stack  # 'cat'


# ════════════════════════════════════════════════════════════════════
# Uniform projective baseline
# ════════════════════════════════════════════════════════════════════

def uniform_projective_angle(D: int, n_samples: int = 4096,
                               seed: int = 0) -> float:
    """Mean pairwise projective angle for uniformly random points on ℝP^(D-1).

    Computed empirically by drawing ``n_samples`` Gaussian vectors,
    normalizing, sign-canonicalizing, and averaging projective angles.
    Reproducible at fixed ``seed``.

    Used as the baseline against which a measured codebook's mean
    projective angle is compared. Deviation < 0.05 is the
    "projective-clean" threshold.
    """
    g = torch.Generator().manual_seed(int(seed))
    pts = torch.randn(n_samples, D, generator=g)
    pts = pts / pts.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pts = torch.stack([_canonicalize_sign(p) for p in pts])
    cos = pts @ pts.T
    cos = cos.clamp(-1, 1)
    angles = torch.acos(cos.abs())  # projective: fold to [0, π/2]
    iu = torch.triu_indices(n_samples, n_samples, offset=1)
    return float(angles[iu[0], iu[1]].mean())


def codebook_mean_projective_angle(axes: torch.Tensor) -> float:
    """Mean projective angle between rows of ``axes`` (codebook)."""
    if axes.shape[0] < 2:
        return float('nan')
    norms = axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = axes / norms
    cos = (unit @ unit.T).clamp(-1, 1)
    angles = torch.acos(cos.abs())
    iu = torch.triu_indices(axes.shape[0], axes.shape[0], offset=1)
    return float(angles[iu[0], iu[1]].mean())


# ════════════════════════════════════════════════════════════════════
# Codebook — first-class artifact
# ════════════════════════════════════════════════════════════════════

@dataclass
class CodebookMetadata:
    """Metadata describing a codebook's source and structure.

    All fields are JSON-serializable.
    """
    # ── Source ──
    model_id: str = ''                  # 'v40_freckles_noise', 'h2-64/battery_0', etc.
    model_class: str = ''               # 'PatchSVAE', 'BatteryArrayModel/sphere-bank'
    D: int = 0                          # singular-value dim
    V: int = 0                          # rows of M before collapse
    n_axes: int = 0                     # rows after collapse
    n_pairs: int = 0                    # antipodal pairs found
    n_unpaired: int = 0                 # singletons

    # ── Calibration ──
    calibration: str = ''               # name from CALIBRATION_REGISTRY
    n_calibration_images: int = 0
    calibration_size: int = 0           # H = W
    sample_agg: str = 'mean'
    patch_agg: str = 'mean'
    patch_idx: Optional[int] = None
    threshold: float = -0.9

    # ── Geometric ──
    mean_projective_angle: float = 0.0
    uniform_baseline: float = 0.0       # angle for uniform RP^(D-1)
    deviation: float = 0.0              # mean_proj_angle - uniform_baseline
    is_projective_clean: bool = False   # |deviation| < 0.05

    # ── Free-form ──
    notes: str = ''
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Codebook:
    """A projective-axis codebook with full provenance.

    Attributes:
        axes: ``[n_axes, D]`` sphere-normed sign-canonicalized rows.
        metadata: ``CodebookMetadata`` describing source + structure.
        pairs: list of original-row index pairs that were collapsed.
        unpaired: list of original-row indices kept as singletons.

    Save/load uses a safetensors + JSON sidecar pair:
        codebook.safetensors  — axes tensor + a packed pairs/unpaired tensor
        codebook.json         — metadata
    """
    axes: torch.Tensor
    metadata: CodebookMetadata
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    unpaired: List[int] = field(default_factory=list)

    # ── Geometric inspection ──

    @property
    def D(self) -> int:
        return int(self.axes.shape[1]) if self.axes.numel() else 0

    @property
    def n_axes(self) -> int:
        return int(self.axes.shape[0])

    def deviation(self) -> float:
        """Recompute deviation from uniform RP^(D-1) baseline."""
        uniform = uniform_projective_angle(self.D)
        observed = codebook_mean_projective_angle(self.axes)
        return observed - uniform

    def is_projective_clean(self, threshold: float = 0.05) -> bool:
        """True iff ``|deviation| < threshold``."""
        return abs(self.deviation()) < threshold

    # ── Compatibility ──

    def compatible_with(
        self,
        model,
        require_V: bool = False,
    ) -> Tuple[bool, str]:
        """Check whether this codebook can be applied to ``model``.

        Args:
            model: any model with a ``D`` attribute (PatchSVAE-like).
            require_V: if True, also require V match. Default False
                because codebook V is often less than model V due
                to antipodal collapse (n_axes ≤ V).

        Returns:
            (is_compatible, reason). reason is a human-readable
            explanation; empty string when compatible.
        """
        model_D = getattr(model, 'D', None)
        if model_D is None:
            return False, "model has no 'D' attribute"
        if int(model_D) != self.D:
            return False, (
                f"D mismatch: model.D={model_D}, codebook.D={self.D}"
            )
        if require_V:
            model_V = getattr(model, 'V', None)
            if model_V is None:
                return False, "model has no 'V' attribute (require_V=True)"
            if int(model_V) != self.metadata.V:
                return False, (
                    f"V mismatch: model.V={model_V}, "
                    f"codebook source V={self.metadata.V}"
                )
        return True, ''

    # ── Persistence ──

    def save(self, path: Union[str, Path]) -> Path:
        """Save as a safetensors + JSON sidecar pair.

        ``path`` is the stem; the function writes ``{stem}.safetensors``
        and ``{stem}.json``. Returns the resolved stem Path.
        """
        from safetensors.torch import save_file
        path = Path(path)
        # Strip a known extension if user passed one
        if path.suffix in {'.safetensors', '.json'}:
            stem = path.with_suffix('')
        else:
            stem = path
        stem.parent.mkdir(parents=True, exist_ok=True)

        # Pack pairs / unpaired as int64 tensors so safetensors can hold them
        pair_tensor = (
            torch.tensor(self.pairs, dtype=torch.int64)
            if self.pairs else torch.zeros(0, 2, dtype=torch.int64)
        )
        unpaired_tensor = torch.tensor(self.unpaired, dtype=torch.int64)

        save_file(
            {
                'axes': self.axes.contiguous().to(torch.float32),
                'pairs': pair_tensor.contiguous(),
                'unpaired': unpaired_tensor.contiguous(),
            },
            str(stem.with_suffix('.safetensors')),
        )
        with open(stem.with_suffix('.json'), 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)
        return stem

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Codebook':
        """Load a codebook from its safetensors + JSON sidecar pair."""
        from safetensors.torch import load_file
        path = Path(path)
        if path.suffix in {'.safetensors', '.json'}:
            stem = path.with_suffix('')
        else:
            stem = path

        st_path = stem.with_suffix('.safetensors')
        json_path = stem.with_suffix('.json')
        if not st_path.exists():
            raise FileNotFoundError(f"Codebook safetensors missing: {st_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Codebook json missing: {json_path}")

        tensors = load_file(str(st_path))
        with open(json_path) as f:
            meta_dict = json.load(f)

        # Tolerate metadata schema additions: filter to known fields
        known = set(CodebookMetadata.__dataclass_fields__.keys())
        filtered = {k: v for k, v in meta_dict.items() if k in known}
        extra_keys = set(meta_dict) - known
        if extra_keys:
            filtered.setdefault('extra', {})
            for k in extra_keys:
                filtered['extra'][k] = meta_dict[k]
        metadata = CodebookMetadata(**filtered)

        pairs_t = tensors.get('pairs')
        if pairs_t is not None and pairs_t.numel():
            pairs = [(int(p[0]), int(p[1])) for p in pairs_t.tolist()]
        else:
            pairs = []
        unpaired_t = tensors.get('unpaired')
        unpaired = (
            [int(x) for x in unpaired_t.tolist()]
            if unpaired_t is not None else []
        )

        return cls(
            axes=tensors['axes'],
            metadata=metadata,
            pairs=pairs,
            unpaired=unpaired,
        )

    # ── Display ──

    def __repr__(self) -> str:
        return (
            f"Codebook(D={self.D}, n_axes={self.n_axes}, "
            f"pairs={len(self.pairs)}, unpaired={len(self.unpaired)}, "
            f"dev={self.metadata.deviation:+.4f}, "
            f"clean={self.metadata.is_projective_clean})"
        )


# ════════════════════════════════════════════════════════════════════
# Extraction
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_codebook(
    model,
    calibration_images: torch.Tensor,
    sample_agg: str = 'mean',
    patch_agg: str = 'mean',
    patch_idx: Optional[int] = None,
    threshold: float = -0.9,
    batch_size: int = 64,
    *,
    model_id: str = '',
    model_class: str = '',
    calibration_name: str = '',
) -> Codebook:
    """Extract the projective-axis codebook from any sphere-solver model.

    Runs ``model(calibration_images)`` to collect M tensors, aggregates
    across samples and patches per the agg kwargs, then performs
    antipodal-pair collapse to produce the codebook. Returns a
    ``Codebook`` artifact with full provenance.

    Args:
        model: any model whose forward returns ``dict`` with
            ``'svd' → 'M'`` of shape ``[B, n_patches, V, D]``.
        calibration_images: ``[N, C, H, W]``
            The content and size of these images is a research variable.
            Many tests show that the codebook emerges even with pure noise, but the images
            must be large enough to activate the full V (rows of M) and D (columns of M).
            This is a verifiable research variable: the codebook is visible in each patch, however
            the sample and patch aggregation methods affect the clarity of the codebook.
            Without enough calibration images, the codebook may be incomplete (missing axes) or degenerate
            (spurious axes, low mean projective angle, etc.). Research is ongoing to understand the
            relationship between calibration and codebook quality.
        sample_agg: 'mean' (default), 'median', 'first', 'cat'.
            How to aggregate M across the N calibration samples.
            'mean' and 'median' produce a single codebook; 'cat' produces one per sample.
        patch_agg: 'mean' (default), 'median', 'first', 'cat'. Ignored
            'mean' and 'median' produce one M per image by aggregating across patches;
            if ``patch_idx`` is set.
        patch_idx: if set, use only this single patch index per image
            (legacy A0–A3 verification path). Default ``None`` =
            per-patch averaging (the corrected default from 000104).
            Research shows that the codebook emerges in EACH patch,
            so the codebook is visible even at a single patch index.
            This is a verifiable legacy path for the research.
        threshold: antipodal cosine threshold (default -0.9).
            Antipodal pairs must have cosine below this to be considered valid.
            Antipodal pairs are identified by mutual-strongest matching, but this
            threshold is a sanity check to prevent degenerate pairs in low-D models.
            This requires ablation to set properly per model architecture;
            -0.9 is a good starting point for D=3, D=4 models.
        batch_size: forward-pass chunk size.
        model_id: free-form identifier saved in metadata
            (e.g. ``'v40_freckles_noise'``, ``'h2-64/battery_0'``).
        model_class: e.g. ``'PatchSVAE'``, ``'BatteryArrayModel/sphere-bank'``.
        calibration_name: e.g. ``'gaussian'``, ``'sixteen_noise'``.

    Returns:
        ``Codebook`` with axes, metadata, pairs, unpaired.
    """
    model.eval()
    device = next(model.parameters()).device
    calibration_images = calibration_images.to(device)

    # ── Collect M tensors ──
    all_M = []
    N = calibration_images.shape[0]
    for start in range(0, N, batch_size):
        chunk = calibration_images[start:start + batch_size]
        out = model(chunk)
        if not (isinstance(out, dict) and 'svd' in out and 'M' in out['svd']):
            raise RuntimeError(
                "Model forward must return dict with 'svd' → 'M'. "
                "extract_codebook requires a sphere-solver model."
            )
        M = out['svd']['M']  # [B, n_patches, V, D]
        if patch_idx is not None:
            all_M.append(M[:, patch_idx].cpu())
        else:
            all_M.append(M.cpu())
    M_collected = torch.cat(all_M, dim=0)

    # ── Aggregate ──
    if patch_idx is not None:
        M_for_collapse = _aggregate_M(M_collected, sample_agg, 'sample')
        V_source = int(M_for_collapse.shape[0])
    else:
        N_, P, V, D_ = M_collected.shape
        V_source = V

        if patch_agg == 'cat':
            M_after_patch = M_collected.reshape(N_ * P, V, D_)
        else:
            per_image = []
            for n in range(N_):
                per_image.append(
                    _aggregate_M(M_collected[n], patch_agg, 'patch')
                )
            M_after_patch = torch.stack(per_image, dim=0)

        M_for_collapse = _aggregate_M(M_after_patch, sample_agg, 'sample')

    if M_for_collapse.dim() == 3:
        K, V, D_ = M_for_collapse.shape
        M_for_collapse = M_for_collapse.reshape(K * V, D_)

    pairs, unpaired = identify_antipodal_pairs(M_for_collapse, threshold=threshold)
    axes = collapse_to_axes(M_for_collapse, pairs, unpaired)

    # ── Build metadata ──
    D = int(axes.shape[1]) if axes.numel() else 0
    n_axes = int(axes.shape[0])
    H = int(calibration_images.shape[-2])

    if D > 0 and n_axes > 1:
        observed = codebook_mean_projective_angle(axes)
        uniform = uniform_projective_angle(D)
        dev = observed - uniform
        clean = abs(dev) < 0.05
    else:
        observed = float('nan')
        uniform = float('nan')
        dev = float('nan')
        clean = False

    metadata = CodebookMetadata(
        model_id=model_id,
        model_class=model_class or type(model).__name__,
        D=D,
        V=V_source,
        n_axes=n_axes,
        n_pairs=len(pairs),
        n_unpaired=len(unpaired),
        calibration=calibration_name,
        n_calibration_images=N,
        calibration_size=H,
        sample_agg=sample_agg,
        patch_agg=patch_agg,
        patch_idx=patch_idx,
        threshold=threshold,
        mean_projective_angle=float(observed) if observed == observed else 0.0,
        uniform_baseline=float(uniform) if uniform == uniform else 0.0,
        deviation=float(dev) if dev == dev else 0.0,
        is_projective_clean=clean,
    )

    return Codebook(
        axes=axes,
        metadata=metadata,
        pairs=pairs,
        unpaired=unpaired,
    )


__all__ = [
    # Helpers
    'identify_antipodal_pairs',
    'collapse_to_axes',
    'SUPPORTED_AGG',
    # Geometry
    'uniform_projective_angle',
    'codebook_mean_projective_angle',
    # Codebook artifact
    'CodebookMetadata',
    'Codebook',
    # Extraction
    'extract_codebook',
]
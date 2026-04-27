"""
geolip_svae.tests.framework
============================
Reusable test harness for Phase U lens-scope characterization.

Each Phase U cell (U1-U6) subclasses ``LensScopeTestCase`` and provides:
    - A name and description
    - A list of subcells, each defining a perturbation axis
    - For each subcell: a generator producing perturbed inputs, plus
      a strength label per perturbed input

The framework runs all subcells, computes the three measurement axes
(M1 codebook Procrustes, M2 CV deviation, M3 reconstruction MSE ratio)
on each, and produces standardized outputs (JSON + plot).

Three measurement axes:

    M1: Procrustes distance between perturbed-input codebook and
        reference codebook. Measures whether the LENS itself moves.
        Low M1 (< 0.05): codebook is invariant to the perturbation.
        High M1: the lens is shifting under the perturbation.

    M2: |observed_cv_perturbed − uniform_RP^(D-1)_baseline|. Measures
        whether the projective BASIN shifts. Low M2 (< 0.05): codebook
        stays projective-clean. High M2: codebook moves off the basin.

    M3: test_mse(perturbed) / native_mse. Measures whether the LIGHT
        passes through with information preserved. Low M3 (< ~5×):
        reconstruction quality holds. High M3: reconstruction breaks.

A perturbation can preserve some axes but not others. Headline metric
is whichever breaks first as perturbation strength scales — that's
the diagnostic value.

Usage::

    from geolip_svae.tests.framework import LensScopeTestCase

    class MyCell(LensScopeTestCase):
        name = "U2.5"
        description = "Variance scale"

        def subcells(self):
            yield "scale_axis", self._scale_axis_subcell

        def _scale_axis_subcell(self, ref_input):
            for sigma in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
                yield f"sigma={sigma}", ref_input * sigma

    cell = MyCell(model=model, ref_codebook=cb,
                  ref_input=calib, output_dir='./out')
    cell.run()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from geolip_svae.inference.codebook import (
    Codebook,
    codebook_mean_projective_angle,
    extract_codebook,
    uniform_projective_angle,
)


# ════════════════════════════════════════════════════════════════════
# Measurement axes
# ════════════════════════════════════════════════════════════════════

def procrustes_distance(
    A: torch.Tensor,
    B: torch.Tensor,
) -> float:
    """Procrustes distance between two codebooks A and B.

    Both ``[n_axes, D]``. Returns the minimum Frobenius distance after
    optimal rotation alignment. Honors that codebooks may have
    different ``n_axes`` by aligning on the smaller and reporting the
    distance over that aligned subset.

    For projective codebooks each row is also free to flip sign; we
    take the absolute value of the cross-correlation matrix before SVD
    so antipodal-equivalent rows match correctly.
    """
    A = A.detach().cpu().to(torch.float64)
    B = B.detach().cpu().to(torch.float64)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"D mismatch: A.D={A.shape[1]}, B.D={B.shape[1]}"
        )
    n = min(A.shape[0], B.shape[0])
    if n < 2:
        return float('nan')
    A_n = A[:n]
    B_n = B[:n]
    # Procrustes: A ≈ B @ R, R = argmin ||A - B@R||
    # Standard solution: SVD of A.T @ B
    M = A_n.T @ B_n
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    aligned = B_n @ R.T
    return float((A_n - aligned).norm()) / max(n ** 0.5, 1.0)


def cv_deviation(
    codebook: Codebook,
    uniform_baseline: Optional[float] = None,
) -> float:
    """|observed mean projective angle − uniform RP^(D-1) baseline|.

    Uses the codebook's own metadata if uniform_baseline isn't passed.
    """
    if codebook.n_axes < 2:
        return float('nan')
    observed = codebook_mean_projective_angle(codebook.axes)
    if uniform_baseline is None:
        uniform_baseline = uniform_projective_angle(codebook.D)
    return abs(observed - uniform_baseline)


def reconstruction_mse_ratio(
    perturbed_mse: float,
    native_mse: float,
    eps: float = 1e-12,
) -> float:
    """Ratio of perturbed MSE to native MSE. Higher = more degradation."""
    return perturbed_mse / max(native_mse, eps)


# ════════════════════════════════════════════════════════════════════
# Subcell + Cell datatypes
# ════════════════════════════════════════════════════════════════════

@dataclass
class SubcellPoint:
    """A single (strength_label, perturbed_input) pair within a subcell."""
    strength_label: str
    inputs: torch.Tensor

    # Filled by the framework after measurement:
    M1: float = float('nan')
    M2: float = float('nan')
    M3: float = float('nan')
    n_axes_perturbed: int = 0
    notes: str = ''


@dataclass
class SubcellResult:
    """All measurements from one subcell."""
    name: str
    description: str
    points: List[SubcellPoint] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ''
    elapsed_s: float = 0.0


@dataclass
class CellResult:
    """All measurements from one cell."""
    cell_name: str
    description: str
    subcells: List[SubcellResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: str = ''


# ════════════════════════════════════════════════════════════════════
# Skip-threshold defaults
# ════════════════════════════════════════════════════════════════════

@dataclass
class SkipPolicy:
    """When to abort a subcell early on lack of measurable effect."""
    enabled: bool = True
    min_points_before_skip: int = 2
    threshold_M1: float = 1e-3
    threshold_M2: float = 1e-3
    threshold_M3_log: float = 0.05  # |log10(M3)| < this means ~native MSE


# ════════════════════════════════════════════════════════════════════
# LensScopeTestCase base
# ════════════════════════════════════════════════════════════════════

# Type alias for a subcell generator: yields (strength_label, perturbed_inputs)
SubcellGen = Callable[[torch.Tensor], Iterator[Tuple[str, torch.Tensor]]]


class LensScopeTestCase:
    """Base class for Phase U lens-scope test cells.

    Subclasses override:
        ``name`` (str): cell identifier, e.g. "U2.5"
        ``description`` (str): human-readable summary
        ``subcells()``: generator yielding ``(subcell_name, subcell_fn)``
            pairs. Each ``subcell_fn(ref_input)`` is itself a generator
            yielding ``(strength_label, perturbed_input)`` tuples.

    Args:
        model: any PatchSVAE-like model.
        ref_codebook: the reference codebook computed at native conditions.
        ref_input: native calibration tensor used as the unperturbed reference.
        output_dir: where to write per-cell outputs.
        native_mse: pre-computed native MSE for M3 normalization. If None,
            the framework computes it on first ref_input forward pass.
        codebook_kwargs: kwargs forwarded to ``extract_codebook`` for
            perturbed inputs (sample_agg, patch_agg, threshold, etc.).
        skip_policy: when to abort a subcell early.
        verbose: print progress as subcells run.
    """

    name: str = "Cell"
    description: str = ""

    def __init__(
        self,
        model,
        ref_codebook: Codebook,
        ref_input: torch.Tensor,
        output_dir: Union[str, Path],
        *,
        native_mse: Optional[float] = None,
        codebook_kwargs: Optional[Dict[str, Any]] = None,
        skip_policy: Optional[SkipPolicy] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.ref_codebook = ref_codebook
        self.ref_input = ref_input
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codebook_kwargs = codebook_kwargs or {}
        self.skip_policy = skip_policy or SkipPolicy()
        self.verbose = verbose

        self._uniform_baseline = uniform_projective_angle(self.ref_codebook.D)

        if native_mse is None:
            native_mse = self._compute_native_mse()
        self.native_mse = float(native_mse)

    # ── Subclass hooks ───────────────────────────────────────────────

    def subcells(self) -> Iterator[Tuple[str, SubcellGen]]:
        """Yield ``(subcell_name, subcell_fn)`` pairs.

        Each ``subcell_fn(ref_input)`` is itself a generator yielding
        ``(strength_label, perturbed_input)`` tuples. Override in subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement subcells()"
        )

    # ── Measurement primitives ───────────────────────────────────────

    @torch.no_grad()
    def _compute_native_mse(self) -> float:
        """MSE of model on the unperturbed reference input."""
        device = next(self.model.parameters()).device
        x = self.ref_input.to(device)
        recon = self.model(x)['recon']
        return float(((recon - x) ** 2).mean())

    @torch.no_grad()
    def _measure_point(
        self,
        perturbed_inputs: torch.Tensor,
    ) -> Tuple[float, float, float, int]:
        """Measure M1, M2, M3, and n_axes for one perturbed input batch.

        Returns ``(M1, M2, M3, n_axes_perturbed)``.
        """
        device = next(self.model.parameters()).device
        perturbed_inputs = perturbed_inputs.to(device)

        # MSE on perturbed (M3 numerator)
        recon = self.model(perturbed_inputs)['recon']
        perturbed_mse = float(((recon - perturbed_inputs) ** 2).mean())
        M3 = reconstruction_mse_ratio(perturbed_mse, self.native_mse)

        # Codebook on perturbed inputs (drives M1 and M2)
        try:
            cb_perturbed = extract_codebook(
                self.model, perturbed_inputs,
                **self.codebook_kwargs,
            )
        except Exception as e:
            if self.verbose:
                print(f"      ⚠ codebook extraction failed: "
                      f"{type(e).__name__}: {str(e)[:80]}")
            return float('nan'), float('nan'), M3, 0

        if cb_perturbed.D != self.ref_codebook.D:
            # Non-comparable codebooks (shouldn't happen unless model swapped)
            return float('nan'), float('nan'), M3, cb_perturbed.n_axes

        M1 = procrustes_distance(self.ref_codebook.axes, cb_perturbed.axes)
        M2 = cv_deviation(cb_perturbed, self._uniform_baseline)
        return M1, M2, M3, cb_perturbed.n_axes

    def _should_skip(
        self,
        points_so_far: List[SubcellPoint],
    ) -> Tuple[bool, str]:
        """Decide whether to abort the subcell on insufficient measurable effect."""
        if not self.skip_policy.enabled:
            return False, ''
        if len(points_so_far) < self.skip_policy.min_points_before_skip:
            return False, ''

        for p in points_so_far:
            # If any of M1/M2 is NaN we DO NOT skip (signal might be hiding
            # behind a transient extraction failure)
            if not (p.M1 == p.M1 and p.M2 == p.M2):
                return False, ''

            big_M1 = p.M1 > self.skip_policy.threshold_M1
            big_M2 = p.M2 > self.skip_policy.threshold_M2
            big_M3 = abs(np.log10(max(p.M3, 1e-12))) > self.skip_policy.threshold_M3_log
            if big_M1 or big_M2 or big_M3:
                return False, ''

        # All measured points show no effect on any axis → skip remainder
        return True, (
            f"first {len(points_so_far)} points show no measurable effect "
            f"(M1<{self.skip_policy.threshold_M1}, "
            f"M2<{self.skip_policy.threshold_M2}, "
            f"|log10 M3|<{self.skip_policy.threshold_M3_log}); "
            f"skipping remainder of subcell."
        )

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> CellResult:
        """Run all subcells. Saves JSON to ``output_dir/{cell_name}.json``."""
        if self.verbose:
            print(f"\n{'═' * 68}")
            print(f"Cell {self.name}: {self.description}")
            print(f"{'═' * 68}")
            print(f"  Reference codebook: D={self.ref_codebook.D}, "
                  f"n_axes={self.ref_codebook.n_axes}, "
                  f"deviation={self.ref_codebook.metadata.deviation:+.4f}")
            print(f"  Native MSE:         {self.native_mse:.6f}")
            print(f"  Uniform baseline:   {self._uniform_baseline:.4f}")

        cell_t0 = time.time()
        cell = CellResult(cell_name=self.name, description=self.description)

        for subname, subfn in self.subcells():
            sub = self._run_subcell(subname, subfn)
            cell.subcells.append(sub)

        cell.elapsed_s = time.time() - cell_t0
        self._save_json(cell)
        if self.verbose:
            print(f"\n  Cell {self.name} elapsed: {cell.elapsed_s:.1f}s")
        return cell

    def _run_subcell(
        self,
        subname: str,
        subfn: SubcellGen,
    ) -> SubcellResult:
        if self.verbose:
            print(f"\n  ── {self.name}.{subname} ──")

        sub = SubcellResult(name=subname, description=subname)
        sub_t0 = time.time()

        for strength_label, perturbed in subfn(self.ref_input):
            t0 = time.time()
            M1, M2, M3, n_ax = self._measure_point(perturbed)
            point = SubcellPoint(
                strength_label=strength_label,
                inputs=torch.empty(0),  # don't store inputs, only metadata
                M1=M1, M2=M2, M3=M3, n_axes_perturbed=n_ax,
            )
            sub.points.append(point)
            if self.verbose:
                print(
                    f"      {strength_label:<24s} "
                    f"M1={M1:.4f}  M2={M2:.4f}  M3={M3:.3f}  "
                    f"n_ax={n_ax}  ({time.time()-t0:.1f}s)"
                )

            should_skip, reason = self._should_skip(sub.points)
            if should_skip:
                sub.skipped = True
                sub.skip_reason = reason
                if self.verbose:
                    print(f"      ⏭  {reason}")
                break

        sub.elapsed_s = time.time() - sub_t0
        return sub

    # ── Persistence ──────────────────────────────────────────────────

    def _save_json(self, cell: CellResult) -> Path:
        path = self.output_dir / f"{self.name}.json"
        # Convert SubcellPoint.inputs (tensor) out of the dump
        data = asdict(cell)
        for sub in data['subcells']:
            for p in sub['points']:
                p.pop('inputs', None)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path


__all__ = [
    # Measurement primitives
    'procrustes_distance',
    'cv_deviation',
    'reconstruction_mse_ratio',
    # Datatypes
    'SubcellPoint',
    'SubcellResult',
    'CellResult',
    'SkipPolicy',
    # Base class
    'LensScopeTestCase',
]
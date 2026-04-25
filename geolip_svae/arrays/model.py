"""
geolip_svae.arrays.model
==========================
BatteryArrayModel — generic PreTrainedModel that dispatches to any
sphere-solver battery class declared in its config.

Sampling APIs (all take images, return sampled output):

    forward(images)                    — MSE signature [B, n_batt, n_phase]
    compute_signature(images)          — same as forward, with phase filter
    forward_full(images)               — recon + per-bank outputs (debug)

    compute_axis_codebook(...)         — one bank's projective-axis codebook
    compute_axis_codebooks(...)        — batched: many banks at once
    encode_axes(images, ...)           — per-patch axis activations
    quantize_axes(images, ...)         — discrete codes via argmax

Empirical foundation (000101 in scratchpad):
Every trained sphere-solver tested (19 models across D ∈ {3, 4, 5})
produces an M tensor whose rows, when antipodal pairs are collapsed,
form a uniformly-distributed codebook on ℝP^(D-1).

Reference: AbstractPhil/geolip-svae-implicit-solver-experiments
"""

import importlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from geolip_svae.arrays.config import BatteryArrayConfig


# ════════════════════════════════════════════════════════════════════
# Module-level helpers for projective-axis collapse
# ════════════════════════════════════════════════════════════════════

def identify_antipodal_pairs(
    M: torch.Tensor,
    threshold: float = -0.9,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Find rows of M that form antipodal pairs (mutual-strongest matching).

    Args:
        M: [V, D] sphere-norm row vectors
        threshold: cosine threshold for "antipodal", default -0.9

    Returns:
        pairs: list of (i, j) tuples with i < j
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


def collapse_to_axes(
    M: torch.Tensor,
    pairs: List[Tuple[int, int]],
    unpaired: List[int],
) -> torch.Tensor:
    """Collapse antipodal pairs into single-axis representatives.

    Each pair (i, j) becomes one axis: (row_i - row_j) / 2 normalized.
    Each axis is sign-canonicalized.

    Returns:
        axes: [n_axes, D] where n_axes = len(pairs) + len(unpaired)
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


def _canonicalize_sign(v: torch.Tensor) -> torch.Tensor:
    """Flip v so its first nonzero coordinate is positive."""
    for k in range(v.shape[0]):
        if v[k].abs() > 1e-6:
            return -v if v[k] < 0 else v
    return v


# ════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ════════════════════════════════════════════════════════════════════

SUPPORTED_AGG = ('mean', 'median', 'first', 'cat')


def _aggregate_M(
    M_stack: torch.Tensor,
    method: str,
    axis_label: str = '',
) -> torch.Tensor:
    """Aggregate M tensors stacked along dim 0.

    Args:
        M_stack: [N, V, D] tensor of M matrices
        method: one of SUPPORTED_AGG
        axis_label: descriptive name for error messages ('sample', 'patch')

    Returns:
        - 'mean'   → [V, D]
        - 'median' → [V, D]
        - 'first'  → [V, D] (just M_stack[0])
        - 'cat'    → unchanged [N, V, D] — caller's job to handle multi-row
                     codebook from concatenated samples/patches
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
    return M_stack  # 'cat' — caller handles


# ════════════════════════════════════════════════════════════════════
# BatteryArrayModel
# ════════════════════════════════════════════════════════════════════

class BatteryArrayModel(PreTrainedModel):
    """Generic array over N batteries × K training phases = N*K banks.

    Architecture-agnostic: the battery class is resolved at init time by
    importing config.battery_module and getattr'ing config.battery_class.
    Each bank receives config.battery_kwargs.

    State dict keys follow the pattern:
        banks.{bank_idx}.{battery_state_dict_key}
    where bank_idx = battery_idx * n_epoch_phases + phase_idx.
    """

    config_class = BatteryArrayConfig
    base_model_prefix = "battery_array"
    supports_gradient_checkpointing = False

    def __init__(self, config: BatteryArrayConfig):
        super().__init__(config)
        self.config = config

        battery_cls = self._resolve_battery_class()

        self.banks = nn.ModuleList([
            battery_cls(**config.battery_kwargs)
            for _ in range(config.n_banks)
        ])

        self.post_init()

    def _resolve_battery_class(self):
        module = importlib.import_module(self.config.battery_module)
        try:
            return getattr(module, self.config.battery_class)
        except AttributeError:
            raise ImportError(
                f"Battery class '{self.config.battery_class}' not found in "
                f"module '{self.config.battery_module}'. Check that "
                f"geolip-svae (or the package providing the class) is installed."
            )

    # ── Bank access ──────────────────────────────────────────────────

    def bank(self, battery_idx: int, phase: str) -> nn.Module:
        return self.banks[self.config.bank_index(battery_idx, phase)]

    def bank_by_flat_idx(self, bank_idx: int) -> nn.Module:
        return self.banks[bank_idx]

    # ── MSE sampling ─────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.compute_signature(images)

    @torch.no_grad()
    def compute_signature(
        self,
        images: torch.Tensor,
        phase: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute per-bank MSE signature.

        Returns [B, n_batteries, n_epoch_phases] or
        [B, n_batteries] if phase given.
        """
        B = images.shape[0]
        n_batt = self.config.n_batteries
        n_phase = self.config.n_epoch_phases

        mse_flat = torch.empty(
            B, self.config.n_banks,
            device=images.device, dtype=images.dtype,
        )

        for bank_idx, bank in enumerate(self.banks):
            out = bank(images)
            recon = out['recon'] if isinstance(out, dict) else out
            mse_per_image = F.mse_loss(
                recon, images, reduction='none'
            ).mean(dim=(1, 2, 3))
            mse_flat[:, bank_idx] = mse_per_image

        mse_signature = mse_flat.reshape(B, n_batt, n_phase)

        if phase is not None:
            phase_idx = self.config.epoch_phase_names.index(phase)
            return mse_signature[:, :, phase_idx]
        return mse_signature

    @torch.no_grad()
    def forward_full(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full per-bank output — recon + any additional battery outputs.

        Memory-heavy. Use for debugging or small-batch analysis.
        """
        B, C, H, W = images.shape

        recon_all = torch.empty(
            B, self.config.n_banks, C, H, W,
            device=images.device, dtype=images.dtype,
        )
        bank_outputs = []

        for bank_idx, bank in enumerate(self.banks):
            out = bank(images)
            if isinstance(out, dict):
                recon_all[:, bank_idx] = out['recon']
                bank_outputs.append(out)
            else:
                recon_all[:, bank_idx] = out
                bank_outputs.append({'recon': out})

        mse_per_bank = F.mse_loss(
            recon_all,
            images.unsqueeze(1).expand(-1, self.config.n_banks, -1, -1, -1),
            reduction='none',
        ).mean(dim=(2, 3, 4))
        mse_signature = mse_per_bank.reshape(
            B, self.config.n_batteries, self.config.n_epoch_phases
        )

        return {
            'mse_signature': mse_signature,
            'recon_per_bank': recon_all,
            'bank_outputs': bank_outputs,
        }

    # ── Projective-axis sampling ─────────────────────────────────────

    @torch.no_grad()
    def _bank_M_collected(
        self,
        bank: nn.Module,
        calibration_images: torch.Tensor,
        patch_idx: Optional[int] = None,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Run calibration_images through bank, return stacked M tensors.

        Returns:
            If patch_idx is None: [N_images, n_patches, V, D] — full spatial
            If patch_idx is int:  [N_images, V, D] — single patch (legacy)
        """
        device = next(bank.parameters()).device
        calibration_images = calibration_images.to(device)

        all_M = []
        N = calibration_images.shape[0]
        for start in range(0, N, batch_size):
            chunk = calibration_images[start:start + batch_size]
            out = bank(chunk)
            if not isinstance(out, dict) or 'svd' not in out:
                raise RuntimeError(
                    "Bank forward must return dict with 'svd' key. "
                    "Axis sampling requires a sphere-solver battery."
                )
            M = out['svd']['M']  # [B, n_patches, V, D]
            if patch_idx is not None:
                all_M.append(M[:, patch_idx].cpu())
            else:
                all_M.append(M.cpu())
        return torch.cat(all_M, dim=0)

    @torch.no_grad()
    def compute_axis_codebook(
        self,
        battery_idx: int,
        phase: str,
        calibration_images: torch.Tensor,
        sample_agg: str = 'mean',
        patch_agg: str = 'mean',
        patch_idx: Optional[int] = None,
        threshold: float = -0.9,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Sample the projective-axis codebook from one trained bank.

        Args:
            battery_idx: which battery
            phase: which training phase ('epoch_1' / 'best' / 'final' typically)
            calibration_images: [N, C, H, W] inputs for averaging
            sample_agg: how to aggregate across calibration images.
                'mean' (default), 'median', 'first', 'cat'.
            patch_agg: how to aggregate across patches per image.
                'mean' (default), 'median', 'first', 'cat'. Ignored if
                patch_idx is set.
            patch_idx: if set, use only this single patch index per image
                (legacy behavior matching A0-A3 probes — the structural
                projective verification path). Overrides patch_agg. Use this
                to reproduce 000101's verified results exactly.
            threshold: antipodal cosine threshold (default -0.9)
            batch_size: forward-pass batch size

        Returns:
            codebook: [n_axes, D] sphere-normed sign-canonicalized axes.

        Aggregation semantics:
            mean/median/first reduce to a single [V, D] M_avg before the
            collapse step. 'cat' at either level keeps the rows separate
            so the collapse runs on a much larger M, producing a richer
            (and larger) codebook.
        """
        bank = self.bank(battery_idx, phase)
        bank.eval()

        # Collect M (with optional single-patch shortcut for legacy)
        M_collected = self._bank_M_collected(
            bank, calibration_images,
            patch_idx=patch_idx, batch_size=batch_size,
        )

        # Aggregate
        if patch_idx is not None:
            # Single-patch path — only sample agg applies
            M_for_collapse = _aggregate_M(M_collected, sample_agg, 'sample')
        else:
            # Full per-patch tensor [N, P, V, D]
            N, P, V, D = M_collected.shape

            # Step 1: patch aggregation per image
            if patch_agg == 'cat':
                # Flatten patch dim into sample dim → [N*P, V, D]
                M_after_patch = M_collected.reshape(N * P, V, D)
            else:
                # Aggregate per image's patches independently
                per_image = []
                for n in range(N):
                    per_image.append(
                        _aggregate_M(M_collected[n], patch_agg, 'patch')
                    )
                M_after_patch = torch.stack(per_image, dim=0)

            # Step 2: sample aggregation
            M_for_collapse = _aggregate_M(
                M_after_patch, sample_agg, 'sample',
            )

        # If aggregation returned a stack (cat used), flatten the
        # outer dim into V to give one big M tensor for collapse
        if M_for_collapse.dim() == 3:
            K, V, D = M_for_collapse.shape
            M_for_collapse = M_for_collapse.reshape(K * V, D)

        pairs, unpaired = identify_antipodal_pairs(
            M_for_collapse, threshold=threshold,
        )
        return collapse_to_axes(M_for_collapse, pairs, unpaired)

    @torch.no_grad()
    def compute_axis_codebooks(
        self,
        targets: Sequence[Tuple[int, str]],
        calibration_images: Union[
            torch.Tensor, Dict[Tuple[int, str], torch.Tensor]
        ],
        sample_agg: str = 'mean',
        patch_agg: str = 'mean',
        patch_idx: Optional[int] = None,
        threshold: float = -0.9,
        batch_size: int = 64,
    ) -> Dict[Tuple[int, str], torch.Tensor]:
        """Batched codebook calibration across many (battery, phase) pairs.

        Args:
            targets: list of (battery_idx, phase) tuples to calibrate
            calibration_images: either:
                - a single tensor [N, C, H, W] used for all targets
                  (typical: gaussian noise as a shared probe)
                - a dict {(battery_idx, phase): tensor} for per-target
                  calibration distributions
            sample_agg, patch_agg, patch_idx, threshold, batch_size:
                forwarded to compute_axis_codebook

        Returns:
            codebooks: dict {(battery_idx, phase): [n_axes, D] tensor}

        Example:
            # Single calibration set, many banks
            targets = [(i, 'final') for i in range(16)]
            codebooks = model.compute_axis_codebooks(
                targets, gaussian_calib_imgs,
            )

            # Per-target calibration distributions
            calib_per = {(i, 'final'): noise_imgs_for_type(i)
                          for i in range(16)}
            codebooks = model.compute_axis_codebooks(targets, calib_per)
        """
        is_per_target = isinstance(calibration_images, dict)

        codebooks = {}
        for tgt in targets:
            calib = (calibration_images[tgt] if is_per_target
                     else calibration_images)
            codebooks[tgt] = self.compute_axis_codebook(
                battery_idx=tgt[0],
                phase=tgt[1],
                calibration_images=calib,
                sample_agg=sample_agg,
                patch_agg=patch_agg,
                patch_idx=patch_idx,
                threshold=threshold,
                batch_size=batch_size,
            )
        return codebooks

    @torch.no_grad()
    def encode_axes(
        self,
        images: torch.Tensor,
        battery_idx: int,
        phase: str,
        codebook: torch.Tensor,
        patch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode images against a bank's axis codebook.

        Each row of the bank's M tensor gets cosine-matched against
        every codebook axis. Uses absolute cosine since antipodal axes
        represent the same line on ℝP^(D-1).

        Args:
            images: [B, C, H, W]
            battery_idx, phase: which bank to read M from
            codebook: [n_axes, D] from compute_axis_codebook
            patch_idx: if set, use only that patch → returns [B, V, n_axes].
                Otherwise return full per-patch activations
                → [B, n_patches, V, n_axes].

        Returns:
            activations: absolute cosines in [0, 1].
                [B, V, n_axes] if patch_idx is set, else
                [B, n_patches, V, n_axes].
        """
        bank = self.bank(battery_idx, phase)
        bank.eval()

        device = next(bank.parameters()).device
        images = images.to(device)
        codebook = codebook.to(device).to(images.dtype)

        out = bank(images)
        M_full = out['svd']['M']  # [B, n_patches, V, D]

        if patch_idx is not None:
            M = M_full[:, patch_idx]  # [B, V, D]
            norms = M.norm(dim=2, keepdim=True).clamp_min(1e-12)
            unit = M / norms
            return torch.einsum('bvd,nd->bvn', unit, codebook).abs()
        else:
            norms = M_full.norm(dim=3, keepdim=True).clamp_min(1e-12)
            unit = M_full / norms
            return torch.einsum('bpvd,nd->bpvn', unit, codebook).abs()

    @torch.no_grad()
    def quantize_axes(
        self,
        images: torch.Tensor,
        battery_idx: int,
        phase: str,
        codebook: torch.Tensor,
        patch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Hard quantization: each input row → strongest codebook axis index.

        Returns:
            codes: integer indices into codebook.
                [B, V] if patch_idx is set, else [B, n_patches, V].
        """
        return self.encode_axes(
            images, battery_idx, phase, codebook, patch_idx,
        ).argmax(dim=-1)
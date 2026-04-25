"""
geolip_svae.arrays.model
==========================
BatteryArrayModel — generic PreTrainedModel that dispatches to any
sphere-solver battery class declared in its config.

The array holds n_batteries × n_epoch_phases bank instances, all of the
same battery class with the same architecture kwargs. Banks differ only
in their loaded weights (which epoch of which training run).

Sampling APIs (all take images, return sampled output):

    forward(images)                — MSE signature [B, n_batt, n_phase]
    compute_signature(images)      — same as forward, with phase filter
    forward_full(images)           — recon + per-bank outputs (debug)

    compute_axis_codebook(...)     — projective-axis codebook per bank
    encode_axes(images, ...)       — axis activations per input

Empirical foundation for axis sampling
--------------------------------------
Every trained sphere-solver tested (19 models across D ∈ {3, 4, 5})
produces an M tensor whose rows, when antipodal pairs are collapsed,
form a uniformly-distributed codebook on ℝP^(D-1). The "axis codebook"
sampling exposes this structure as a discrete representation.

Reference: AbstractPhil/geolip-svae-implicit-solver-experiments

Example:
    from transformers import AutoModel
    model = AutoModel.from_pretrained("AbstractPhil/geolip-svae-h2-64")

    # MSE-based sampling (existing)
    sig = model(images)                          # [B, 64, 3]

    # Axis-based sampling (new)
    codebook = model.compute_axis_codebook(0, 'final', calib_images)
    activations = model.encode_axes(images, 0, 'final', codebook)
"""

import importlib
import math
from typing import Any, Dict, List, Optional, Tuple

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
    Each axis is sign-canonicalized so antipodally-equivalent axes get
    the same representation.

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

    # ── MSE sampling (existing) ──────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.compute_signature(images)

    @torch.no_grad()
    def compute_signature(
        self,
        images: torch.Tensor,
        phase: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute per-bank MSE signature.

        Returns [B, n_batteries, n_epoch_phases] or [B, n_batteries] if phase given.
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
    def compute_axis_codebook(
        self,
        battery_idx: int,
        phase: str,
        calibration_images: torch.Tensor,
        patch_idx: int = 0,
        threshold: float = -0.9,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Sample the projective-axis codebook from a trained bank.

        Runs calibration_images through the specified bank, averages
        the M tensors, identifies antipodal row pairs, collapses to
        axis representatives. Result is the bank's axis codebook on
        ℝP^(D-1).

        Args:
            battery_idx: which battery
            phase: which training phase ('epoch_1' / 'best' / 'final' typically)
            calibration_images: [N, C, H, W] inputs for averaging.
                Use 256-1024 samples for stable codebook.
                Gaussian noise inputs work as a noise-distribution-independent
                calibration probe.
            patch_idx: which patch's M to read (default 0)
            threshold: antipodal cosine threshold (default -0.9)
            batch_size: forward-pass batch size for calibration

        Returns:
            codebook: [n_axes, D] sphere-normed sign-canonicalized axes.
                n_axes varies per bank (typically 24-27 at V=32 D=4).
        """
        bank = self.bank(battery_idx, phase)
        bank.eval()

        device = next(bank.parameters()).device
        calibration_images = calibration_images.to(device)

        all_M = []
        N = calibration_images.shape[0]
        for start in range(0, N, batch_size):
            chunk = calibration_images[start:start + batch_size]
            out = bank(chunk)
            if not isinstance(out, dict) or 'svd' not in out:
                raise RuntimeError(
                    f"Bank {battery_idx}/{phase} forward must return dict "
                    f"with 'svd' key. compute_axis_codebook requires a "
                    f"sphere-solver battery."
                )
            all_M.append(out['svd']['M'][:, patch_idx].cpu())

        M_avg = torch.cat(all_M, dim=0).mean(dim=0)  # [V, D]
        pairs, unpaired = identify_antipodal_pairs(M_avg, threshold=threshold)
        return collapse_to_axes(M_avg, pairs, unpaired)

    @torch.no_grad()
    def encode_axes(
        self,
        images: torch.Tensor,
        battery_idx: int,
        phase: str,
        codebook: torch.Tensor,
        patch_idx: int = 0,
    ) -> torch.Tensor:
        """Encode images against a bank's axis codebook.

        Each row of the bank's M tensor gets cosine-matched against
        every codebook axis. Uses absolute cosine since antipodal axes
        represent the same line on ℝP^(D-1).

        Args:
            images: [B, C, H, W]
            battery_idx: which battery to read M from
            phase: which training phase
            codebook: [n_axes, D] from compute_axis_codebook
            patch_idx: which patch's M to read (default 0)

        Returns:
            activations: [B, V, n_axes] absolute cosines in [0, 1].
                Higher = stronger axis activation for that input row.
        """
        bank = self.bank(battery_idx, phase)
        bank.eval()

        device = next(bank.parameters()).device
        images = images.to(device)
        codebook = codebook.to(device).to(images.dtype)

        out = bank(images)
        M = out['svd']['M'][:, patch_idx]  # [B, V, D]

        norms = M.norm(dim=2, keepdim=True).clamp_min(1e-12)
        unit = M / norms

        # [B, V, D] @ [D, n_axes] → [B, V, n_axes]; abs for antipodal-equivalence
        return torch.einsum('bvd,nd->bvn', unit, codebook).abs()

    @torch.no_grad()
    def quantize_axes(
        self,
        images: torch.Tensor,
        battery_idx: int,
        phase: str,
        codebook: torch.Tensor,
        patch_idx: int = 0,
    ) -> torch.Tensor:
        """Hard quantization: each input row → strongest codebook axis index.

        Returns:
            codes: [B, V] integer indices into codebook
        """
        return self.encode_axes(
            images, battery_idx, phase, codebook, patch_idx
        ).argmax(dim=-1)
"""
geolip_svae.arrays.model
==========================
BatteryArrayModel — generic PreTrainedModel that dispatches to any
sphere-solver battery class declared in its config.

The array holds n_batteries × n_epoch_phases bank instances, all of the
same battery class with the same architecture kwargs. Banks differ only
in their loaded weights (which epoch of which training run).

Forward API:
    signature = model(images)
        images: [B, 3, H, W]
        returns: [B, n_batteries, n_epoch_phases]  MSE per battery per phase

forward_full(images):
    returns full per-bank output — memory-heavy, for debugging.

Example:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "AbstractPhil/geolip-svae-h2-64",
        trust_remote_code=True,   # optional if geolip-svae is pip-installed
    )
    sig = model(images)   # [B, 64, 3]
"""

import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from geolip_svae.arrays.config import BatteryArrayConfig


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

        # Dynamic dispatch: load the battery class
        battery_cls = self._resolve_battery_class()

        # Build n_banks instances, all same architecture
        self.banks = nn.ModuleList([
            battery_cls(**config.battery_kwargs)
            for _ in range(config.n_banks)
        ])

        self.post_init()

    def _resolve_battery_class(self):
        """Import and return the battery class declared in config."""
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
        """Fetch a bank by (battery, phase) coordinates."""
        return self.banks[self.config.bank_index(battery_idx, phase)]

    def bank_by_flat_idx(self, bank_idx: int) -> nn.Module:
        return self.banks[bank_idx]

    # ── Forward API ──────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Primary forward: MSE signature across all banks.

        Returns [B, n_batteries, n_epoch_phases].
        """
        return self.compute_signature(images)

    @torch.no_grad()
    def compute_signature(
        self,
        images: torch.Tensor,
        phase: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute per-bank MSE signature.

        Args:
            images: [B, 3, H, W]
            phase: if given, return only that phase's slice [B, n_batteries].
                    Otherwise return full [B, n_batteries, n_epoch_phases].
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
            # Battery forward returns dict with 'recon' key
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

        Returns dict with:
            'mse_signature': [B, n_batteries, n_epoch_phases]
            'recon_per_bank': [B, n_banks, 3, H, W]
            'bank_outputs': list of length n_banks, each the full output
                            dict from that bank's forward
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
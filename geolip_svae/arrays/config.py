"""
geolip_svae.arrays.config
==========================
BatteryArrayConfig — generic PretrainedConfig for any battery array.

A "battery array" is a bundle of N×K independently-trained sphere-solver
batteries, where N is the battery count and K is the number of training
phases (typically 3: epoch_1, best, final). The config captures:

  - Which battery class to instantiate (by name + module path)
  - What kwargs to pass the battery constructor
  - The array structure (n_batteries, epoch_phases)
  - Per-battery metadata (which noise types each trained on, MSE at each phase)

The config is architecture-agnostic: the same class serves h2-64, future
h3 arrays, frequency-triad arrays, etc. All that changes is the values.
"""

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class BatteryArrayConfig(PretrainedConfig):
    """Configuration for a battery array of any sphere-solver variant.

    The array is organized as n_batteries × n_epoch_phases banks.
    Bank index = battery_idx * n_epoch_phases + phase_idx.
    """

    model_type = "battery_array"

    def __init__(
        self,
        # ── Architecture — what each bank IS ─────────────────────────
        battery_class: str = "PatchSVAE",
        battery_module: str = "geolip_svae.model",
        battery_kwargs: Optional[Dict[str, Any]] = None,

        # ── Array structure ──────────────────────────────────────────
        n_batteries: int = 64,
        n_epoch_phases: int = 3,
        epoch_phase_names: Optional[List[str]] = None,

        # ── Per-battery metadata ─────────────────────────────────────
        batteries: Optional[List[Dict[str, Any]]] = None,

        # ── Provenance ───────────────────────────────────────────────
        source_repo: Optional[str] = None,
        built_at_utc: Optional[str] = None,
        array_spec_name: Optional[str] = None,

        **kwargs,
    ):
        super().__init__(**kwargs)

        # Architecture (enables dynamic bank construction at load time)
        self.battery_class = battery_class
        self.battery_module = battery_module
        self.battery_kwargs = battery_kwargs if battery_kwargs is not None else {}

        # Array shape
        self.n_batteries = n_batteries
        self.n_epoch_phases = n_epoch_phases
        self.epoch_phase_names = (
            epoch_phase_names
            if epoch_phase_names is not None
            else ['epoch_1', 'best', 'final']
        )

        # Battery metadata
        self.batteries = batteries if batteries is not None else []

        # Provenance
        self.source_repo = source_repo
        self.built_at_utc = built_at_utc
        self.array_spec_name = array_spec_name

    # ── Derived properties ───────────────────────────────────────────

    @property
    def n_banks(self) -> int:
        """Total bank count: n_batteries × n_epoch_phases."""
        return self.n_batteries * self.n_epoch_phases

    # ── Bank indexing helpers ────────────────────────────────────────

    def bank_index(self, battery_idx: int, phase: str) -> int:
        """Flat bank index for a (battery, phase) pair."""
        if phase not in self.epoch_phase_names:
            raise ValueError(
                f"phase must be one of {self.epoch_phase_names}, got '{phase}'"
            )
        return (battery_idx * self.n_epoch_phases
                + self.epoch_phase_names.index(phase))

    def battery_and_phase(self, bank_idx: int) -> Dict[str, Any]:
        """Inverse: given flat bank idx, return battery + phase metadata."""
        battery_idx = bank_idx // self.n_epoch_phases
        phase_idx = bank_idx % self.n_epoch_phases
        phase = self.epoch_phase_names[phase_idx]
        battery = (
            self.batteries[battery_idx]
            if battery_idx < len(self.batteries)
            else None
        )
        return {
            'bank_idx': bank_idx,
            'battery_idx': battery_idx,
            'phase_idx': phase_idx,
            'phase': phase,
            'battery': battery,
        }
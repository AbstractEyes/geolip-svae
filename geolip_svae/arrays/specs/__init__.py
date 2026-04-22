"""
geolip_svae.arrays.specs
==========================
Array specifications. Each spec describes a specific battery array:
    - which battery class the banks use
    - what architecture kwargs
    - how the training configs are organized (noise types per battery)
    - how to derive checkpoint paths on HF

Built-in specs:
    - h2_64  — 64-battery H2_linear_matched array (sphere solver v1)

To add a new spec, create a module next to this file with:
    NAME: str
    BATTERY_CLASS: str           # e.g., "PatchSVAE"
    BATTERY_MODULE: str          # e.g., "geolip_svae.model"
    BATTERY_KWARGS: dict
    N_BATTERIES: int
    EPOCH_PHASE_NAMES: List[str]
    def get_configs() -> List[Dict[str, Any]]
    def checkpoint_path(config: Dict, epoch: int) -> str
"""

from geolip_svae.arrays.specs import h2_64

SPECS = {
    "h2_64": h2_64,
}


def get_spec(name: str):
    """Resolve a spec module by name."""
    if name not in SPECS:
        raise ValueError(
            f"Unknown spec '{name}'. Known specs: {list(SPECS.keys())}"
        )
    return SPECS[name]
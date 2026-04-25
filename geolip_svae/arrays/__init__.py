"""
geolip_svae.arrays
====================
Battery array infrastructure — generic N×K bank bundles of sphere-solver
batteries with dynamic architecture dispatch.

Usage:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "AbstractPhil/geolip-svae-h2-64",
        trust_remote_code=False,   # geolip-svae must be pip-installed
    )
    # or:
    from geolip_svae.arrays import BatteryArrayModel, BatteryArrayConfig
    config = BatteryArrayConfig.from_pretrained("AbstractPhil/geolip-svae-h2-64")
    model = BatteryArrayModel.from_pretrained("AbstractPhil/geolip-svae-h2-64")

Build a new array from an HF training repo:
    from geolip_svae.arrays import build_array
    build_array(
        source_repo="AbstractPhil/geolip-svae-h2-64",
        spec_name="h2_64",
    )

Read a trained bank's projective-axis codebook (empirically validated
across 19 sphere-solver models, D ∈ {3, 4, 5}):
    from geolip_svae.arrays import ProjectiveReader
    reader = ProjectiveReader.from_bank(model.bank(0, 'final'), calib_inputs)
    activations = reader.encode(new_M)   # [B, V, n_axes]
    codes = reader.quantize(new_M)        # [B, V] discrete axis indices
"""

from geolip_svae.arrays.config import BatteryArrayConfig
from geolip_svae.arrays.model import BatteryArrayModel
from geolip_svae.arrays.builder import build_array
from geolip_svae.arrays.projective_reader import (
    ProjectiveReader,
    ProjectiveProbeMetrics,
    identify_antipodal_pairs,
    collapse_to_axes,
    projective_pairwise_angles,
    uniform_rp_baseline,
)

# Register with HF Auto* so the model loads without trust_remote_code
# when geolip-svae is pip-installed.
try:
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("battery_array", BatteryArrayConfig)
    AutoModel.register(BatteryArrayConfig, BatteryArrayModel)
except (ImportError, ValueError):
    # ValueError: already registered (re-imports during development)
    # ImportError: transformers not available (user loading just the data)
    pass

__all__ = [
    # Battery array (existing)
    "BatteryArrayConfig",
    "BatteryArrayModel",
    "build_array",
    # Projective reader (new — see projective_reader.py)
    "ProjectiveReader",
    "ProjectiveProbeMetrics",
    "identify_antipodal_pairs",
    "collapse_to_axes",
    "projective_pairwise_angles",
    "uniform_rp_baseline",
]
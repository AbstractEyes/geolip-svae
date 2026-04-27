"""
geolip_svae.experimental
========================
Experimental sphere-solver variants and codebook tooling that emerged
from research probes but are not part of the canonical inference path.

These modules are preserved because they encode useful potentials for
future work. They are NOT the path public users should reach for first
— see ``geolip_svae.inference`` for the production inference framework
and the canonical projective-axis codebook implementation.

Modules
-------
spectral_cell
    Single-cell spectral sphere-solver variant. Standalone module that
    can be instantiated independently of the BatteryArrayModel hierarchy.

spectral_battery
    Reusable spectral-battery class. Container form of ``spectral_cell``.

experimental_codebook
    Earlier codebook implementation (formerly ``spectral_codebook.py``)
    that pre-dates the projective-axis discovery (scratchpad entry
    000101). Kept for reference and as a comparator to the canonical
    ``geolip_svae.inference.codebook.Codebook``. The two are NOT
    interchangeable: the experimental codebook does not perform
    antipodal-pair collapse and reports different geometric statistics.

Stability
---------
These modules are explicitly experimental. Their public surface MAY
change without warning. The canonical inference path
(``geolip_svae.inference``) will not.
"""

from geolip_svae.experimental import spectral_cell
from geolip_svae.experimental import spectral_battery
from geolip_svae.experimental import experimental_codebook

__all__ = [
    'spectral_cell',
    'spectral_battery',
    'experimental_codebook',
]
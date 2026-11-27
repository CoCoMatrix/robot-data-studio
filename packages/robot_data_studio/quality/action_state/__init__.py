"""Action-state consistency operator: per-episode, per-channel action supervision checks.

CPU only and read-only on source data. See README.md in this directory. The CLI modules
(run_operator, profiles, calibrate, validate_real) are run with ``python -m`` and are not
imported here.
"""

from .action_state import analyze, usable_mask
from .contract import VERSION, validate_profile
from .probe import pairing, probe

__all__ = ["VERSION", "analyze", "pairing", "probe", "usable_mask", "validate_profile"]

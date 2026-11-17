"""phasecond: detection of phase ill-conditioning of autonomous initial value problems.

    from phasecond import phase_report, scalar_report, quick_check, orbit_test, shift_estimate
"""
from .core import phase_report, PhaseReport, error_scale, pi_star_from_run
from .scalar import scalar_report, ScalarReport
from .posteriori import orbit_test, OrbitTest, shift_estimate, quick_check

__version__ = "0.1.0"
__all__ = ["phase_report", "PhaseReport", "error_scale", "pi_star_from_run", "scalar_report",
           "ScalarReport", "orbit_test", "OrbitTest", "shift_estimate", "quick_check"]

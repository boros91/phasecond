"""Tests against values saved with the manuscript (results/index_scalar_cases.csv, sir_cases.csv,
ml_cases.csv, h2_cases.csv of the phase ill-conditioning paper; SciPy scale, atol/rtol = 1e-2).

    pytest -q                 fast tests (about one minute)
    pytest -q --runslow       also Morris-Lecar and hydrogen-air (needs Cantera; about 3 minutes)
"""
import os, sys
import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import problems as P
from phasecond import (phase_report, scalar_report, orbit_test, shift_estimate, quick_check)

R, A = 1e-4, 1e-6          # atol/rtol = 1e-2

SCALAR = [  # f, x0, xe, Theta_e, Pi, Pi_relative, T_event
    (P.flame(2), 1e-2, 0.5, 4.08, 49.51475539710834, 25.252525252525253, 102.5951198501346),
    (P.flame(2), 1e-4, 0.5, 4.08, 247573.77698554166, 2500.2500250025, 10007.210240366976),
    (P.flame(2), 1e-6, 0.5, 4.08, 2451227941.42402, 250000.25000024997, 1000011.815509558),
    (P.flame(3), 1e-2, 0.5, 8.16, 2475.737769855417, 1262.6262626262626, 5100.595119850134),
    (P.flame(3), 1e-3, 0.5, 8.16, 1349388.604290565, 125125.12512512514, 501002.90675477864),
    (P.ghost(1e-4), -1.0, 1.0, 1.0098990100989902, 119.52741853096789, 50.004999999999995, 312.15933202164626),
    (P.ghost(1e-8), -1.0, 1.0, 1.0099999899, 990123.7715161835, 5000.00005, 31413.926535904597),
    (P.logistic, 1e-1, 0.5, 2.04, 1.0, 1.0, 2.197224577336219),
    (P.logistic, 1e-5, 0.5, 2.04, 490.69118142161807, 1.0, 11.512915464920228),
    (P.logistic, 1e-7, 0.5, 2.04, 49020.10294122598, 1.0, 16.118095550958316),
]


@pytest.mark.parametrize("f,x0,xe,Th,Pi,Pirel,Te", SCALAR)
def test_scalar_report(f, x0, xe, Th, Pi, Pirel, Te):
    r = scalar_report(f, x0, xe, R, A)
    assert r.Theta_e == pytest.approx(Th, rel=1e-9)
    assert r.Pi == pytest.approx(Pi, rel=2e-4)
    assert r.Pi_rel == pytest.approx(Pirel, rel=2e-4)
    assert r.T_e == pytest.approx(Te, rel=1e-8)


@pytest.mark.parametrize("f,x0,xe,Th,Pi,Pirel,Te", [SCALAR[1], SCALAR[5], SCALAR[8]])
def test_system_code_on_scalar(f, x0, xe, Th, Pi, Pirel, Te):
    r = phase_report(f, [x0], lambda x: x[0] - xe, 3 * Te, R, A)
    assert r.T_e == pytest.approx(Te, rel=1e-7)
    assert r.Pi == pytest.approx(Pi, rel=1e-4)
    assert r.Pi_star == pytest.approx(Pi, rel=1e-4)      # equality for scalar equations
    assert r.adjoint_check < 1e-4


def test_finite_tolerance():
    # exponential growth from below atol/rtol: exact shift ln(1 + r s/x0) instead of r s/x0
    r = scalar_report(P.logistic, 1e-7, 0.5, R, A)
    d = R * (1e-7 + A / R)
    assert R * r.Pi_early * r.Theta_e == pytest.approx(np.log((1e-7 + d) / 1e-7) - np.log((1 - 1e-7 - d) / (1 - 1e-7)), rel=1e-6)
    assert not np.isfinite(r.Pi_late)                    # an admissible error reaches x = 0
    r = scalar_report(P.logistic, 1e-5, 0.5, R, A)
    assert np.isfinite(r.Pi_late) and r.Pi_late == pytest.approx(r.Pi, rel=0.1)
    assert not np.isfinite(scalar_report(P.flame(2), 1e-6, 0.5, R, A).Pi_late)
    assert np.isfinite(scalar_report(P.flame(2), 1e-4, 0.5, R, A).Pi_late)


SIR = [  # I0, Te, Theta_e, Pi, Pi_relative, Pi_star
    (1e-2, 26.632580022313157, 11.300822948753147, 3.207221212011773, 2.782890338787038, 1.2422667146408473),
    (1e-4, 49.897044181070186, 11.425557790077045, 49.65322634069042, 5.924827479568168, 44.20578589194622),
    (1e-6, 72.92699786350511, 11.426812728522174, 4384.663698288757, 9.038594745205659, 4376.117487494663),
    (1e-8, 95.95290691900718, 11.426825278671442, 437578.9435276416, 12.151884424409879, 437567.2991458761),
]


@pytest.mark.parametrize("I0,Te,Th,Pi,Pirel,Pistar", SIR)
@pytest.mark.parametrize("jac", [P.sir_jac, None])
def test_sir(I0, Te, Th, Pi, Pirel, Pistar, jac):
    r = phase_report(P.sir, [1 - I0, I0], P.sir_event, 1000, R, A, jac=jac)
    assert r.T_e == pytest.approx(Te, rel=1e-8)
    assert r.Theta_e == pytest.approx(Th, rel=1e-6)
    assert r.Pi == pytest.approx(Pi, rel=1e-5)
    assert r.Pi_rel == pytest.approx(Pirel, rel=1e-5)
    assert r.Pi_star == pytest.approx(Pistar, rel=1e-3)
    assert r.adjoint_check < 1e-5


def test_sir_matlab_scale_and_scan():
    r = phase_report(P.sir, [1 - 1e-7, 1e-7], P.sir_event, 1000, R, A, scale="matlab", jac=P.sir_jac)
    assert r.rtolPi == pytest.approx(4.51, abs=0.01)                 # Sect. 7.2 of the paper
    assert r.pi_for(atol=1e-12) == pytest.approx(10.6, abs=0.05)     # Sect. 9
    assert r.uncontrolled == [1]
    assert r.level == 2


def test_transverse_mode():                  # Example 3.4: Pi/Pi_* = 1.04, 1.25, 1.45
    for e, ratio in ((1e-1, 1.04), (1e-2, 1.25), (1e-3, 1.45)):
        r = phase_report(P.coupled(), [e, 0.0], lambda u: u[0] - 0.5, 10 / e, R, A, ref_atol=1e-14)
        assert r.Pi / r.Pi_star == pytest.approx(ratio, abs=0.006)


def test_production_run_as_reference():
    """Indicators from a run with the production tolerances: Pi_* to 1%, Pi within a factor 2.5."""
    for f, x0, g, tmax in ((P.flame(2), [1e-4], lambda y: y[0] - 0.5, 2e4),
                           (P.ghost(1e-6), [-1.0], lambda y: y[0] - 1.0, 1e5),
                           (P.sir, [1 - 1e-6, 1e-6], P.sir_event, 1000)):
        a = phase_report(f, x0, g, tmax, R, A)
        for m in ("BDF", "LSODA"):
            b = phase_report(f, x0, g, tmax, R, A, ref_rtol=R, ref_atol=A, method=m)
            assert abs(b.T_e - a.T_e) > 100 * R * a.T_e * 0 + 0.5      # the run itself is phase-wrong
            assert b.Pi_star == pytest.approx(a.Pi_star, rel=1e-2)
            assert 1 / 2.5 < b.Pi / a.Pi < 2.5


def test_quick_check_and_shift_estimate():
    f, x0, g = P.sir, [1 - 1e-6, 1e-6], P.sir_event
    rep = phase_report(f, x0, g, 1000, R, A, jac=P.sir_jac)
    ev = lambda t, x: g(x); ev.terminal = True
    for m, tol in (("LSODA", 1e-6), ("BDF", 1e-6), ("Radau", 1e-4), ("RK45", 1e-6)):
        s = solve_ivp(lambda t, x: f(x), (0, 1000), x0, method=m, rtol=tol, atol=tol / 100, events=ev)
        D = s.t_events[0][0] - rep.T_e
        Dhat, Asum = shift_estimate(f, s.t, s.y, rep)
        assert Dhat == pytest.approx(D, rel=2e-2)
        assert abs(Dhat) <= Asum * (1 + 1e-12)
    s = solve_ivp(lambda t, x: f(x), (0, 1000), x0, method="LSODA", rtol=R, atol=A, events=ev)
    ps, msg = quick_check(f, s, g, R, A)
    assert ps == pytest.approx(4376.1, rel=2e-2)


def test_orbit_test():
    r = orbit_test(P.flame(2), [1e-4], (0, 2e4), 1e-4, method="LSODA")
    assert r.E_time == pytest.approx(0.9908286489986204, rel=1e-6)   # results/orbit_test.csv
    assert r.E_orbit == pytest.approx(2.8433374066327843e-06, rel=1e-4)
    assert r.verdict.startswith("PHASE ERROR")
    lor = lambda u: np.array([10 * (u[1] - u[0]), u[0] * (28 - u[2]) - u[1], u[0] * u[1] - 8 / 3 * u[2]])
    assert orbit_test(lor, [1.0, 1.0, 20.0], (0, 25), 1e-4, method="LSODA").verdict == "not phase-dominated"
    assert orbit_test(P.logistic, [0.1], (0, 40), 1e-4, method="LSODA").verdict == "difference at the tolerance level"


ML = [(1e-1, 582.1239975214866, 1.0235373599613655, 6401.299027717757, 3072.1270293162875, 6049.65197298046),
      (1e-2, 1789.1008698243377, 1.0243295022498027, 63999.874257690484, 30658.80301807841, 60477.01025698228)]


@pytest.mark.slow
@pytest.mark.parametrize("Delta,period,Th,Pi,Pistar,Pirel", ML)
def test_morris_lecar(Delta, period, Th, Pi, Pistar, Pirel):
    f = P.morris_lecar(Delta)
    r = phase_report(f, P.ml_cycle_point(f), lambda x: x[0] - 10.0, 20000, R, A, direction=1)
    assert r.T_e == pytest.approx(period, rel=1e-8)
    assert r.Theta_e == pytest.approx(Th, rel=1e-6)
    assert r.Pi == pytest.approx(Pi, rel=1e-4)
    assert r.Pi_star == pytest.approx(Pistar, rel=1e-3)
    assert r.Pi_rel == pytest.approx(Pirel, rel=1e-4)
    assert r.x_max[0] == pytest.approx(-29.4, abs=0.05)             # the ghost of the saddle-node


@pytest.mark.slow
def test_hydrogen():
    pytest.importorskip("cantera")
    import h2_problem as H
    r = phase_report(H.rhs, H.x0, H.event, 1e-3, 1e-3, 1e-6, names=H.names, ref_atol=1e-24)
    assert r.T_e == pytest.approx(3.1113776213198617e-4, rel=1e-8)
    assert r.Theta_e == pytest.approx(1.9798e-05, rel=1e-3)
    for ratio, Pi in ((1e-2, 380308513.3991367), (1e-6, 38410.21183258366), (1e-10, 382.9298923536942)):
        assert r.pi_for(rtol=1.0, atol=ratio) == pytest.approx(Pi, rel=1e-4)
    assert r.Pi_rel == pytest.approx(379.1267838461042, rel=1e-4)
    assert r.names[int(np.argmax(r.share_max))] == "H"


# ---------------------------------------------------------------- version 0.1.1: finite-tolerance edge cases
ONE = lambda x: np.ones_like(x)
ZERO = lambda x: np.zeros_like(x)


def test_constant_field_forward_clipping():
    # x' = 1, s = x, rtol = 0.01: advance min(r x, x_e - x), largest at x = x_e/(1 + r); delay r x_e
    r = 0.01
    for xe, early in ((2.0, 2 * r / (1 + r)), (1.001, 1e-3), (1.000001, 1e-6)):
        o = scalar_report(ONE, 1.0, xe, rtol=r, atol=0.0, fprime=ZERO)
        assert r * o.Theta_e * o.Pi_early == pytest.approx(early, rel=1e-6)
        assert r * o.Theta_e * o.Pi_late == pytest.approx(r * xe, rel=1e-6)
        assert o.late_status == "ok" and np.isnan(o.x_lost)


def test_constant_field_decreasing():
    # x' = -1 from 2 to 1.995: the advance is clipped at the event (x - 1.995 <= 0.005 < r x)
    r = 0.01
    o = scalar_report(lambda x: -np.ones_like(x), 2.0, 1.995, rtol=r, atol=0.0, fprime=ZERO)
    assert r * o.Theta_e * o.Pi_early == pytest.approx(0.005, rel=1e-6)
    assert r * o.Theta_e * o.Pi_late == pytest.approx(0.02, rel=1e-6)


def test_loss_at_even_order_zero():
    # y' = y^2 (1 - y) from below atol: a late defect crosses y = 0, where f has a double zero
    o = scalar_report(P.flame(2), 1e-7, 0.5, R, A)
    assert not np.isfinite(o.Pi_late) and o.late_status == "equilibrium"
    # a known equilibrium passed by the user
    o = scalar_report(lambda x: (x - 0.3) ** 2 * (1.5 - x) + 0.0, 0.3 + 1e-7, 0.5, R, A, equilibria=[0.3])
    assert not np.isfinite(o.Pi_late)


def test_finite_near_equilibrium():
    # f = c + x^2 > 0 everywhere: the late shift through x = 0 is large but finite
    for c in (1e-12, 1e-30):
        f = lambda x, c=c: c + x * x
        o = scalar_report(f, 1e-3, 2e-3, 1e-2, 2e-3)
        xs = np.linspace(1e-3, 2e-3, 20001); y = xs - 1e-2 * (xs + 0.2); q = np.sqrt(c)
        exact = ((np.arctan(xs / q) - np.arctan(y / q)) / q).max()
        assert np.isfinite(o.Pi_late) and o.late_status == "ok"
        assert 1e-2 * o.Theta_e * o.Pi_late == pytest.approx(exact, rel=1e-6)


def test_tolerance_advice_does_not_reward_a_coarser_event():
    # x' = -1 from 100 to 1, s = x + a/r: Pi(a) = (100 + a/r)/(1 + a/r) falls only because Theta(T_e) grows
    o = phase_report(lambda x: np.array([-1.0]), [100.0], lambda x: x[0] - 1.0, 101, 1e-4, 1e-8,
                     jac=lambda x: np.zeros((1, 1)), direction=-1)
    assert np.isnan(o.atol_safe)
    assert o.atol_floor <= 1e-4


def test_sign_flag_sir():
    o = phase_report(P.sir, [1 - 1e-8, 1e-8], P.sir_event, 1000, R, A, jac=P.sir_jac, names=["S", "I"])
    assert o.sign_unprotected == [1] and o.uncontrolled == [1]
    o = phase_report(P.sir, [1 - 1e-2, 1e-2], P.sir_event, 1000, R, A, jac=P.sir_jac, names=["S", "I"])
    assert o.sign_unprotected == []

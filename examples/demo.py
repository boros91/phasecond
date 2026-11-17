"""Reports for five problems of the paper.  Run:  python demo.py"""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import problems as P
from phasecond import phase_report, scalar_report, quick_check, orbit_test

R, A = 1e-4, 1e-6
line = lambda s: print("\n" + "=" * 100 + "\n" + s)

line("flame problem y' = y^2 - y^3, y(0) = 1e-4, event y = 1/2 (scalar report)")
print(scalar_report(P.flame(2), 1e-4, 0.5, R, A))

line("logistic equation from x0 = 1e-7, below the absolute tolerance (scalar report)")
print(scalar_report(P.logistic, 1e-7, 0.5, R, A))

line("SIR, I0 = 1e-8: check of a run just made, then the full report")
ev = lambda t, u: P.sir_event(u); ev.terminal = True
s = solve_ivp(lambda t, u: P.sir(u), (0, 1000), [1 - 1e-8, 1e-8], method="RK45", rtol=R, atol=A, events=ev)
print("quick_check of an RK45 run:", quick_check(P.sir, s, P.sir_event, R, A)[1])
print(phase_report(P.sir, [1 - 1e-8, 1e-8], P.sir_event, 1000, R, A, jac=P.sir_jac, names=["S", "I"]))

line("Morris-Lecar neuron near its SNIC bifurcation, Delta = 1e-2, event: next spike")
f = P.morris_lecar(1e-2)
print(phase_report(f, P.ml_cycle_point(f), lambda x: x[0] - 10.0, 20000, R, A, direction=1, names=["V", "w"]))

line("orbit test (two LSODA runs, tol and tol/10) on the flame problem")
print(orbit_test(P.flame(2), [1e-4], (0, 2e4), R, method="LSODA"))

try:
    import h2_problem as H
    line("hydrogen-air ignition with SciPy's default tolerances (rtol 1e-3, atol 1e-6)")
    print(phase_report(H.rhs, H.x0, H.event, 1e-3, 1e-3, 1e-6, names=H.names, ref_atol=1e-24))
except ImportError:
    print("\n(Cantera not installed: hydrogen example skipped)")

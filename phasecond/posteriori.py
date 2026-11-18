"""Checks that use computed solutions.

orbit_test       black box: two runs (tol and tol/10) compared at equal times and as orbits.
                 Needs neither a reference solution nor an adjoint.
shift_estimate   signed first-order estimate of the event-time error of a given run,
                 D_hat = - sum_k lambda(t_{k+1}) . d_k, with the true defects d_k of its steps.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["orbit_test", "OrbitTest", "shift_estimate", "quick_check"]


@dataclass
class OrbitTest:
    E_time: float       # largest difference of the two runs at equal times (scaled max norm)
    E_orbit: float      # largest distance of the first run from the orbit of the second
    tol: float
    shift: np.ndarray   # relative phase sigma(t) - t of the two runs on the comparison grid
    t: np.ndarray
    shift_fast: float   # relative phase where the first run moves fastest (at the event)

    @property
    def ratio(self):
        return self.E_orbit / self.E_time if self.E_time > 0 else np.nan

    @property
    def verdict(self):
        if self.E_time <= 10 * self.tol:
            return "difference at the tolerance level"
        if self.E_orbit < 0.1 * self.E_time:
            return "PHASE ERROR: the two runs follow the same orbit at different times"
        return "not phase-dominated"

    def __str__(self):
        return (f"E_time = {self.E_time:.2e}, E_orbit = {self.E_orbit:.2e}, ratio = {self.ratio:.1e}, "
                f"time shift between the runs at the fastest point = {self.shift_fast:+.3g}\n--> {self.verdict}")


def _track(P, Q, tq, W=300):
    """Follow the polygon Q continuously: for each point of P (in time order) the nearest point of
    Q is sought only on the W segments around the previous match."""
    nseg = len(Q) - 1
    A, AB = Q[:-1], Q[1:] - Q[:-1]
    L2 = np.maximum((AB * AB).sum(1), 1e-300)
    j = 0
    dist = np.empty(len(P)); tm = np.empty(len(P))
    for i in range(len(P)):
        lo, hi = max(j - W, 0), min(j + W, nseg - 1)
        a_, ab = A[lo:hi + 1], AB[lo:hi + 1]
        lam = np.clip(((P[i] - a_) * ab).sum(1) / L2[lo:hi + 1], 0, 1)
        dd = np.abs(P[i] - (a_ + lam[:, None] * ab)).max(1)
        kk = int(np.argmin(dd))
        j = lo + kk
        dist[i] = dd[kk]
        tm[i] = tq[j] + lam[kk] * (tq[j + 1] - tq[j])
    return dist, tm


def orbit_test(f, x0, t_span, rtol=1e-4, atol=None, method="LSODA", jac=None, factor=10.0, npts=50001):
    """Solve with (rtol, atol) and with both divided by `factor`; compare at equal times and as
    orbits.  E_orbit << E_time with E_time >> rtol means that the two runs differ by a time shift.
    The test is a diagnostic: two runs with the same shift pass it unnoticed."""
    x0 = np.atleast_1d(np.asarray(x0, float))
    atol = rtol / 100 if atol is None else atol
    F = lambda t, x: np.atleast_1d(f(x))
    sols = []
    for c in (1.0, factor):
        kw = dict(rtol=rtol / c, atol=np.asarray(atol) / c, dense_output=True)
        if jac is not None and method in ("Radau", "BDF", "LSODA"):
            kw["jac"] = lambda t, x: np.atleast_2d(jac(x))
        s = solve_ivp(F, t_span, x0, method=method, **kw)
        if not s.success:
            raise RuntimeError("solver failure: " + s.message)
        sols.append(s)
    tg = np.linspace(t_span[0], t_span[1], npts)
    U1, U2 = sols[0].sol(tg).T, sols[1].sol(tg).T
    sc = np.maximum(np.abs(U2).max(0), 1e-300)
    E_time = float(np.abs(U1 / sc - U2 / sc).max())
    dist, tm = _track(U1 / sc, U2 / sc, tg)
    speed = np.abs(np.diff(U1 / sc, axis=0)).max(1)
    kf = int(np.argmax(speed))
    return OrbitTest(E_time=E_time, E_orbit=float(dist.max()), tol=rtol, shift=tm - tg, t=tg,
                     shift_fast=float(tm[kf] - tg[kf]))


def shift_estimate(f, t, X, report, tight=1e-12):
    """First-order estimate of the event-time error of a computed solution.

    t, X : accepted steps of the run, X[:, k] = state at t[k] (as returned by solve_ivp)
    report : PhaseReport of the same problem (provides the adjoint lambda)

    Each step is repeated from X[:, k] with a tight tolerance to obtain its true defect d_k;
    the estimate is  D_hat = - sum lambda(t_{k+1}) . d_k  over the steps before the event,
    with lambda continued beyond T_e.  Returns (D_hat, A) where A = sum |lambda . d_k| measures
    how much cancellation there is (|D_hat| = A: none).
    """
    F = lambda s, x: np.atleast_1d(f(x))
    t = np.asarray(t, float); X = np.asarray(X, float)
    lam = lambda s: report.adj.sol(min(max(s, 0.0), report.T_e))
    D = 0.0; A = 0.0
    for k in range(t.size - 1):
        if t[k] >= report.T_e + abs(D) + 1e-12 * report.T_e:   # past the (shifted) event
            break
        h = t[k + 1] - t[k]
        s = solve_ivp(F, (0.0, h), X[:, k], method="DOP853", rtol=tight,
                      atol=tight * 1e-10 * max(np.abs(X[:, k]).max(), 1e-300))
        d = X[:, k + 1] - s.y[:, -1]
        c = -float(lam(t[k + 1]) @ d)
        D += c; A += abs(c)
    return D, A


def quick_check(f, sol, g, rtol, atol, scale="scipy", target=1e-3):
    """Cheapest test, for a run that has just been made: the adjoint-free bound Pi_* from the
    accepted steps of `sol` (a solve_ivp result, or a pair (t, X)) up to the first step past the
    event g = 0.  Costs one evaluation of f per step.  Returns (Pi_star, message).

    On the exact orbit Pi_* <= Pi, so a large value flags possible ill-conditioning (computed from an
    approximate orbit and sampling grid it is an estimate, not a certificate: near a decision threshold,
    verify with phase_report and a tighter reference); a small value does not exclude it
    (errors transverse to the flow are not seen), and phase_report should then be used."""
    from .core import pi_star_from_run
    X = sol.y if hasattr(sol, "y") else np.asarray(sol[1], float)
    gv = np.array([g(X[:, k]) for k in range(X.shape[1])])
    s0 = np.sign(gv[0] if gv[0] != 0 else gv[1])
    cross = np.where(gv[1:] * s0 <= 1e-12 * np.abs(gv).max())[0]     # first point at or past the event
    if cross.size == 0:
        return np.nan, "the run does not reach the event"
    X = X[:, :cross[0] + 2]
    ps = pi_star_from_run(f, X, g, rtol, atol, scale)
    if rtol * ps >= 1:
        msg = (f"rtol*Pi_* = {rtol*ps:.3g} >= 1: the event time of this run is not reliable; "
               "one accepted step can move the event by its width")
    elif rtol * ps >= target:
        msg = f"rtol*Pi_* = {rtol*ps:.3g}: grey zone; run phase_report"
    else:
        msg = (f"rtol*Pi_* = {rtol*ps:.3g}: no slow passage in the error scale along the flow; "
               "transverse sensitivity is not excluded (phase_report)")
    return ps, msg

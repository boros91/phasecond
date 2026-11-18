"""Scalar autonomous equations x' = f(x): everything by quadrature, and exactly.

For a scalar equation the event-time adjoint is 1/f, a defect d at the state x moves the event by
exactly  int_x^{x+d} dz/f(z),  and the indicators need no differential equation.  This module
also gives the finite-tolerance versions of Pi: the exact shift caused by one admissible defect
+- rtol*s(x), which the first-order number Pi overestimates when rtol*s(x) is not small compared
with x, and which is infinite when an admissible defect can carry the solution onto an
equilibrium (the event can then be lost).

A defect in the direction of motion that would carry the solution past the event is clipped at
the event: the event is then reached at once, and the advance is the remaining time T(x_e) - T(x).
The suprema over the route are estimated on a grid with local refinement, not certified.
A late shift is reported as infinite only when an equilibrium between the perturbed and the
original state is demonstrated: a sign change of f on a sampling grid, f = 0 at a sampled point or
at x = 0, an equilibrium given by the user (equilibria=...), or contributions of 1/f that keep
growing on shells around an interior minimum of |f|.  Otherwise it is finite; if quadrature
cannot resolve it, the value is kept and flagged as unresolved (a zero of even order away from
x = 0 is found only if the user passes it in equilibria).
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from .core import error_scale, TINY

__all__ = ["scalar_report", "ScalarReport"]


@dataclass
class ScalarReport:
    T_e: float
    Theta_e: float
    Pi: float
    x_max: float
    Pi_rel: float
    Lambda: float
    alpha: float
    rtol: float
    Pi_early: float      # finite-tolerance Pi, defect in the direction of motion (event early)
    Pi_late: float       # finite-tolerance Pi, defect against the motion (event late); inf: event can be lost
    x_lost: float        # first state from which an admissible defect reaches an equilibrium (nan if none)
    late_status: str = "ok"   # "ok", "equilibrium" (demonstrated), "unresolved" (quadrature), "undefined" (f not finite)
    x_unresolved: float = float("nan")   # a state at which the late shift was not resolved

    @property
    def rtolPi(self):
        return self.rtol * self.Pi

    def __str__(self):
        L = [f"event time T_e   = {self.T_e:.6g}",
             f"Theta(T_e)       = {self.Theta_e:.3g}",
             f"Pi               = {self.Pi:.3g}  at x = {self.x_max:.3g}   (relative control only: {self.Pi_rel:.3g})",
             f"rtol * Pi        = {self.rtolPi:.3g}",
             f"finite tolerance : rtol*Pi_early = {self.rtol*self.Pi_early:.3g},  rtol*Pi_late = {self.rtol*self.Pi_late:.3g}",
             f"Lambda_alpha (alpha={self.alpha:g}) = {self.Lambda:.3g};  rtol^alpha * Lambda = {self.rtol**self.alpha*self.Lambda:.3g}"]
        if not np.isfinite(self.Pi_late):
            L.append(f"--> THE EVENT CAN BE LOST: at x = {self.x_lost:.3g} an admissible error "
                     "reaches an equilibrium of f")
        elif self.rtol * max(self.Pi_early, self.Pi_late) >= 1:
            L.append("--> PHASE ILL-CONDITIONED at this tolerance: one admissible error can move the event "
                     "by its time resolution")
        elif self.rtolPi >= 1e-3:
            L.append("--> grey zone")
        else:
            L.append("--> phase well-conditioned at this tolerance")
        if self.late_status == "unresolved":
            L.append(f"    NOTE: the late shift from x = {self.x_unresolved:.3g} was not resolved by quadrature "
                     "(f is very small there, but no equilibrium was found); the value shown is an estimate")
        elif self.late_status == "undefined":
            L.append(f"    NOTE: f is not finite on the way back from x = {self.x_unresolved:.3g}; "
                     "the late shift there could not be computed")
        if self.Theta_e > self.T_e:
            L.append(f"    WARNING: Theta(T_e) = {self.Theta_e:.3g} exceeds T_e = {self.T_e:.3g}: the event level is "
                     "below the error scale, its time resolution is coarser than the time to reach it, and a small "
                     "Pi does not mean an accurate event time")
        return "\n".join(L)


def _grid(x0, xe, f, n=4000):
    """Points between x0 and xe, dense where |f| is small (where the solution is slow)."""
    a, b = (x0, xe) if x0 < xe else (xe, x0)
    g = [np.linspace(a, b, n)]
    for c in (a, b):                       # geometric refinement towards both ends
        w = b - a
        d = np.geomspace(1e-14 * max(w, abs(c)) + TINY, w, n)
        g.append(c + d if c == a else c - d)
    if a < 0 < b:                          # and towards zero, where the relative scale is small
        d = np.geomspace(1e-16 * max(abs(a), b), max(abs(a), b), n)
        g += [d, -d]
    x = np.unique(np.concatenate(g))
    x = x[(x >= a) & (x <= b)]
    for _ in range(3):                     # refine around the slowest point
        k = int(np.argmin(np.abs(f(x))))
        lo, hi = x[max(k - 2, 0)], x[min(k + 2, x.size - 1)]
        x = np.unique(np.concatenate([x, np.linspace(lo, hi, 400)]))
    return x


def scalar_report(f, x0, xe, rtol=1e-4, atol=1e-6, scale="scipy", alpha=1.0, fprime=None, equilibria=None):
    """Indicators for the scalar problem x' = f(x), x(0) = x0, event x = xe.

    f must not vanish between x0 and xe (the event is reached).  f and fprime must accept arrays.
    equilibria: optional list of known zeros of f; a late defect that reaches one of them loses the event.
    """
    sgn = 1.0 if xe > x0 else -1.0
    x = _grid(x0, xe, f)
    fx = f(x)
    if np.any(fx * sgn <= 0):
        raise ValueError("f does not keep the sign of xe - x0 on the way: the event is not reached")
    s = error_scale(x, rtol, atol, scale)
    Theta = s / np.abs(fx)
    Theta_e = float(error_scale(np.array([xe]), rtol, atol, scale)[0] / abs(f(np.array([xe]))[0]))
    k = int(np.argmax(Theta))
    Pi = float(Theta[k] / Theta_e)
    th_rel = np.abs(x) / np.abs(fx)
    Pi_rel = float(th_rel.max() / (abs(xe) / abs(f(np.array([xe]))[0])))

    inv = lambda z: 1.0 / f(np.array([z]))[0]
    T_e = abs(_integral(inv, x))

    if fprime is None:
        fprime = lambda z: (f(z * (1 + 1e-7) + 1e-300) - f(z * (1 - 1e-7) - 1e-300)) / (2e-7 * np.abs(z) + 2e-300)
    sigma = (s * np.abs(fprime(x)) / np.abs(fx)) ** alpha
    integrand = sigma / np.abs(fx)
    Lambda = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x)) / Theta_e)

    # exact shift by one admissible defect of size rtol*s(x), in both directions
    eq = np.asarray([] if equilibria is None else equilibria, float)
    s_at = lambda z: float(error_scale(np.array([z]), rtol, atol, scale)[0])

    def early_at(z):          # advance: defect in the direction of motion, clipped at the event
        y = z + sgn * rtol * s_at(z)
        if sgn * (y - xe) >= 0:
            y = xe
        return abs(_quad(inv, z, y))

    def late_at(z):           # delay: defect against the motion; (value, status)
        return _late_shift(f, inv, z, z - sgn * rtol * s_at(z), sgn, eq)

    sub = np.unique(np.concatenate([np.linspace(0, x.size - 1, 500).astype(int),
                                    np.arange(max(k - 5, 0), min(k + 6, x.size))]))
    order = sub[np.argsort(sgn * x[sub])]  # along the motion
    early = np.array([early_at(x[i]) for i in order])
    late = np.empty(order.size); status = []
    for j, i in enumerate(order):
        late[j], st = late_at(x[i])
        status.append(st)
    status = np.array(status)
    x_lost = np.nan; x_unres = np.nan; late_status = "ok"
    lost = status == "equilibrium"
    if lost.any():
        x_lost = float(x[order[np.argmax(lost)]]); late_status = "equilibrium"
    elif (status == "undefined").any():
        x_unres = float(x[order[np.argmax(status == "undefined")]]); late_status = "undefined"
    elif (status == "unresolved").any():
        x_unres = float(x[order[np.argmax(status == "unresolved")]]); late_status = "unresolved"
    # local refinement of the two suprema between neighbouring grid points
    xs = x[order]
    E = _refine_max(early_at, xs, early)
    if late_status == "equilibrium":
        Lt = np.inf
    else:
        ok = np.isfinite(late)
        Lt = _refine_max(lambda z: (lambda v: v[0] if v[1] in ("ok", "unresolved") else -np.inf)(late_at(z)),
                         xs[ok], late[ok]) if ok.any() else np.nan
    return ScalarReport(T_e=float(T_e), Theta_e=Theta_e, Pi=Pi, x_max=float(x[k]), Pi_rel=Pi_rel,
                        Lambda=Lambda, alpha=alpha, rtol=rtol,
                        Pi_early=float(E / (rtol * Theta_e)), Pi_late=float(Lt / (rtol * Theta_e)),
                        x_lost=x_lost, late_status=late_status, x_unresolved=x_unres)


def _quad(fun, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, _ = quad(fun, a, b, limit=200, epsabs=0, epsrel=1e-10)
    return v


def _refine_max(fun, xs, vals):
    """Largest value of fun near the best of the sampled points xs (bounded scalar search between
    the neighbouring samples); the result is never below the sampled maximum."""
    if xs.size == 0:
        return np.nan
    j = int(np.argmax(vals)); best = float(vals[j])
    lo, hi = xs[max(j - 1, 0)], xs[min(j + 1, xs.size - 1)]
    lo, hi = min(lo, hi), max(lo, hi)
    if hi > lo:
        r = minimize_scalar(lambda z: -fun(z), bounds=(lo, hi), method="bounded",
                            options=dict(xatol=1e-12 * max(abs(lo), abs(hi), hi - lo)))
        if r.success and np.isfinite(r.fun):
            best = max(best, -float(r.fun))
    return best


def _late_shift(f, inv, xa, xb, sgn, eq):
    """Exact delay int_{xb}^{xa} dz/f(z) caused by moving the state from xa back to xb.
    Returns (value, status): status 'ok', 'equilibrium' (value inf), 'unresolved' or 'undefined'."""
    lo, hi = min(xa, xb), max(xa, xb)
    if eq.size and np.any((eq >= lo) & (eq <= hi)):
        return np.inf, "equilibrium"
    zz = np.linspace(lo, hi, 257)
    with np.errstate(all="ignore"):
        fz = f(zz)
    if not np.all(np.isfinite(fz)):
        return np.nan, "undefined"
    if np.any(fz * sgn <= 0):                      # a zero or a sign change: an equilibrium on the way
        return np.inf, "equilibrium"
    if lo <= 0.0 <= hi and f(np.array([0.0]))[0] == 0:   # the common equilibrium x = 0, tested exactly
        return np.inf, "equilibrium"
    # interior minimum of |f|
    j = int(np.argmin(np.abs(fz)))
    a_, b_ = zz[max(j - 1, 0)], zz[min(j + 1, zz.size - 1)]
    r = minimize_scalar(lambda z: abs(f(np.array([z]))[0]), bounds=(a_, b_), method="bounded",
                        options=dict(xatol=1e-15 * max(abs(a_), abs(b_), hi - lo)))
    zm = float(r.x)
    fm = f(np.array([zm]))[0]
    if fm * sgn <= 0:
        return np.inf, "equilibrium"
    fend = min(abs(fz[0]), abs(fz[-1]))
    if abs(fm) >= 1e-3 * fend:                      # no pronounced minimum: plain quadrature
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _, info, *msg = quad(inv, lo, hi, limit=500, epsabs=0, epsrel=1e-10, full_output=1)
        if not msg:
            return abs(val), "ok"
    # a pronounced minimum of |f| at zm: does the integral of 1/f diverge there?  Integrate on shells
    # zm +- [e_k, e_(k-1)], e_k = w 10^(-2k), each of two decades, and watch their contributions.
    w = hi - lo
    edges = [w * 10.0 ** (-2 * kk) for kk in range(0, 16)]
    total, shells = 0.0, []
    for e_out, e_in in zip(edges[:-1], edges[1:]):
        c = 0.0
        for u, v in ((zm - e_out, zm - e_in), (zm + e_in, zm + e_out)):
            u, v = max(u, lo), min(v, hi)
            if v > u:
                c += abs(_quad(inv, u, v))
        shells.append(c); total += c
    shells = np.array(shells)
    nz = shells[shells > 0]
    if nz.size >= 4 and np.all(nz[-3:] / np.maximum(nz[-4:-1], 1e-300) > 5):
        return np.inf, "equilibrium"               # contributions grow like 1/e^(q-1), q >= 2
    if shells[-1] <= 1e-8 * total:
        return float(total), "ok"
    return float(total), "unresolved"


def _integral(fun, x):
    """int over the grid x of fun, piecewise with adaptive quadrature."""
    tot = 0.0
    idx = np.linspace(0, x.size - 1, 65).astype(int)
    for a, b in zip(x[idx[:-1]], x[idx[1:]]):
        if b > a:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v, _ = quad(fun, a, b, limit=200, epsabs=0, epsrel=1e-12)
            tot += v
    return tot

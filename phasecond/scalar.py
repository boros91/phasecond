"""Scalar autonomous equations x' = f(x): everything by quadrature, and exactly.

For a scalar equation the event-time adjoint is 1/f, a defect d at the state x moves the event by
exactly  int_x^{x+d} dz/f(z),  and the indicators need no differential equation.  This module
also gives the finite-tolerance versions of Pi: the exact shift caused by one admissible defect
+- rtol*s(x), which the first-order number Pi overestimates when rtol*s(x) is not small compared
with x, and which is infinite when an admissible defect can carry the solution onto an
equilibrium (the event can then be lost).
"""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import quad
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
            L.append("--> PHASE ILL-CONDITIONED at this tolerance")
        elif self.rtolPi >= 1e-3:
            L.append("--> grey zone")
        else:
            L.append("--> phase well-conditioned at this tolerance")
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


def scalar_report(f, x0, xe, rtol=1e-4, atol=1e-6, scale="scipy", alpha=1.0, fprime=None):
    """Indicators for the scalar problem x' = f(x), x(0) = x0, event x = xe.

    f must not vanish between x0 and xe (the event is reached).  f and fprime must accept arrays.
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
    early = np.zeros(x.size); late = np.zeros(x.size); x_lost = np.nan
    sub = np.unique(np.concatenate([np.linspace(0, x.size - 1, 500).astype(int),
                                    np.arange(max(k - 5, 0), min(k + 6, x.size))]))
    order = sub[np.argsort(sgn * x[sub])]  # along the motion
    for i in order:
        d = rtol * s[i]
        for arr, y in ((early, x[i] + sgn * d), (late, x[i] - sgn * d)):
            zz = np.linspace(x[i], y, 65)
            fz = f(zz)
            if np.any(fz * sgn <= 0) or not np.all(np.isfinite(fz)):
                arr[i] = np.inf
                if arr is late and np.isnan(x_lost):
                    x_lost = float(x[i])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val, _, info, *msg = quad(inv, x[i], y, limit=200, epsabs=0, epsrel=1e-10,
                                              full_output=1)
                # a zero of even order of f inside the interval: the integral diverges
                lost = bool(msg) or abs(val) > 1e3 * T_e
                arr[i] = np.inf if lost else abs(val)
                if lost and arr is late and np.isnan(x_lost):
                    x_lost = float(x[i])
    return ScalarReport(T_e=float(T_e), Theta_e=Theta_e, Pi=Pi, x_max=float(x[k]), Pi_rel=Pi_rel,
                        Lambda=Lambda, alpha=alpha, rtol=rtol,
                        Pi_early=float(early.max() / (rtol * Theta_e)),
                        Pi_late=float(late.max() / (rtol * Theta_e)), x_lost=x_lost)


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

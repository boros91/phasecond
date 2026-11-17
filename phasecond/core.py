"""A priori indicators of phase ill-conditioning for autonomous initial value problems

    x' = f(x),   x(0) = x0,   event g(x) = 0.

For the error scale s(x) of a solver (relative and absolute tolerance) the module computes

    Theta(t)  = sum_i s_i(x(t)) |lambda_i(t)|      timing sensitivity (lambda: event-time adjoint)
    Pi        = sup_t Theta(t) / Theta(T_e)         phase condition number
    Pi_star   = sup_t (1/mu(t)) / Theta(T_e)        adjoint-free lower bound, mu = max_i |f_i|/s_i
    Lambda    = (1/Theta(T_e)) int (Theta*omega)^alpha dt     integrated indicator

and derived quantities: Pi for relative control only, Pi in logarithmic unknowns, the largest
uniform absolute tolerance for which rtol*Pi stays below a target, and the components in which
the admissible error exceeds the solution itself.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["error_scale", "phase_report", "PhaseReport", "pi_star_from_run"]

TINY = 1e-300


def error_scale(x, rtol, atol, kind="scipy"):
    """Error scale s_i(x) of a solver that accepts a step when ||le_i / s_i|| <= rtol:
    |x_i| + atol_i/rtol (SciPy, SUNDIALS, ODEPACK) or max(|x_i|, atol_i/rtol) (MATLAB)."""
    x = np.asarray(x, float)
    ar = np.asarray(atol, float) / rtol
    if ar.ndim == 1 and x.ndim == 2:
        ar = ar[:, None]
    if kind == "scipy":
        return np.abs(x) + ar
    if kind == "matlab":
        return np.maximum(np.abs(x), ar)
    raise ValueError("kind must be 'scipy' or 'matlab'")


def _fd_jacobian(f, x, floor):
    n = x.size
    J = np.empty((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(x[j]), floor[j])
        e = np.zeros(n); e[j] = h
        J[:, j] = (f(x + e) - f(x - e)) / (2 * h)
    return J


def _fd_gradient(g, x, floor):
    n = x.size
    out = np.empty(n)
    for j in range(n):
        h = 1e-6 * max(abs(x[j]), floor[j])
        e = np.zeros(n); e[j] = h
        out[j] = (g(x + e) - g(x - e)) / (2 * h)
    return out


@dataclass
class PhaseReport:
    """Result of :func:`phase_report`.  ``print(report)`` gives a readable summary."""
    T_e: float
    Pi: float
    Pi_star: float
    Pi_rel: float
    Pi_log: float
    Lambda: float
    alpha: float
    rtol: float
    atol: object
    scale: str
    Theta_e: float
    t_max: float
    x_max: np.ndarray
    share_max: np.ndarray           # share of each component in Theta at its maximum
    uncontrolled: list              # components with s_i > |x_i| carrying >= 10% of Theta there
    adjoint_check: float
    atol_safe: float
    atol_floor: float               # largest uniform atol with Pi <= 2*Pi_rel
    target: float
    ref_rtolPi: float               # rtol*Pi of the reference run itself
    n_ref_steps: int
    cost: dict                      # function evaluations: reference run, Jacobians, total
    names: list
    t: np.ndarray = field(repr=False)
    Theta: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)
    Lam: np.ndarray = field(repr=False)
    ref: object = field(repr=False, default=None)
    adj: object = field(repr=False, default=None)

    # ------------------------------------------------------------------ derived
    @property
    def rtolPi(self):
        return self.rtol * self.Pi

    @property
    def level(self):
        """0: well-conditioned, 1: grey zone, 2: ill-conditioned at this tolerance."""
        return 2 if self.rtolPi >= 1 else (1 if self.rtolPi >= self.target else 0)

    @property
    def verdict(self):
        return ("phase well-conditioned at this tolerance",
                "grey zone: low-order implicit codes may misplace the event visibly",
                "PHASE ILL-CONDITIONED at this tolerance: one accepted step can move the event by its width")[self.level]

    def theta_for(self, rtol=None, atol=None, scale=None):
        """Theta(t) on the stored grid for another error scale (no new integration)."""
        S = error_scale(self.X, self.rtol if rtol is None else rtol,
                        self.atol if atol is None else atol, self.scale if scale is None else scale)
        return (S * np.abs(self.Lam)).sum(axis=0)

    def pi_for(self, rtol=None, atol=None, scale=None):
        """Pi for another error scale (no new integration: the adjoint does not depend on it)."""
        th = self.theta_for(rtol, atol, scale)
        return th.max() / th[-1]

    def time_tolerance(self, tol=None):
        """Absolute tolerance on t that corresponds to the tolerance tol at the event,
        for use with time as a dependent variable."""
        return (self.rtol if tol is None else tol) * self.Theta_e

    def __str__(self):
        nm = self.names
        dom = int(np.argmax(self.share_max))
        L = [f"event time T_e            = {self.T_e:.6g}",
             f"Theta(T_e)                = {self.Theta_e:.3g}   (time resolution of the event in the solver's scale)",
             f"Pi                        = {self.Pi:.3g}   (adjoint-free bound Pi_* = {self.Pi_star:.3g})",
             f"rtol * Pi                 = {self.rtolPi:.3g}",
             f"Lambda_alpha (alpha={self.alpha:g})  = {self.Lambda:.3g};  rtol^alpha * Lambda = {self.rtol**self.alpha*self.Lambda:.3g}",
             f"Theta is largest at t = {self.t_max:.4g} ({self.t_max/self.T_e:.1%} of T_e), "
             f"{self.share_max[dom]:.0%} from component {nm[dom]}",
             f"--> {self.verdict}"]
        if self.uncontrolled:
            L.append("    the admissible error exceeds the solution itself in: "
                     + ", ".join(nm[i] for i in self.uncontrolled)
                     + "  (the error test does not control their sign)")
        if self.level > 0:
            L.append("    what would help:")
            if np.isfinite(self.atol_safe):
                L.append(f"      - absolute tolerance <= {self.atol_safe:.0e} (then rtol*Pi <= {self.target:g})")
            else:
                L.append(f"      - absolute tolerance <= {self.atol_floor:.0e} gives Pi = {self.pi_for(atol=self.atol_floor):.3g}"
                         f" (relative control alone: {self.Pi_rel:.3g}); rtol must then be below "
                         f"{self.target / self.Pi_rel:.1e} as well")
            if np.isfinite(self.Pi_log):
                L.append(f"      - logarithmic unknowns: Pi = {self.Pi_log:.3g}")
            L.append(f"      - time as a dependent variable with AbsTol_t = {self.time_tolerance():.2g}")
        if self.ref_rtolPi > self.target:
            L.append(f"    WARNING: the reference run itself has rtol*Pi = {self.ref_rtolPi:.2g}; "
                     "tighten ref_rtol / ref_atol")
        if self.adjoint_check > 1e-4:
            L.append(f"    WARNING: adjoint check max|lambda.f - 1| = {self.adjoint_check:.1e}")
        return "\n".join(L)


def phase_report(f, x0, g, t_max, rtol=1e-4, atol=1e-6, scale="scipy", alpha=1.0, jac=None,
                 direction=0, occurrence=1, names=None, target=1e-3,
                 ref_rtol=1e-11, ref_atol=None, method="Radau", refine=8):
    """A priori phase conditioning of the event g(x) = 0 for x' = f(x), x(0) = x0.

    Parameters
    ----------
    f : callable x -> array            autonomous right-hand side
    x0 : array                         initial value
    g : callable x -> float            event function
    t_max : float                      upper bound for the event time
    rtol, atol : float / array         tolerances of the run to be judged
    scale : 'scipy' | 'matlab'         form of the error scale
    alpha : float                      exponent of the integrated indicator, p/(m+1)
    jac : callable x -> matrix         Jacobian of f (central differences if omitted)
    direction, occurrence              crossing direction and which crossing counts (1 = first after t = 0)
    ref_rtol, ref_atol, method         tolerances and method of the reference run.  With the
                                       tolerances of the production run (ref_rtol=rtol, ref_atol=atol)
                                       the indicators are computed from a cheap run; they depend on
                                       the orbit, not on its timing, so this works as long as that
                                       run stays near the orbit.
    """
    x0 = np.atleast_1d(np.asarray(x0, float))
    n = x0.size
    names = list(names) if names is not None else [f"x[{i}]" for i in range(n)]
    fv = lambda x: np.atleast_1d(np.asarray(f(x), float))
    F = lambda t, x: fv(x)
    if ref_atol is None:
        nz = np.abs(x0[x0 != 0])
        ref_atol = max(ref_rtol * 1e-12 * (nz.min() if nz.size else 1.0), TINY)

    ev = lambda t, x: g(x)
    ev.direction = direction
    # a start on the event surface (a point of a cycle taken at the event) may count as a crossing
    on_surface = abs(g(x0)) <= 1e-9 * max(np.abs(x0).max(), 1.0)
    ev.terminal = occurrence + (1 if on_surface else 0)
    kw = {}
    if jac is not None and method in ("Radau", "BDF", "LSODA"):
        kw["jac"] = lambda t, x: np.atleast_2d(jac(x))
    ref = solve_ivp(F, (0.0, t_max), x0, method=method, rtol=ref_rtol, atol=ref_atol,
                    events=ev, dense_output=True, **kw)
    te = ref.t_events[0]
    te = te[te > 1e-12 * t_max]
    if te.size < occurrence:
        raise RuntimeError("the reference solution did not reach the event before t_max"
                           + ("" if ref.success else f" (solver: {ref.message})"))
    T_e = float(te[occurrence - 1])
    x_e = ref.sol(T_e)

    tt = np.linspace(0.0, T_e, 400)
    xmax = np.abs(ref.sol(tt)).max(axis=1)
    # steps of the difference quotients: relative to the size of each component along the orbit;
    # a component that stays at zero gets a step relative to the largest component
    floor = np.maximum(1e-4 * xmax, 1e-8 * max(xmax.max(), TINY))

    # Jacobian along the reference: the user's, or central differences tabulated at the
    # reference steps (linear in t between them)
    tj = np.unique(np.concatenate([ref.t[ref.t < T_e], [0.0, T_e]]))
    if jac is not None:
        J_at = lambda s: np.atleast_2d(jac(ref.sol(s)))
        Jtab = None
    else:
        tj = np.unique(np.concatenate([tj, 0.5 * (tj[1:] + tj[:-1])]))
        Jtab = np.array([_fd_jacobian(fv, ref.sol(s), floor) for s in tj])

        def J_at(s):
            i = min(max(np.searchsorted(tj, s) - 1, 0), tj.size - 2)
            w = (s - tj[i]) / (tj[i + 1] - tj[i])
            return (1 - w) * Jtab[i] + w * Jtab[i + 1]

    # event-time adjoint: lambda' = -J^T lambda, lambda(T_e) = grad g / (grad g . f)
    gg = _fd_gradient(g, x_e, np.maximum(floor, 1e-4 * np.abs(x_e)))
    lam_e = gg / (gg @ fv(x_e))
    adj = solve_ivp(lambda s, lam: -J_at(s).T @ lam, (T_e, 0.0), lam_e, method="Radau",
                    rtol=1e-9, atol=1e-13 * np.abs(lam_e).max(), dense_output=True,
                    jac=lambda s, lam: -J_at(s).T)
    if not adj.success:
        raise RuntimeError("adjoint integration failed: " + adj.message)

    knots = np.unique(np.concatenate([ref.t[ref.t <= T_e], adj.t, [0.0, T_e]]))
    t = np.unique(np.concatenate([np.linspace(a, b, refine, endpoint=False)
                                  for a, b in zip(knots[:-1], knots[1:])] + [[T_e]]))
    X = ref.sol(t)
    Lam = adj.sol(t)
    Fx = np.array([fv(X[:, k]) for k in range(t.size)]).T
    check = float(np.abs((Lam * Fx).sum(axis=0) - 1).max())

    S = error_scale(X, rtol, atol, scale)
    contrib = S * np.abs(Lam)
    Theta = contrib.sum(axis=0)
    Theta_e = Theta[-1]
    k = int(np.argmax(Theta))
    share = contrib[:, k] / Theta[k]
    unc = [i for i in range(n) if S[i, k] > 2 * abs(X[i, k]) and share[i] >= 0.1]
    mu = (np.abs(Fx) / S).max(axis=0)
    Pi_star = float((1.0 / np.maximum(mu, TINY)).max() / Theta_e)

    # integrated indicator, sigma = Theta * omega, omega = ||J f||_s / ||f||_s, on the points tj
    Xj = ref.sol(tj); Sj = error_scale(Xj, rtol, atol, scale)
    Thj = (Sj * np.abs(adj.sol(tj))).sum(axis=0)
    sig = np.empty(tj.size)
    for i in range(tj.size):
        fi = fv(Xj[:, i])
        xpp = (Jtab[i] if Jtab is not None else J_at(tj[i])) @ fi
        om = (np.abs(xpp) / Sj[:, i]).max() / max((np.abs(fi) / Sj[:, i]).max(), TINY)
        sig[i] = (Thj[i] * om) ** alpha
    Lambda = float(np.sum(0.5 * (sig[1:] + sig[:-1]) * np.diff(tj)) / Theta_e)

    # quantities that need no further integration
    def pi_of(S_):
        th = (S_ * np.abs(Lam)).sum(axis=0)
        return float(th.max() / th[-1])
    Pi_rel = pi_of(np.abs(X) + TINY)
    atol_safe = np.nan
    for a in 10.0 ** np.arange(0, -40.5, -0.5):
        if rtol * pi_of(error_scale(X, rtol, a, scale)) <= target:
            atol_safe = a
            break
    atol_floor = np.nan
    for a in 10.0 ** np.arange(0, -40.5, -0.5):
        if pi_of(error_scale(X, rtol, a, scale)) <= 2 * Pi_rel:
            atol_floor = a
            break
    Pi_log = np.nan
    if np.all(X > 0):          # unknowns u = ln x: du = dx/x, absolute control of u
        ar = np.max(np.asarray(atol, float)) / rtol
        Slog = error_scale(np.log(X), rtol, np.max(np.asarray(atol, float)), scale)
        th = (Slog * np.abs(Lam * X)).sum(axis=0)
        Pi_log = float(th.max() / th[-1])
    ref_pi = pi_of(error_scale(X, ref_rtol, ref_atol, "scipy"))
    cost = dict(reference_nfev=int(ref.nfev), reference_steps=int(ref.t.size), adjoint_steps=int(adj.t.size),
                jacobians=int(tj.size) if Jtab is not None else int(adj.njev + adj.nfev + tj.size),
                fd_jacobian_nfev=int(2 * n * tj.size) if Jtab is not None else 0)

    return PhaseReport(T_e=T_e, Pi=float(Theta[k] / Theta_e), Pi_star=Pi_star, Pi_rel=Pi_rel, Pi_log=Pi_log,
                       Lambda=Lambda, alpha=alpha, rtol=rtol, atol=atol, scale=scale, Theta_e=float(Theta_e),
                       t_max=float(t[k]), x_max=X[:, k].copy(), share_max=share, uncontrolled=unc,
                       adjoint_check=check, atol_safe=float(atol_safe), atol_floor=float(atol_floor), target=target,
                       ref_rtolPi=float(ref_rtol * ref_pi), n_ref_steps=int(ref.t.size), cost=cost, names=names,
                       t=t, Theta=Theta, X=X, Lam=Lam, ref=ref, adj=adj)


def pi_star_from_run(f, X, g, rtol, atol, scale="scipy"):
    """Adjoint-free lower bound Pi_* from the points X[:, k] of any computed trajectory that
    ends at (or just after) the event g = 0.  Costs one evaluation of f per point."""
    X = np.atleast_2d(np.asarray(X, float))
    fv = lambda x: np.atleast_1d(np.asarray(f(x), float))
    S = error_scale(X, rtol, atol, scale)
    mu = np.array([(np.abs(fv(X[:, k])) / S[:, k]).max() for k in range(X.shape[1])])
    xe = X[:, -1]
    floor = np.maximum(1e-4 * np.abs(X).max(axis=1), TINY)
    gg = _fd_gradient(g, xe, floor)
    Theta_e = (S[:, -1] * np.abs(gg)).sum() / abs(gg @ fv(xe))
    return float((1.0 / np.maximum(mu, TINY)).max() / Theta_e)

# Changelog

## 0.1.2 (2026-11-18)

- `scalar_report`, delay (`Pi_late`): the status `"equilibrium"` (and `Pi_late = inf`, "THE EVENT
  CAN BE LOST") is now reserved for a demonstrated equilibrium: a value passed in `equilibria`, an
  evaluated zero of f, or a sign change of f. In 0.1.1, contributions of 1/f that grew on three
  successive shells around a minimum of |f| were also taken as a zero; a positive f with a very
  narrow minimum, e.g. f = (x - 1.5)^2 + 1e-32, was then reported as a loss of the event. Such
  cases are now `"unresolved"`.
- The integral of 1/f near a narrow minimum is accepted (`"ok"`) only if the part that floating
  point cannot resolve, around the minimum, has a known contribution (f nearly constant there).
  In 0.1.1 shells that collapsed at machine precision contributed zero and could give status ok
  with a delay above the exact bound (f = (x - 1)^2 + 1e-34). For `"unresolved"` the value is the
  resolved part of the integral, a lower estimate. The minimum of |f| is located by repeated
  sampling down to floating-point resolution.
- The printed verdict follows `late_status`: an undefined delay (the admissible error leaves the
  domain of f, e.g. f = 1 + sqrt(x) near 0) gets its own note instead of "the event can be lost
  ... at x = nan". When the backward shift is undefined or unresolved and no large shift is
  demonstrated, the verdict says that the finite-tolerance classification is incomplete, instead
  of "well-conditioned" or "grey zone".
- New tests for these cases. The validation tables of the paper are unchanged.

## 0.1.1 (2026-11-18)

Corrections to version 0.1.0.

- `scalar_report`, advance (`Pi_early`): a defect in the direction of motion that would carry the
  solution past the event is now clipped at the event, so the advance is at most the remaining time.
  Version 0.1.0 integrated past the event and could exceed the exact bound (for x' = 1 from 1 to
  1.001 with rtol = 0.01 it reported 0.01001 instead of 0.001).
- `scalar_report`, delay (`Pi_late`): infinity is reported only when an equilibrium between the
  perturbed and the original state is demonstrated: a sign change of f or f = 0 on a sampling
  grid, f(0) = 0 when x = 0 lies in the interval, an equilibrium passed in the new argument
  `equilibria`, or contributions of 1/f that keep growing on shells around an interior minimum of
  |f|. Version 0.1.0 also reported infinity when quadrature warned or the delay exceeded 1000 T_e,
  which declared a false loss of the event for short trips (x' = 1 from 1 to 1.000001). A delay that
  quadrature cannot resolve is kept as an estimate, with `late_status = "unresolved"`.
- `scalar_report`: the suprema over the route are refined between the grid points (bounded scalar
  search); they remain estimates, not certified bounds.
- `phase_report`, tolerance advice: an absolute tolerance is recommended only if it and every smaller
  tested value (half-decade grid) pass, and only while the event's time resolution Theta(T_e) stays
  within 10% of its value for the smallest tested tolerance. Version 0.1.0 printed "absolute
  tolerance <= a", which did not hold in general, and could reward a larger tolerance that lowers
  the normalized Pi only by coarsening the event (x' = -1 from 100 to 1). The report now gives the
  tested value, rtol*Pi there and the largest one-defect shift in time units.
- `phase_report`, component flags: "the admissible error exceeds the solution itself" is replaced
  by two flags: the absolute-tolerance regime (s_i > 2|x_i| where Theta is largest, the field
  `uncontrolled` as before) and the components whose sign one admissible error does not protect
  (rtol*s_i >= |x_i| while they carry at least 10% of Theta, new field `sign_unprotected`).
- Both reports warn when Theta(T_e) exceeds T_e: the event level is then below the error scale,
  and a small Pi does not mean an accurate event time.
- `quick_check`: a large Pi_* computed from a run flags possible ill-conditioning; it is an
  estimate, not a proof (wording of the message and of the documentation).
- New tests for these cases; the validation tables of the paper (`results/table_detector.tex`,
  `results/table_finite.tex`) are unchanged.

## 0.1.0 (2026-11-17)

First release.

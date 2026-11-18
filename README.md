# phasecond

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24321464.svg)](https://doi.org/10.5281/zenodo.24321464)

Detection of *phase ill-conditioning* of autonomous initial value problems

    x' = f(x),   x(0) = x0,   event g(x) = 0.

In problems with a slow passage before a fast event (ignition, an epidemic peak, a spike, a
front), a solver that controls the local error in the state can return a solution with the right
shape at the wrong time, or lose the event, although every step passes its error test. This
package computes, before or after a run, the numbers that tell whether this will happen for a
given solver error scale (relative and absolute tolerance), where the sensitivity comes from, and
which change of tolerances or unknowns removes it. The theory is in the paper *Phase
ill-conditioning of autonomous initial value problems: event times under local error control* (I. Boros).

Version 0.1.0. Python 3.9 or later with NumPy and SciPy (tested with Python 3.13 and SciPy 1.18.1);
Cantera 3.2.0 only for the hydrogen example.

## Installation

```
pip install "git+https://github.com/boros91/phasecond@v0.1.0"
```

or, from a clone of this repository, `pip install .` (add `[test]` for pytest). The examples and
tests are run from the clone: `python examples/demo.py`, `pytest -q`.

## What it computes

| quantity | meaning |
|---|---|
| `Theta(t)` | timing sensitivity: largest first-order shift of the event caused by an error of one unit of the solver's error scale at time t |
| `Pi` | phase condition number, sup Theta / Theta(T_e). `rtol*Pi >= 1`: one accepted step can move the event by its own time resolution |
| `Pi_star` | lower bound of Pi that needs no adjoint (speed of the solution in its error scale) |
| `Lambda` | integrated indicator; `K * rtol**alpha * Lambda` models the accumulated shift |
| `Pi_early`, `Pi_late` | scalar equations: exact shift by one admissible error, in both directions; `Pi_late = inf` means the event can be lost |

## The four entry points

```python
from phasecond import phase_report, scalar_report, quick_check, orbit_test, shift_estimate
```

1. **`quick_check(f, sol, g, rtol, atol)`**: after a run. `Pi_star` from the accepted steps of a
   `solve_ivp` result; one evaluation of `f` per step. A large value proves the problem.
2. **`phase_report(f, x0, g, t_max, rtol, atol)`**: the full a priori report: one reference
   integration to the event, one backward adjoint integration. Prints the verdict, the component
   and the time where the sensitivity sits, and what would help (absolute tolerance, logarithmic
   unknowns, tolerance on time). `report.pi_for(atol=...)` rescans tolerances without any new
   integration, because the adjoint does not depend on the error scale.
   With `ref_rtol=rtol, ref_atol=atol, method=...` the report is computed from a run with the
   production tolerances (cheap mode).
3. **`scalar_report(f, x0, xe, rtol, atol)`**: scalar equations, by quadrature, including the
   finite-tolerance numbers.
4. **`orbit_test(f, x0, t_span, rtol)`** and **`shift_estimate(f, t, X, report)`**: a black-box
   comparison of two runs at equal times and as orbits, and a signed estimate of the event-time
   error of a given run from its true defects and the adjoint.

## Example

```python
import numpy as np
from phasecond import phase_report

beta, gamma = 0.3, 0.1
sir = lambda u: np.array([-beta*u[0]*u[1], beta*u[0]*u[1] - gamma*u[1]])
rep = phase_report(sir, [1 - 1e-8, 1e-8], lambda u: u[0] - gamma/beta, t_max=1000,
                   rtol=1e-4, atol=1e-6, names=["S", "I"])
print(rep)
```

```
event time T_e            = 95.9529
Theta(T_e)                = 11.4   (time resolution of the event in the solver's scale)
Pi                        = 4.38e+05   (adjoint-free bound Pi_* = 4.38e+05)
rtol * Pi                 = 43.8
...
--> PHASE ILL-CONDITIONED at this tolerance: one accepted step can move the event by its width
    the admissible error exceeds the solution itself in: I  (the error test does not control their sign)
    what would help:
      - absolute tolerance <= 1e-11 gives Pi = 16.7 (relative control alone: 12.2); rtol must then be below 8.2e-05 as well
      - logarithmic unknowns: Pi = 7.6
      - time as a dependent variable with AbsTol_t = 0.0011
```

(With `scale="matlab"` the error scale is `max(|x|, atol/rtol)` instead of `|x| + atol/rtol`.)

## Files

| | |
|---|---|
| `phasecond/core.py` | `phase_report`, `PhaseReport`, `error_scale`, `pi_star_from_run` |
| `phasecond/scalar.py` | `scalar_report` |
| `phasecond/posteriori.py` | `quick_check`, `orbit_test`, `shift_estimate` |
| `examples/problems.py`, `examples/h2_problem.py` | test problems of the paper |
| `examples/demo.py` | prints the reports for five problems |
| `examples/validate.py` | validation runs for the paper -> `results/` (Tables SM12 and SM13 of the paper) |
| `tests/` | `pytest -q` (30 s); `pytest -q --runslow` adds Morris-Lecar and hydrogen-air |

## Validation (see `results/validation.txt`)

- The tests compare with the values saved for the paper: Pi, Pi_*, Pi for relative control,
  Theta(T_e) and T_e for 10 scalar cases, SIR (4 initial values, with and without a user
  Jacobian), Morris-Lecar (2 cases) and hydrogen-air (3 tolerance ratios). Agreement: 1e-4 or
  better in Pi.
- Cheap mode, 7 problems, runs with rtol 1e-4, atol 1e-6: from the 12 BDF and LSODA runs that
  reach the event, Pi_* agrees with the tight value to 0.05% and Pi lies between 0.49 and 1.01
  times the tight value, although these runs misplace the event by up to 36 times its time
  resolution. From RK45 runs (few steps, coarse dense output) Pi is within 0.31..1.14 and Lambda
  is unreliable. Two implicit runs (hydrogen) do not reach the event at all; the function then
  raises an error and the tight mode must be used.
- `shift_estimate` reproduces the measured event-time error of SIR runs to 2% or better at
  tolerance 1e-6.

## Limits

- Pi is a first-order, worst-case, one-step quantity. It says that the event time is unprotected,
  not how large the error of a particular code will be; method constants differ by orders of
  magnitude between codes.
- The Jacobian is taken by central differences unless `jac` is given; right-hand sides that are
  not defined for slightly negative arguments need `jac`.
- The reference run must itself be phase-accurate (the report warns if its own `rtol*Pi` is not
  small) or, in cheap mode, must stay near the orbit.
- Autonomous problems only; one event; no MATLAB version yet.

## Related

The scripts and data that produce the figures and tables of the paper are in the repository
[phase-ill-conditioning](https://github.com/boros91/phase-ill-conditioning), which uses this
package at version 0.1.0.

## Citation

If you use this package, please cite the paper and the package (see `CITATION.cff`). The package is archived on Zenodo: https://doi.org/10.5281/zenodo.24321464 (all versions); version 0.1.0: https://doi.org/10.5281/zenodo.24321465.

## License

MIT, see `LICENSE`.

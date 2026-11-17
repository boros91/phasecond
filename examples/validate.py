"""Validation runs of the detector for the paper.

1. cheap mode: indicators computed from a run made with the production tolerances
   (rtol 1e-4, atol 1e-6) compared with those from a tight reference, with the costs;
2. finite-tolerance Pi for the scalar test problems.
Output: ../results/validation_cheap.csv, validation_finite.csv, validation.txt, and the LaTeX
tables table_detector.tex, table_finite.tex.
"""
import csv, os, sys, time
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import problems as P
from phasecond import phase_report, scalar_report

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R, A = 1e-4, 1e-6
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.append(s)


cases = [("flame, $q=2$, $\\delta=10^{-4}$", "I", P.flame(2), [1e-4], lambda y: y[0] - 0.5, 2e4, {}),
         ("flame, $q=3$, $\\delta=10^{-2}$", "I", P.flame(3), [1e-2], lambda y: y[0] - 0.5, 2e4, {}),
         ("logistic, $x_0=10^{-5}$", "II", P.logistic, [1e-5], lambda y: y[0] - 0.5, 100, {}),
         ("SIR, $I_0=10^{-6}$", "II", P.sir, [1 - 1e-6, 1e-6], P.sir_event, 1000, {}),
         ("ghost, $\\varepsilon=10^{-6}$", "III", P.ghost(1e-6), [-1.0], lambda y: y[0] - 1.0, 1e5, {})]
f = P.morris_lecar(1e-2)
cases.append(("Morris--Lecar, $\\Delta=10^{-2}$", "III", f, P.ml_cycle_point(f), lambda x: x[0] - 10.0, 20000,
              dict(direction=1)))
try:
    import h2_problem as H
    cases.append(("H$_2$--air", "II", H.rhs, H.x0, H.event, 1e-3, dict(ref_atol=1e-24)))
except ImportError:
    say("Cantera not available: hydrogen case skipped")

rows = []
for name, cls, f, x0, g, tmax, kw in cases:
    t0 = time.time()
    a = phase_report(f, x0, g, tmax, R, A, **kw)
    ta = time.time() - t0
    say(f"{name}: tight  T_e={a.T_e:.8g} Pi={a.Pi:.5g} Pi*={a.Pi_star:.5g} Lambda={a.Lambda:.5g} "
        f"nfev={a.cost['reference_nfev']} steps={a.cost['reference_steps']} time={ta:.1f}s")
    for m in ("BDF", "LSODA", "RK45"):
        kw2 = {k: v for k, v in kw.items() if k != "ref_atol"}
        t0 = time.time()
        try:
            b = phase_report(f, x0, g, tmax, R, A, ref_rtol=R, ref_atol=A, method=m, **kw2)
        except RuntimeError as e:
            say(f"   {m}: no event ({str(e)[:60]})")
            rows.append(dict(case=name, cls=cls, method=m, event=False, Pi=a.Pi, Pi_star=a.Pi_star,
                             Lambda=a.Lambda, tight_nfev=a.cost["reference_nfev"], tight_time=ta))
            continue
        tb = time.time() - t0
        tot = b.cost["reference_nfev"] + b.cost["fd_jacobian_nfev"]
        say(f"   {m}: shift={b.T_e - a.T_e:+.4g} Pi={b.Pi:.5g} Pi*={b.Pi_star:.5g} Lambda={b.Lambda:.5g} "
            f"run nfev={b.cost['reference_nfev']} steps={b.cost['reference_steps']} "
            f"detector extra nfev={b.cost['fd_jacobian_nfev']} time={tb:.2f}s")
        rows.append(dict(case=name, cls=cls, method=m, event=True, Pi=a.Pi, Pi_star=a.Pi_star, Lambda=a.Lambda,
                         tight_nfev=a.cost["reference_nfev"], tight_time=ta, shift=b.T_e - a.T_e,
                         eta=abs(b.T_e - a.T_e) / a.Theta_e, Pi_run=b.Pi, Pi_star_run=b.Pi_star,
                         Lambda_run=b.Lambda, run_nfev=b.cost["reference_nfev"], run_steps=b.cost["reference_steps"],
                         extra_nfev=b.cost["fd_jacobian_nfev"], adjoint_steps=b.cost["adjoint_steps"], time=tb))

keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[1]).index(k) if k in rows[1] else 99)
with open(os.path.join(OUT, "validation_cheap.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["event"]]
rp = np.array([r["Pi_run"] / r["Pi"] for r in ok]); rs = np.array([r["Pi_star_run"] / r["Pi_star"] for r in ok])
rl = np.array([r["Lambda_run"] / r["Lambda"] for r in ok])
say(f"cheap/tight over {len(ok)} runs with event ({len(rows) - len(ok)} without): "
    f"Pi {rp.min():.2f}..{rp.max():.2f}, Pi_* {rs.min():.3f}..{rs.max():.3f}, Lambda {rl.min():.2f}..{rl.max():.2f}")
imp = [r for r in ok if r["method"] != "RK45"]
say(f"implicit codes only ({len(imp)} runs): Pi {min(r['Pi_run']/r['Pi'] for r in imp):.2f}.."
    f"{max(r['Pi_run']/r['Pi'] for r in imp):.2f}, Pi_* {min(r['Pi_star_run']/r['Pi_star'] for r in imp):.4f}.."
    f"{max(r['Pi_star_run']/r['Pi_star'] for r in imp):.4f}")


def sci(x, d=1):
    if not np.isfinite(x):
        return "$\\infty$"
    if 0.01 <= abs(x) < 1000:
        return f"${x:.3g}$"
    m, e = f"{x:.{d}e}".split("e")
    if float(m) == 1.0:
        return f"$10^{{{int(e)}}}$"
    return f"${m}\\cdot10^{{{int(e)}}}$"


with open(os.path.join(OUT, "table_detector.tex"), "w") as fh:
    fh.write("\\begin{tabular}{llcccccc}\n\\toprule\n"
             "& & \\multicolumn{2}{c}{tight reference} & \\multicolumn{4}{c}{from a BDF run with the production tolerances}\\\\\n"
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
             "problem & class & $\\Pi$ & $\\Pi_*$ & $\\eta$ of the run & $\\Pi$ & $\\Pi_*$ & steps\\\\\n\\midrule\n")
    for name, cls, *_ in cases:
        r = [q for q in rows if q["case"] == name and q["method"] == "BDF"][0]
        if r["event"]:
            fh.write(f"{name} & {cls} & {sci(r['Pi'])} & {sci(r['Pi_star'])} & {sci(r['eta'])} & {sci(r['Pi_run'])} & "
                     f"{sci(r['Pi_star_run'])} & {r['run_steps']}\\\\\n")
        else:
            fh.write(f"{name} & {cls} & {sci(r['Pi'])} & {sci(r['Pi_star'])} & \\multicolumn{{4}}{{c}}{{no event}}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- finite tolerance, scalar problems
fin = []
sc = [("flame, $q=2$", P.flame(2), [1e-2, 1e-4, 1e-5, 1e-6], 0.5), ("logistic", P.logistic, [1e-3, 1e-5, 1e-6, 1e-7], 0.5)]
for name, f, x0s, xe in sc:
    for x0 in x0s:
        for tol in (1e-4, 1e-6):
            r = scalar_report(f, x0, xe, tol, tol / 100)
            fin.append(dict(case=name, x0=x0, tol=tol, atol=tol / 100, rPi=r.rtolPi, rPi_early=tol * r.Pi_early,
                            rPi_late=tol * r.Pi_late, Te_over_Theta=r.T_e / r.Theta_e))
            say(f"{name} x0={x0:g} tol={tol:g}: rPi={r.rtolPi:.4g} early={tol*r.Pi_early:.4g} late={tol*r.Pi_late:.4g} "
                f"T_e/Theta_e={r.T_e/r.Theta_e:.4g}")
with open(os.path.join(OUT, "validation_finite.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fin[0])); w.writeheader(); w.writerows(fin)
with open(os.path.join(OUT, "table_finite.tex"), "w") as fh:
    fh.write("\\begin{tabular}{llcccc}\n\\toprule\nproblem & $x_0$ & $\\AbsTol/x_0$ & $r\\,\\Pi$ & $r\\,\\Pi^{+}$ & $r\\,\\Pi^{-}$\\\\\n\\midrule\n")
    for r in fin:
        if r["tol"] == 1e-4:
            fh.write(f"{r['case']} & {sci(r['x0'])} & {sci(r['atol']/r['x0'])} & {sci(r['rPi'])} & {sci(r['rPi_early'])} & {sci(r['rPi_late'])}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(OUT, "validation.txt"), "w") as fh:
    fh.write("\n".join(log) + "\n")

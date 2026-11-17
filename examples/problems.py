"""Test problems of the manuscript, in the form used by phasecond (autonomous f(x), x0, event g)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


# ---------------------------------------------------------------- scalar problems
def flame(q=2):
    return lambda y: y**q * (1 - y)

logistic = lambda x: x * (1 - x)

def ghost(eps):
    return lambda x: eps + x * x


# ---------------------------------------------------------------- SIR
BETA, GAMMA = 0.3, 0.1

def sir(u):
    S, I = u
    return np.array([-BETA * S * I, BETA * S * I - GAMMA * I])

def sir_jac(u):
    S, I = u
    return np.array([[-BETA * I, -BETA * S], [BETA * I, BETA * S - GAMMA]])

sir_event = lambda u: u[0] - GAMMA / BETA          # peak of I


# ---------------------------------------------------------------- decaying transverse mode (Example 3.4)
def coupled(c=0.5, k=1.0):
    return lambda u: np.array([u[0]**2 * (1 - u[0]) + c * u[1], -k * u[1]])


# ---------------------------------------------------------------- Morris-Lecar, class I parameters
C, gL, gCa, gK = 20.0, 2.0, 4.0, 8.0
VL, VCa, VK = -60.0, 120.0, -84.0
V1, V2, V3, V4, PHI = -1.2, 18.0, 12.0, 17.4, 1.0 / 15.0
minf = lambda V: 0.5 * (1 + np.tanh((V - V1) / V2))
winf = lambda V: 0.5 * (1 + np.tanh((V - V3) / V4))
tauw = lambda V: 1.0 / np.cosh((V - V3) / (2 * V4))

def ml_snic_current():
    Iss = lambda V: gL * (V - VL) + gCa * minf(V) * (V - VCa) + gK * winf(V) * (V - VK)
    r = minimize_scalar(lambda V: -Iss(V), bounds=(-40, 0), method="bounded", options=dict(xatol=1e-12))
    return Iss(r.x)

def morris_lecar(Delta):
    I = ml_snic_current() + Delta
    def f(x):
        V, w = x
        return np.array([(I - gL * (V - VL) - gCa * minf(V) * (V - VCa) - gK * w * (V - VK)) / C,
                         PHI * (winf(V) - w) / tauw(V)])
    return f

def ml_cycle_point(f, VE=10.0):
    """A point of the limit cycle at an upward crossing of V = VE."""
    s = solve_ivp(lambda t, x: f(x), (0, 40000.0), [-30.0, 0.1], method="DOP853", rtol=1e-13, atol=1e-15,
                  dense_output=True)
    V = s.y[0]
    up = np.where((V[:-1] < VE) & (V[1:] >= VE))[0]
    i = up[-2]
    tc = brentq(lambda t: s.sol(t)[0] - VE, s.t[i], s.t[i + 1], xtol=1e-13)
    return s.sol(tc)

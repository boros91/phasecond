"""Adiabatic constant-pressure H2-air ignition (Cantera, h2o2.yaml), unknowns (T, Y_1..Y_10).
Needs Cantera.  The right-hand side passes the mass fractions without normalization or clipping."""
import numpy as np
import cantera as ct

gas = ct.Solution("h2o2.yaml")
P0, T0 = ct.one_atm, 1000.0
gas.TPX = T0, P0, "H2:2, O2:1, N2:3.76"
x0 = np.concatenate([[T0], gas.Y])
W = gas.molecular_weights
names = ["T"] + list(gas.species_names)


def rhs(x):
    try:
        gas.set_unnormalized_mass_fractions(x[1:])
        gas.TP = x[0], P0
    except ct.CanteraError:
        return np.full(len(x), np.nan)
    rho, wdot = gas.density, gas.net_production_rates
    return np.concatenate([[-np.dot(gas.partial_molar_enthalpies, wdot) / (rho * gas.cp_mass)], wdot * W / rho])


event = lambda x: x[0] - (T0 + 400.0)

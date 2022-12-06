"""Module to compute transport quantities for the toy antiporter system.

It is capable of producing predictions for electrogenic and electroneutral
systems.
"""

from multibind.nonequilibrium import rate_matrix
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

k_B = 1.380649e-23  # J/K
T = 310  # K
kT = k_B * T
e = 1.602176634e-19  # C


def transport(c, rates, h_counter, na_counter, **kwargs):
    """Calculate the proton transfer as a function of system paramters.

    h_counter: the proton concentration at which the non-standard state rates
               were defined.

    na_counter: the sodium concentration at which the non-standard state rates
               were defined.

    Charges are defined in multiples of the elementary charge e.

    Default values below.

    c0 = kwargs.get('c0', 1)

    c_h_in = kwargs.get('c_h_in', c0 * 10 ** (-7.4))

    c_h_out = kwargs.get('c_h_out', c0 * 10 ** (-7.0))

    c_na_in = kwargs.get('c_na_in', 0.010)

    c_na_out = kwargs.get('c_na_out', 0.100)

    voltage = kwargs.get('voltage', -0.100) Measured as in-out
    """
    c0 = kwargs.get('c0', 1)  # molar
    c_h_in = kwargs.get('c_h_in', c0 * 10 ** (-7.4))   # pH = 7.4
    c_h_out = kwargs.get('c_h_out', c0 * 10 ** (-7.0))  # pH = 7
    c_na_in = kwargs.get('c_na_in', 0.010)  # 10 mM
    c_na_out = kwargs.get('c_na_out', 0.100)  # 100 mM
    voltage = kwargs.get('voltage', -0.100)  # V

    # c, rates, std = rate_matrix('inputs/rates_transport.csv')
    states = c.states.values
    connections = c.graph[['state1', 'state2']].values

    Gp = rates.copy()

    Gp[1, 0] *= c_h_in / h_counter
    Gp[1, 2] *= c_na_in / na_counter
    Gp[4, 3] *= c_na_out / na_counter
    Gp[4, 5] *= c_h_out / h_counter

    charge = kwargs.get('charge', 1)

    charge_H = kwargs.get('charge_H', charge)
    charge_N = kwargs.get('charge_N', charge)

    # H+ out to H+ in
    Gp[0, 5] *= np.exp(charge_H * e * 0.5 * voltage / kT)
    # H+ in to H+ out
    Gp[5, 0] *= np.exp(-charge_H * e * 0.5 * voltage / kT)

    # Na+ out to Na+ in
    Gp[2, 3] *= np.exp(charge_N * e * 0.5 * voltage / kT)
    # Na+ in to Na+ out
    Gp[3, 2] *= np.exp(-charge_N * e * 0.5 * voltage / kT)

    n = c.states.shape[0]
    Gp = Gp.T

    for i in range(n):
        Gp[i, i] = 0
        Gp[i, i] = -Gp[:, i].sum()

    Gp = np.vstack((Gp, np.ones(n)))

    U, S, VT = svd(Gp, full_matrices=False)

    Gp_reconstructed = U @ np.diag(S) @ VT

    if not np.allclose(Gp, Gp_reconstructed, atol=1e-3):
        raise ValueError("SVD failed.")

    U_inv = U.T
    S_inv = np.diag(1 / S)
    VT_inv = VT.T

    sums = n

    assert np.isclose((U_inv @ U).sum(), sums)
    assert np.isclose((VT @ VT_inv).sum(), sums)
    assert np.isclose((S_inv @ np.diag(S)).sum(), sums)
    assert np.isclose((np.diag(S) @ S_inv).sum(), sums)

    Gp_inv = VT_inv @ S_inv @ U_inv
    assert np.isclose((Gp_inv @ Gp).sum(), sums)
    # set up steady state conditions. dp_i/dt = 0
    # and populations = 1
    steady_state_conditions = np.zeros(n + 1)
    steady_state_conditions[-1] = 1
    steady_state_populations = Gp_inv @ steady_state_conditions
    # print(steady_state_populations)
    assert np.isclose(steady_state_populations.sum(), 1)

    net = []
    forward_flux = []
    backward_flux = []

    connection_labels = []
    drive = []

    states = c.states.values

    for i, j in connections:
        i_i = np.argwhere(states == i)[0, 0]
        i_j = np.argwhere(states == j)[0, 0]

        # remember that Gp <- Gp.T
        ff = Gp[i_j, i_i] * steady_state_populations[i_i]
        fb = Gp[i_i, i_j] * steady_state_populations[i_j]
        net.append(ff - fb)
        forward_flux.append(ff)
        backward_flux.append(fb)
        drive.append(-np.log(Gp.T[i_i, i_j] / Gp.T[i_j, i_i]))
        connection_labels.append(f"{i}/{j}")

    return net[0], drive, connection_labels, steady_state_populations, Gp.T

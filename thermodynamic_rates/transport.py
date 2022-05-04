from multibind.nonequilibrium import rate_matrix
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

R = 8.314472
T = 310
F = 96485.3321

RT = R * T


def transport(**kwargs):
    c0 = kwargs.get('c0', 1)  # molar
    c_h_in = kwargs.get('c_h_in', c0 * 10 ** (-7.4))   # pH = 7.4
    c_h_out = kwargs.get('c_h_out', c0 * 10 ** (-7.0))  # pH = 7
    c_na_in = kwargs.get('c_na_in', 0.010)  # 10 mM
    c_na_out = kwargs.get('c_na_out', 0.100)  # 100 mM
    voltage = kwargs.get('voltage', -0.100)  # V

    c, rates, std = rate_matrix('inputs/rates_transport.csv')
    states = c.states.values
    connections = c.graph[['state1', 'state2']].values

    Gp = rates.copy()

    Gp[1, 0] *= c_h_in
    Gp[1, 2] *= c_na_in
    Gp[4, 3] *= c_na_out
    Gp[4, 5] *= c_h_out

    # (Pdb++) p c.states
    #    name
    # 0   IFH
    # 1   IF0
    # 2  IFNA
    # 3  OFNA
    # 4   OF0
    # 5   OFH

    print(-np.log(Gp[0, 5] / Gp[5, 0]))
    # IFH --> OFH
    Gp[0, 5] *= np.exp(voltage * F / RT / 2)
    Gp[5, 0] *= np.exp(-voltage * F / RT / 2)
    print(-np.log(Gp[0, 5] / Gp[5, 0]))
    print(-np.log(Gp[3, 2] / Gp[2, 3]))
    # IFNA --> OFNA
    Gp[2, 3] *= np.exp(voltage * F / RT / 2)
    Gp[3, 2] *= np.exp(-voltage * F / RT / 2)
    print(-np.log(Gp[3, 2] / Gp[2, 3]))

    print((F / RT / 2)**-1)

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
    print(steady_state_populations)
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

    net = np.array(net)
    return net[0], drive, connection_labels, steady_state_populations


def main():

    colors = ['blue', 'orange', 'green']

    fig, (ax_trans, ax_drive, ax_pops) = plt.subplots(3)
    voltages = np.linspace(-0.5, 0.5, 3)
    na_out = np.logspace(-3, 0, 40)
    for i, v in enumerate(voltages):
        fluxes = []
        for na in na_out:
            flux, drive, connection_labels, pops = transport(**{'c_na_out': na, 'voltage': v})
            ax_drive.plot([j + i * 0.1 for j, _ in enumerate(drive)], drive, 'o', color=colors[i])
            ax_pops.plot([j + i * 0.1 for j, _ in enumerate(pops)], pops, 'o', color=colors[i])
            fluxes.append(-flux)
        ax_trans.semilogx(na_out, fluxes, '-', label=f"{v:.5f} V")
    ax_trans.set_xlabel("[Na+] (M)")
    ax_trans.axvline(0.1, color='black')
    ax_trans.set_ylabel("steady-state turnover")
    ax_trans.legend(loc='best')
    print(connection_labels)
    plt.show()


if __name__ == "__main__":
    main()

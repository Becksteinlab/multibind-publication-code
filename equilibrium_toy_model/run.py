#!/usr/bin/env python

from multibind.multibind import MultibindScanner
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

state_colors = [
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#ffff33',
]

state_map = {
    'if0': 'IF(0)',
    'ifh': r'IF(H$^+$)',
    'ifn': r'IF(Na$^+$)',
    'of0': 'OF(0)',
    'ofh': r'OF(H$^+$)',
    'ofn': r'OF(Na$^+$)',
}

sod_colors = [
    '#d0d1e6',
    '#a6bddb',
    '#74a9cf',
    '#3690c0',
    '#0570b0',
    '#045a8d',
    '#023858',
]


def main():
    img_dir = Path('.') / 'img'
    conf_dg_dir = img_dir / 'conf_dg'
    free_energy_dir = img_dir / 'free_energy'
    microstate_probs_dir = img_dir / 'msp'
    protonation_dg_dir = img_dir / 'prot_dg'
    sod_bind_dg_dir = img_dir / 'sod_bind_dg'

    conf_dg_dir.mkdir(parents=True, exist_ok=True)
    free_energy_dir.mkdir(parents=True, exist_ok=True)
    microstate_probs_dir.mkdir(parents=True, exist_ok=True)
    protonation_dg_dir.mkdir(parents=True, exist_ok=True)
    sod_bind_dg_dir.mkdir(parents=True, exist_ok=True)

    concentrations = dict()
    concentrations['Na+'] = [0.01, 0.1, 0.15, 0.2, 0.25]
    concentrations['H+'] = np.linspace(0, 15, 15 * 4)
    concentrations['H+'] = np.concatenate((concentrations['H+'], [8.0]))
    concentrations['H+'].sort()

    scanner = MultibindScanner('inputs/states.csv', 'inputs/graph.csv')
    scanner.run(concentrations, svd=True)

    pH = concentrations['H+']
    Na = concentrations['Na+']

    conf_dg = np.zeros((len(Na), len(pH)))
    prot_dg = np.zeros_like(conf_dg)
    sod_dg = np.zeros_like(conf_dg)

    for i, c_hp in enumerate(pH):
        for j, c_na in enumerate(Na):
            out_c = {'pH': c_hp, 'Na+': c_na}
            conf_dg[j, i], _ = scanner.effective_energy_difference('conf', 'inward', 'outward', **out_c)
            prot_dg[j, i], _ = scanner.effective_energy_difference('prot_bound', 'unbound', 'bound', **out_c)
            sod_dg[j, i], _ = scanner.effective_energy_difference('sod_bound', 'unbound', 'bound', **out_c)

    plot_msp(pH, Na, scanner, microstate_probs_dir)
    plot_free_energies(pH, Na, scanner, free_energy_dir)
    plot_dg(pH, Na, conf_dg, conf_dg_dir)
    plot_dg(pH, Na, prot_dg, protonation_dg_dir)
    plot_dg(pH, Na, sod_dg, sod_bind_dg_dir)

    breakpoint()

    edges = [('ifh', 'if0'),
             ('if0', 'ifn'),
             ('ifn', 'ofn'),
             ('ofn', 'of0'),
             ('of0', 'ofh'),
             ('ofh', 'ifh')]

    concentrations['H+'] = [8]
    scanner.run(concentrations, svd=False)

    dg = scanner.results.free_energy.sel({'pH': 8, 'Na+': 0.100})
    std_err = scanner.results.std_errors.sel({'pH': 8, 'Na+': 0.100})
    print(dg)

    values = []

    for i, j in edges:
        dg_i = dg.sel(state=i).values
        dg_j = dg.sel(state=j).values
        si = std_err.sel(state=i).values
        sj = std_err.sel(state=j).values
        std_err_ij = np.sqrt(si**2 + sj**2)
        diff = dg_j - dg_i
        print(f'{i} ({dg_i}) --> {j} ({dg_j}) => {diff} ± {std_err_ij}')
        values.append(diff)
    print(np.sum(values))


def plot_dg(pH, Na, data, outdir, ylim=[None, None]):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2))

    for i in range(len(Na)):
        ax.plot(pH, data[i, :], label=f'{Na[i]:0.3f} M', color=sod_colors[i])

    ax.axhline(0, ls='--', color='black', alpha=0.5)

    ax.legend(loc='best')
    ax.set_xlim([0, 15])

    filename = outdir / 'profile.pdf'
    plt.savefig(filename)
    plt.tight_layout()
    plt.cla()


def plot_free_energies(pH, Na, scanner, outdir):
    states = scanner.results.state.values

    ds = scanner.results

    for c_na in Na:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2))
        dg = ds.free_energy.sel({'Na+': c_na}).values

        for i in range(6):
            state_name = state_map[states[i]]
            ax.plot(pH, dg[i, :], label=state_name, color=state_colors[i])
        ax.legend(loc='best')

        ax.set_xlim([0, 15])

        sns.despine(offset=5, ax=ax)

        filename = outdir / f'na_{c_na}.pdf'
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()


def plot_msp(pH, Na, scanner, outdir):
    states = scanner.results.state.values

    ds = scanner.results

    for c_na in Na:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2))
        probs = ds.microstate_probs.sel({'Na+': c_na}).values

        for i in range(6):
            state_name = state_map[states[i]]
            ax.plot(pH, probs[i, :], label=state_name, color=state_colors[i])
        ax.legend(loc='best')

        ax.set_xlim([0, 15])
        ax.set_ylim([0, 1])

        sns.despine(offset=5, ax=ax)

        filename = outdir / f'na_{c_na}.pdf'
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()


if __name__ == "__main__":
    main()

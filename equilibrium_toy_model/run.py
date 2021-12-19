from multibind.multibind import MultibindScanner
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():

    img_dir = Path('.') / 'img'
    microstate_probs_dir = img_dir / 'msp'
    conf_dg_dir = img_dir / 'conf_dg'
    protonation_dg_dir = img_dir / 'prot_dg'
    sod_bind_dg_dir = img_dir / 'sod_bind_dg'

    microstate_probs_dir.mkdir(parents=True, exist_ok=True)
    conf_dg_dir.mkdir(parents=True, exist_ok=True)
    protonation_dg_dir.mkdir(parents=True, exist_ok=True)
    sod_bind_dg_dir.mkdir(parents=True, exist_ok=True)

    concentrations = dict()
    concentrations['Na+'] = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    concentrations['H+'] = np.linspace(0, 14, 14 * 4)

    scanner = MultibindScanner('inputs/states.csv', 'inputs/graph.csv')
    scanner.run(concentrations, svd=True)
    ds = scanner.results

    pH = concentrations['H+']
    Na = concentrations['Na+']
    states = scanner.results.state.values

    for c_na in concentrations['Na+']:
        probs = ds.microstate_probs.sel({'Na+': c_na}).values

        for i in range(6):
            plt.plot(pH, probs[i, :], label=states[i])
        plt.title(r'[Na$^+$] = {c_na} M'.format(c_na=c_na))
        plt.legend(loc='best')
        filename = microstate_probs_dir / f'na_{c_na}.pdf'
        plt.savefig(filename)
        plt.clf()

    conf_dg = np.zeros((len(Na), len(pH)))
    prot_dg = np.zeros_like(conf_dg)
    sod_dg = np.zeros_like(conf_dg)

    for i, c_hp in enumerate(pH):
        for j, c_na in enumerate(Na):
            out_c = {'pH': c_hp, 'Na+': c_na}
            conf_dg[j, i] = scanner.effective_energy_difference('conf', 'inward', 'outward', **out_c)
            prot_dg[j, i] = scanner.effective_energy_difference('prot_bound', 'unbound', 'bound', **out_c)
            sod_dg[j, i] = scanner.effective_energy_difference('sod_bound', 'unbound', 'bound', **out_c)

    for i in range(len(Na)):
        plt.plot(pH, conf_dg[i, :])
    plt.title(r'$\Delta G_{if \to of}$')
    filename = conf_dg_dir / 'if_to_of.pdf'
    plt.savefig(filename)
    plt.clf()

    for i in range(len(Na)):
        plt.plot(pH, prot_dg[i, :])
    plt.title(r'$\Delta G_{D \to P}$')
    filename = protonation_dg_dir / 'dep_to_pro.pdf'
    plt.savefig(filename)
    plt.clf()

    for i in range(len(Na)):
        plt.plot(pH, sod_dg[i, :])
    plt.title(r'$\Delta G_{bind}$')
    filename = sod_bind_dg_dir / 'no_sod_to_sod.pdf'
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    main()

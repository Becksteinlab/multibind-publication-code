from typing import List, Union
import pathlib
from multibind.nonequilibrium import rate_matrix
import numpy as np
import math


ifh = "IFH"
if0 = "IF0"
ifn = "IFNA"
ofh = "OFH"
of0 = "OF0"
ofn = "OFNA"

# Order for tables to output
ordering = [(ifh, ofh),
            (ofh, of0),
            (of0, ofn),
            (ofn, ifn),
            (ifn, if0),
            (if0, ifh),
            ]


def dG2pKa(dG : float, pH : float = 0.0) -> float:
    '''Convert Delta G to pKa given the pH.
    '''
    return pH - dG / math.log(10)


def format_name(name : str):
    '''Take in a state name and format it for latex.
    '''
    if "NA" in name:
        return name[0:2] + r"(Na$^+$)"
    if "H" in name:
        return name[0:2] + r"(H$^+$)"
    if "0" in name:
        return name[0:2] + r"(0)"


def table_from_entries(entries, bars=True, dG_err=None) -> None:
    '''Print out a latex table from rate entries.
    '''
    table = """
\\begin{table}[]
\\begin{tabular}{@{}lllll@{}}
\\toprule
State 1 & State 2 & $\\bar k_{12} \\pm \\bar \\sigma_{12}$ (s$^{-1}$) & $\\bar k_{21} \\pm \\bar \\sigma_{21}$ (s$^{-1}$) & $\\Delta \\bar G_{12} \\pm \\bar \\sigma_{\\Delta G}$ ($kT$) \\\\ \\midrule"""

    ordered_entries = []
    for i, (s1, s2) in enumerate(ordering):
        entry = None
        for j, e in enumerate(entries):
            if s1 == e[0] and s2 == e[1]:
                entry = e
                break
        if not entry:
            raise ValueError
        ordered_entries.append(entry)

    if dG_err:
        assert len(ordered_entries) == len(dG_err)

    for i, entry in enumerate(ordered_entries):
        s1, s2, k, var = entry
        bs1, bs2, bk, bvar = list(filter(lambda x: x[1] == s1 and x[0] == s2, entries))[0]

        dG = f"{-math.log(k / bk):0.3f}"
        dG_std = math.sqrt(var / k**2 + bvar / bk**2)

        if dG_err:
            dG_std = dG_err[i]

        append_value = f"{format_name(s1)} & {format_name(s2)} & ${k:0.2f} \\pm {var**0.5:0.2f}$ & ${bk:0.2f} \\pm {bvar**0.5:0.2f}$  & ${dG} \\pm {dG_std:0.3f}$ \\\\"

        table += f"\n{append_value}"

    table += r""" \bottomrule
\end{tabular}
\end{table}
"""

    if not bars:
        table = table.replace("\\bar", '')

    print(table)


def raw_rates_table(rate_file : Union[str, pathlib.Path]) -> None:
    '''Generate latex table from the raw rates file and print to screen.
    '''
    entries = []
    with open(rate_file, 'r') as F:
        for _line in F:
            line = _line.strip().split(",")
            if line[0] == 'state1':
                continue
            s1, s2, v, sigma = line
            entries.append((s1, s2, float(v), float(sigma)))

    table_from_entries(entries, bars=True)


def corrected_rates_table(rate_file : Union[str, pathlib.Path]) -> None:
    '''Generate latex table from the multibind corrected rates and print to screen.
    '''
    pH = 8
    Na = 0.1  # 100 mM

    c, k, kstd = rate_matrix(rate_file)

    states = c.states.values
    #  free_energies = c.g_mle
    g_std_err = c.std_errors

    new_graph = c.graph.copy()

    for index, data in new_graph.iterrows():
        state1, state2, value, variance, ligand, std = data
        # print(state1, state2, value, variance, ligand, std)

        if (state1[-1] == "H" and state2[-1] == "0") or (state1[-1] == "A" and state2[-1] == "0"):
            # backwards proton reaction
            new_graph.at[index, 'state1'] = state2
            new_graph.at[index, 'state2'] = state1
            new_graph.at[index, 'value'] = -value

            value = new_graph.value[index]
            state1 = new_graph.state1[index]
            state2 = new_graph.state2[index]

        if state1[-1] == "0" and state2[-1] == "H":
            new_graph.at[index, 'ligand'] = "H+"
            new_graph.at[index, 'value'] = dG2pKa(new_graph.value[index], pH)
            new_graph.at[index, 'variance'] = new_graph.variance[index] / np.log(10)**2

        if state1[-1] == "0" and state2[-1] == "A":
            new_graph.at[index, 'ligand'] = "Na+"
            new_graph.at[index, 'value'] = new_graph.value[index] + np.log(Na)

    dG_err = []
    entries = []
    for s1, s2 in ordering:
        s1_idx = list(filter(lambda x: x[1][0] == s1, enumerate(states)))[0][0]
        s2_idx = list(filter(lambda x: x[1][0] == s2, enumerate(states)))[0][0]

        entries.append((s1, s2, k[s1_idx, s2_idx], kstd[s1_idx, s2_idx]))
        entries.append((s2, s1, k[s2_idx, s1_idx], kstd[s2_idx, s1_idx]))
        dG_err.append(math.sqrt(g_std_err[s2_idx]**2 + g_std_err[s1_idx]**2))

    table_from_entries(entries, bars=False, dG_err=dG_err)


def main():
    rate_file = "inputs/md_rates.csv"

    raw_rates_table(rate_file)
    corrected_rates_table(rate_file)


if __name__ == "__main__":
    main()

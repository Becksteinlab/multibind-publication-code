from pathlib import Path

def load_data(filename):

    data = []

    with open(filename, 'r') as F:
        for _line in F:
            line = _line.strip().split(',')
            if line[0] == 'state1':
                continue
            s1, s2, pka, _, _, _ = line
            data.append((s1[1:], s2[1:], float(pka)))

    n_prot = lambda x: sum(map(int, x))

    data = sorted(data, key=lambda x: (n_prot(x[0]), x[0], x[1]))

    return data

if __name__ == "__main__":
    runs = Path('runs/')

    results = {}

    data = {}

    for root in runs.glob("*"):
        filename = root / 'inputs' / "best_graph.csv"
        if "None" in str(filename):
            name = 'simult'
        elif "center" in str(filename):
            name = 'central'
        elif "side" in str(filename):
            name = 'terminal'
        else:
            raise ValueError
        data[name] = load_data(filename)

    print(r"\begin{table}[]")
    print(r"    \centering")
    print(r"    \begin{tabular}{@{}lllll@{}}")
    print(r"        \toprule")
    print(r"        State 1   &   State 2   &   Central   &   Terminal   &   Simultaneous \\ \midrule")

    for i in range(12):

        s1, s2 = data['central'][i][:2]
        central = data['central'][i][2]
        terminal = data['terminal'][i][2]
        simult = data['simult'][i][2]

        line = f"\t{s1}   &   {s2}   &   {central:.1f}   &   {terminal:0.1f}   &   {simult:0.1f} \\\\"

        if i == 11:
            line += r" \bottomrule"

        print(line)
    print(r"    \end{tabular}")
    print(r"\end{table}")

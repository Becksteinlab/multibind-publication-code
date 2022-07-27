#!/usr/bin/env python

from multibind.multibind import MultibindScanner
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class Stepper(object):

    def __init__(self, groups, rundir, deventries=20, dev_max=0.1, dev_min=1e-3):

        self.logfilename = rundir / "steps.log"
        self.deventries = deventries

        # is this a continued run?
        if self.logfilename.exists():
            self.read_log()
        else:
            self.deviations = []
            self.success_count = 0
            self.total_count = 0

        self.logger = logging.getLogger('MC')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(rundir / "steps.log")
        self.logger.addHandler(fh)
        self.load_nmr_data(normalize=True)
        self.groups = groups
        self.imgdir = rundir / "img"
        self.inputdir = rundir / "inputs"
        self.repeated_failures = 0
        self.dev_max = dev_max
        self.dev_min = dev_min

        input_states = self.inputdir / "states.csv"
        self.input_graph_best = self.inputdir / "best_graph.csv"

        if not self.input_graph_best.exists():
            input_graph = self.inputdir / "graph.csv"
        else:
            input_graph = self.input_graph_best

        self.scanner = MultibindScanner(input_states, input_graph)

    def read_log(self):
        data = []
        with open(self.logfilename, 'r') as F:
            for line in F:
                if line.startswith("STEP"):
                    data.append(line.strip())
        last_entry = data[-1].split(" ")
        self.total_count = int(last_entry[0].split("=")[1])
        self.success_count = int(last_entry[1].split("=")[1])
        self.deviations = [float(last_entry[5].split("=")[1])] * self.deventries

    def proton_uptake(self, site=None):

        msp = self.scanner.results.microstate_probs
        nprot = []

        index_map = {'left': 1,
                     'center': 2,
                     'right': 3,
                     'side': 1}

        for s in self.scanner.c.states.name.values:
            nprot.append(int(s[index_map[site]]))

        nprot = np.array(nprot)

        pH = self.scanner.results.pH.values

        expected = []

        for p in pH:
            prob = (nprot * msp.sel({"pH": p}).values).sum()
            expected.append(prob)

        return np.array(pH), np.array(expected)

    def load_nmr_data(self, normalize=True):
        nmr_data_dir = Path("data")
        center = np.genfromtxt(nmr_data_dir / "nmr_center.csv", delimiter=",", dtype=np.float64)
        side = np.genfromtxt(nmr_data_dir / "nmr_side.csv", delimiter=",", dtype=np.float64)

        if normalize:
            center[:, 0] = center[:, 0] - center[:, 0].min()
            center[:, 0] = center[:, 0] / center[:, 0].max()
            side[:, 0] = side[:, 0] - side[:, 0].min()
            side[:, 0] = side[:, 0] / side[:, 0].max()

        self.center = center
        self.side = side

    def create_pH_range(self):

        center = self.center
        side = self.side

        pH = list(np.linspace(0, 14, 100))

        for i in range(center.shape[0]):
            if center[i][1] not in pH:
                pH.append(center[i][1])

        for i in range(side.shape[0]):
            if side[i][1] not in pH:
                pH.append(side[i][1])

        pH = list(set(pH))
        pH.sort()

        center_mask = []
        side_mask = []

        for ph in pH:
            if ph in center[:, 1]:
                center_mask.append(True)
            else:
                center_mask.append(False)

            if ph in side[:, 1]:
                side_mask.append(True)
            else:
                side_mask.append(False)

        self.pH = pH
        self.center_mask = center_mask
        self.side_mask = side_mask
        self.concentrations = {'pH': self.pH}

    def results_rmsd(self, target=None):

        center_mask = self.center_mask
        side_mask = self.side_mask
        center_dataset = self.center
        side_dataset = self.side

        scanner_pH, uptake_center = self.proton_uptake(site="center")
        scanner_pH, uptake_side = self.proton_uptake(site="side")

        filtered_uptake_center = uptake_center[center_mask]
        filtered_uptake_side = uptake_side[side_mask]

        nmr_values_center = center_dataset[:, 0]
        nmr_values_side = side_dataset[:, 0]

        if target == "side":
            filtered_uptake = filtered_uptake_side
            nmr_values = nmr_values_side
        elif target == "center":
            filtered_uptake = filtered_uptake_center
            nmr_values = nmr_values_center
        else:
            filtered_uptake = np.concatenate([filtered_uptake_center, filtered_uptake_side])
            nmr_values = np.concatenate([nmr_values_center, nmr_values_side])

        RMSD = np.sqrt(np.mean((nmr_values - filtered_uptake)**2))

        return RMSD

    def step(self, target=None):

        scanner = self.scanner

        old_graph = scanner.c.graph.copy()
        old_rmsd = self.results_rmsd(target=target)
        old_results = scanner.results.copy()

        standard_deviation = self.dev_max

        deviations = np.random.uniform(-0.2, 0.2, size=len(self.groups))
        for deviation, group in zip(deviations, self.groups):

            scanner.c.graph.loc[group[0][2] - 1, ['value']] += deviation

            for g in group:
                scanner.c.graph.loc[g[2] - 1, ['value']] = scanner.c.graph.loc[group[0][2] - 1, ['value']]

        scanner.run(self.concentrations)

        new_rmsd = self.results_rmsd(target=target)
        rmsd = self.results_rmsd()
        rmsd_side = self.results_rmsd(target="side")
        rmsd_center = self.results_rmsd(target="center")

        success = False

        if new_rmsd <= old_rmsd:
            for i in scanner.c.graph.iterrows():
                s1 = i[1].state1
                s2 = i[1].state2

                dG = float(scanner.results.dGs.sel(pH=0, state_i=s1, state_j=s2).values)
                pKa = -dG / np.log(10)

                scanner.c.graph.loc[i[0], "value"] = pKa
                scanner.c.graph.loc[i[0], "variance"] = 1.0
            success = True
            self.success_count += 1
            self.repeated_failures = 0
            self.deviations.append(abs(deviation))
        else:
            scanner.c.graph = old_graph.copy()
            scanner.results = old_results.copy()
            success = False
            self.repeated_failures += 1
        mean_dev = np.mean(self.deviations[-self.deventries:])
        self.logger.info(f"STEP={self.total_count} ACCEPTED={self.success_count} RSIDE={rmsd_side} RCENTER={rmsd_center} RTOT={rmsd} MDEV={mean_dev} STD={standard_deviation}")

        return success

    def run(self, target=None, eps=1e-3, allowed_failures=1000):

        self.create_pH_range()
        self.scanner.run(self.concentrations)

        plt.ion()
        figure, ax = plt.subplots(figsize=[4, 3])
        sns.despine(ax=ax, offset=5)

        uptake_line_center, = ax.plot(*self.proton_uptake(site="center"), color='red')
        uptake_line_side, = ax.plot(*self.proton_uptake(site="side"), color='blue')

        ax.plot(self.center[:, 1], self.center[:, 0], '.', color='red')
        ax.plot(self.side[:, 1], self.side[:, 0], '.', color='blue')

        ax.set_ylabel(r"$\langle x \rangle$")
        ax.set_xlabel(r"pH")

        ax.set_ylim([0, 1])
        ax.set_xlim([0, 14])

        plt.tight_layout()
        plt.savefig(self.imgdir / f"{self.success_count:04d}.png")

        while True:
            success = self.step(target=target)
            self.total_count += 1

            if not success:
                if self.repeated_failures >= allowed_failures:
                    self.logger.info(f"Failed to find a valid move after {allowed_failures}")
                    break
                continue

            self.scanner.c.write_graph(self.input_graph_best)

            pH, uptake_center = self.proton_uptake(site="center")
            pH, uptake_side = self.proton_uptake(site="side")

            uptake_line_center.set_xdata(pH)
            uptake_line_center.set_ydata(uptake_center)

            uptake_line_side.set_xdata(pH)
            uptake_line_side.set_ydata(uptake_side)

            figure.canvas.draw()
            figure.canvas.flush_events()

            plt.savefig(self.imgdir / f"{self.success_count:05d}.png")

            if np.mean(self.deviations[-self.deventries:]) < eps and len(self.deviations) >= self.deventries:
                self.logger.info("Converged.")
                break


def main():

    import uuid
    import shutil
    from sys import argv

    runid = uuid.uuid4()

    if len(argv) == 2:
        site = argv[1]

        # would like to support continuation
        if site not in ('side', 'center'):
            rundir = Path(site)
            site = str(rundir).split('-')[-1]
        else:
            rundir = Path("runs") / f"{runid}-{site}"

    else:
        site = None
        rundir = Path("runs") / f"{runid}-{site}"

    runinputs = rundir / "inputs"
    runimg = rundir / "img"

    runimg.mkdir(exist_ok=True, parents=True)
    runinputs.mkdir(exist_ok=True, parents=True)

    statefile = Path() / "multibind_inputs" / "states.csv"
    graphfile = Path() / "multibind_inputs" / "graph.csv"

    if not (runinputs / "states.csv").exists():
        shutil.copy(statefile, runinputs)
    if not (runinputs / "graph.csv").exists():
        shutil.copy(graphfile, runinputs)

    groups = [[("s000", "s010", 11)],
              [("s000", "s001", 10), ("s000", "s100", 9)],
              [("s100", "s110", 8), ("s001", "s011", 7)],
              [("s100", "s101", 5), ("s001", "s101", 6)],
              [("s010", "s110", 1), ("s010", "s011", 2)],
              [("s110", "s111", 4), ("s011", "s111", 3)],
              [("s101", "s111", 12)],
              ]

    stepper = Stepper(groups, rundir)
    stepper.run(target=site)


if __name__ == "__main__":
    main()

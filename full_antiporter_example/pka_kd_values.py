"""Calculate the binding constants from the equilibrium model.

Hard coding in the results from the equilibrium calculation as not to muddy the
waters. If the model changes then this will also need to be changed.
"""

import numpy as np


class State(object):
    """Class to keep track of conformation and bound ligand."""

    def __init__(self, conf: str, ligand: str):
        """Constructor for State."""
        assert conf in ["IF", "OF"]
        assert ligand in ["Na+", "Empty", "H+"]
        self.conf = conf
        self.ligand = ligand

    @property
    def is_inward(self) -> bool:
        """True if IF."""
        return self.conf == "IF"

    @property
    def is_outward(self) -> bool:
        """True if OF."""
        return not self.is_inward

    @property
    def has_sodium(self) -> bool:
        """True if sodium bound."""
        return self.ligand == "Na+"

    @property
    def has_proton(self) -> bool:
        """True if proton bound."""
        return self.ligand == "H+"

    @property
    def is_empty(self) -> bool:
        """True if nothing is bound."""
        return self.ligand == "Empty"

    def __eq__(self, other) -> bool:
        """Compare states."""
        same_conf = self.conf == other.conf
        same_ligand = self.ligand == other.ligand
        return same_conf and same_ligand

    def __str__(self) -> str:
        """String represenation of a state."""
        ligand = "0" if self.is_empty else self.ligand
        return f"{self.conf}({ligand})"


IF = "IF"
OF = "OF"
H = "H+"
Na = "Na+"
Empty = "Empty"

rates = (
    # OF
    (State(OF, Empty), State(OF, H),           194.45,    3.58),
    (State(OF, H),     State(OF, Empty),      6007.89,    4.14),
    (State(OF, Empty), State(OF, Na),    320043930.33,  395.36),
    (State(OF, Na),    State(OF, Empty), 167088412.30, 3259.91),
    # IF
    (State(IF, Empty), State(IF, H),           215.89,    4.12),
    (State(IF, H),     State(IF, Empty),      7888.67,   31.22),
    (State(IF, Empty), State(IF, Na),    319948241.25, 6504.61),
    (State(IF, Na),    State(IF, Empty),  76739017.89, 2677.41),
    # CONF
    (State(IF, Na),    State(OF, Na),         4990.12,   18.53),
    (State(IF, Na),    State(OF, Na),         8006.22,   12.17),
    (State(IF, H),     State(OF, H),          8006.22,   12.17),
    (State(IF, H),     State(OF, H),          4990.12,   18.53),
)


c_H = 1e-8    # pH = 8
c_Na = 0.100  # [Na^+] = 100 mM


def dG_from_rates(State1, State2, rates):
    """Calculate the free energy difference between two states.

    State1 is the starting state and State2 is the ending rate.
    """
    for values in rates:
        s1, s2, k, sigma = values
        if s1 == State1 and s2 == State2:
            forward = (k, sigma)
        if s1 == State2 and s2 == State1:
            backward = (k, sigma)
    dG = -np.log(forward[0] / backward[0])
    return dG


def dG_to_pKa(dG, pH=-np.log10(c_H)):
    """Calculate the pKa from a free energy difference and system pH."""
    return pH - dG / np.log(10)


def dG_to_kD(dG, c_Na=c_Na):
    """Calculate the kD from a free energy difference and sodium concentration."""
    return np.exp(dG)*c_Na


def print_second_order_rate_constants(rates):
    """Over all rates, print the second order rate constants."""
    for s1, s2, rate, sigma in rates:
        if s1.is_empty and s2.has_proton:
            rate /= c_H
            sigma /= c_H
        if s1.is_empty and s2.has_sodium:
            rate /= c_Na
            sigma /= c_Na
        print(f"{str(s1):8} -> {str(s2):8} ==> {rate} +- {sigma}")


def main():
    """Print resulting K_D and pKas from the equilibrium model.

    The forward and reverse rates are hardcoded globally.
    """
    IFNA = State(IF, Na)
    IFH = State(IF, H)
    IF0 = State(IF, Empty)

    OFNA = State(OF, Na)
    OFH = State(OF, H)
    OF0 = State(OF, Empty)

    IFpKa = dG_to_pKa(dG_from_rates(IF0, IFH, rates))
    OFpKa = dG_to_pKa(dG_from_rates(OF0, OFH, rates))
    IFkD = dG_to_kD(dG_from_rates(IF0, IFNA, rates))
    OFkD = dG_to_kD(dG_from_rates(OF0, OFNA, rates))

    print("===== pKa and kD =====")
    print(f"IF pKa: {IFpKa}")
    print(f"OF pKa: {OFpKa}")
    print(f"IF kD: {1000 * IFkD} mM")
    print(f"OF kD: {1000 * OFkD} mM")
    print("======================")
    print_second_order_rate_constants(rates)


if __name__ == "__main__":
    main()

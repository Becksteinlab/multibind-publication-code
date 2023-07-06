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

none_rates = (
    (State(IF, H), State(OF, H), 8006.223169408338, 12.57314756359655),
    (State(OF, H), State(IF, H), 4990.015244309923, 19.140787642841556),
    (State(OF, H), State(OF, Empty), 6007.98133761928, 4.139689625469715),
    (State(OF, Empty), State(OF, H), 194.45150148101385, 3.5841640766354605),
    (State(OF, Empty), State(OF, Na), 320043930.3290041, 395.36290105497517),
    (State(OF, Na), State(OF, Empty), 167088412.30087355, 3259.9134983152007),
    (State(OF, Na), State(IF, Na), 8006.223169408335, 12.169381784880281),
    (State(IF, Na), State(OF, Na), 4990.01524430993, 18.526112996833973),
    (State(IF, Na), State(IF, Empty), 76739017.89163141, 2677.4053171912883),
    (State(IF, Empty), State(IF, Na), 319948241.25336146, 6504.605667410289),
    (State(IF, Empty), State(IF, H), 215.89117545131967, 4.123301815264508),
    (State(IF, H), State(IF, Empty), 7888.666130672948, 31.218756052842668),
    (State(OF, Empty), State(IF, Empty), 0.0, 0.0),
    (State(IF, Empty), State(OF, Empty), 0.0, 0.0),
)

small_rates = (
    (State(IF, H), State(OF, H), 8006.205593777888, 12.569971591604132),
    (State(OF, H), State(IF, H), 4990.043521784862, 19.13587689337721),
    (State(OF, H), State(OF, Empty), 6007.959673036285, 4.120731730091776),
    (State(OF, Empty), State(OF, H), 194.46702398836024, 3.567464723819803),
    (State(OF, Empty), State(OF, Na), 320044062.77480483, 392.94891103734113),
    (State(OF, Na), State(OF, Empty), 167079476.74613082, 3240.1849035495575),
    (State(OF, Na), State(IF, Na), 8006.242380110126, 12.146920571454022),
    (State(IF, Na), State(OF, Na), 4989.984335655457, 18.491999128412196),
    (State(IF, Na), State(IF, Empty), 76760846.63180155, 2645.7557493785434),
    (State(IF, Empty), State(IF, Na), 319948058.79832435, 6426.80159525659),
    (State(IF, Empty), State(IF, H), 215.84542374567658, 4.076852122475119),
    (State(IF, H), State(IF, Empty), 7889.0151036064735, 30.86985335989601),
    (State(OF, Empty), State(IF, Empty), 99.70388815206579, 2.401037558616935),
    (State(IF, Empty), State(OF, Empty), 135.2183394847137, 3.789998694486533),
)

large_rates = (
    (State(IF, H), State(OF, H), 8006.002877514489, 12.533154300899445),
    (State(OF, H), State(IF, H), 4990.369640303549, 19.07895672839883),
    (State(OF, H), State(OF, Empty), 6007.709441691196, 3.8787346954443973),
    (State(OF, Empty), State(OF, H), 194.64612022231438, 3.354863361763936),
    (State(OF, Empty), State(OF, Na), 320045588.7058398, 361.3535690488545),
    (State(OF, Na), State(OF, Empty), 166976457.88334376, 2981.519619855568),
    (State(OF, Na), State(IF, Na), 8006.463912125178, 11.878496964270255),
    (State(IF, Na), State(OF, Na), 4989.6278679430425, 18.08426456609654),
    (State(IF, Na), State(IF, Empty), 77013041.25217529, 2147.63862169223),
    (State(IF, Empty), State(IF, Na), 319945943.9022867, 5208.28444500144),
    (State(IF, Empty), State(IF, H), 215.31805431858123, 3.3692238082462045),
    (State(IF, H), State(IF, Empty), 7893.0254385417575, 25.53827745474294),
    (State(OF, Empty), State(IF, Empty), 999.6291938053557, 5.654090509498371),
    (State(IF, Empty), State(OF, Empty), 1350.2745135849646, 8.920341089700932),
)

all_rates = {"small": small_rates, "none": none_rates, "large": large_rates}

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
    import sys

    try:
        target = sys.argv[1]
    except:
        target = "small"

    rates = all_rates[target]
    
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

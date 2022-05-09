#!/usr/bin/env python

import math
import chem

# | IF(H+)  | IF(0)   | -3.55 |  |  |
# | IF(0)   | IF(Na+) | -1.33 |  |  |
# | IF(Na+) | OF(Na+) | 0.5   |  |  |
# | OF(Na+) | OF(0)   | 0.65  |  |  |
# | OF(0)   | OF(H+)  | 3.23  |  |  |
# | OF(H+)  | IF(H+)  | 0.50  |  |  |


def koff_from_kon(kon, dG):
    return kon*math.exp(dG)


def kon_from_koff(koff, dG):
    return koff*math.exp(dG)


kon = 10
koff = 20

proton_on, _ = chem.rates(0, 7, 10, "disk", False)
dG = [-3.55, -1.33, 0.5, 0.65, 3.23, 0.50]

# IFH --> IF0
print(f"IF(H+)\tIF(0)\t{kon:.2f}\t{proton_on:.2f}")
# IF0 --> IFNA
print(f"IF(0)\tIF(Na+)\t{kon:.2f}\t{koff:.2f}")
# IFNA --> OFNA
print(f"IF(Na+)\tOF(Na+)\t{kon:.2f}\t{koff:.2f}")
# OFNA --> OF0
print(f"OF(Na+)\tOF(0)\t{kon:.2f}\t{koff:.2f}")
# OF0 --> OFH
print(f"OF(0)\tOF(H+)\t{proton_on:.2f}\t{koff:.2f}")
# OFH --> IFH
print(f"OF(H+)\tIF(H+)\t{kon:.2f}\t{koff:.2f}")

chem.rates(0, 1, 10, "disk", True)

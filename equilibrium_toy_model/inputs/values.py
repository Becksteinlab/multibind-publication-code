import numpy as np


def pKa(pH, dG):
    return pH - dG / np.log(10)


def standard_state(Na, dG):
    return dG + np.log(Na)


def main():
    pH = 8
    Na = 0.1

    print(f'IF0 --> IFH+ = {pKa(pH, -1)}')
    print(f'IF0  --> IFN = {standard_state(Na, -3)}')
    print('IFN  --> OFN = 0.5')
    print(f'OF0  --> OFN = {standard_state(Na, -2.5)}')
    print(f'OF0 --> OFH+ = {pKa(pH, -1.5)}')
    print('OFH+  --> IFH+ = 0.5')


if __name__ == "__main__":
    main()

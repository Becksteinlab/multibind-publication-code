import math

Na = 6.02e23  # Avogadros number mol^-1
D = 9.3e11  # proton diffusion factor cm^2/s


def dG2pKa(deltaG, pH):
    return pH - deltaG / math.log(10)


def sodkon(koff, conc, dG):
    return koff / conc * math.exp(-dG)


def proton_k_off(kon, pKa):
    return kon * 10**-pKa


def rates(pH, pKa, radius, capture_method="disk", verbose=False):

    if capture_method == "hemi":  # if using the spherical coordinates
        factor = 2 * math.pi
    elif capture_method == "disk":  # if using the cylindrical coordinates
        factor = 4

    angstrom3ToL = 1e-27  # Conversion from cubic angstroms (R*D) to liters
    flux = factor * radius * D  # Å^3 / s
    protonConcentration = 10**-pH  # M, mol / L

    konprime = flux * angstrom3ToL * Na  # Pseudo-first-order rate constant
    kon = konprime * protonConcentration
    koff = konprime * 10**-pKa  # NOT DEPENDENT ON PH

    if verbose:
        print(f"Geometry: {capture_method}")
        print(f"pH  = {pH:.1f}")
        print(f"pKa = {pKa:.1f}")
        print(f"R   = {radius:.1f}")

        print(f"k_on  ==> {kon:12.5f} particles / s")
        print(f"k_off ==> {koff:12.5f} particles / s")

    return kon, koff

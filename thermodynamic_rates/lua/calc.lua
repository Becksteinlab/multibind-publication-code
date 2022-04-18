require("math")

Na = 6.02e23 -- Avogadros number
D = 9.3e11 -- Diffusion coeffienct Å^2 / s

local M = {}

function M.dG2pKa(deltaG, pH)
    return pH - deltaG / math.log(10)
end

function M.sodkon(koff, conc, dG)
    return koff / conc * math.exp(-dG)
end

function M.proton_k_off(kon, pKa)
    return kon * 10^-pKa
end

function M.rates(pH, pKa, radius, capture_method, verbose)

    local factor

    if capture_method == "hemi" then -- if using the spherical coordinates
        factor = 2 * math.pi
    elseif capture_method == "disk" then -- if using the cylindrical coordinates from Crank
        factor = 4
    end

    local angstrom3ToL = 1e-27 -- Conversion from cubic angstroms (R*D) to liters
    local flux = factor * radius * D -- Å^3 / s
    local protonConcentration = 10^-pH -- M, mol / L

    local konprime = flux * angstrom3ToL *  Na -- Pseudo-first-order rate constant
    local kon = konprime * protonConcentration
    local koff = konprime * 10^(-pKa) -- NOT DEPENDENT ON PH

    if verbose then
        print(string.format("Geometry: %s", capture_method))
        print(string.format("pH  = %.1f", pH))
        print(string.format("pKa = %.1f", pKa))
        print(string.format("R   = %.1f", radius))

        print(string.format("k_on  ==> %12.5f particles / s", kon))
        print(string.format("k_off ==> %12.5f particles / s", koff))
    end

    return kon, koff
end

return M

--print("--- Regression ---")
--rates(7, 9, 5, "hemi") -- control, should be 17.588 for the koff
--print(string.rep("-", 36))
--
--print("--- Disk ---")
--rates(7, 9, 5, "disk")
--print(string.rep("-", 36))

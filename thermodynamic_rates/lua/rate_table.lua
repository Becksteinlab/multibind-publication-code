#!/usr/bin/env lua

require("math")

local chem = require("calc")

-- | IF(H+)  | IF(0)   | -3.55 |  |  |
-- | IF(0)   | IF(Na+) | -1.33 |  |  |
-- | IF(Na+) | OF(Na+) | 0.5   |  |  |
-- | OF(Na+) | OF(0)   | 0.65  |  |  |
-- | OF(0)   | OF(H+)  | 3.23  |  |  |
-- | OF(H+)  | IF(H+)  | 0.50  |  |  |

local function koff_from_kon(kon, dG)
    return kon*math.exp(dG)
end

local function kon_from_koff(koff, dG)
    return koff*math.exp(dG)
end

local kon = 10
local koff = 20

local proton_on, _ = chem.rates(0, 7, 10, "disk", false)
local dG = {-3.55, -1.33, 0.5, 0.65, 3.23, 0.50}

print(string.format("IF(H+)\tIF(0)\t%.2f\t%.2f", kon, proton_on))
print(string.format("IF(0)\tIF(Na+)\t%.2f\t%.2f", kon, koff))
print(string.format("IF(Na+)\tOF(Na+)\t%.2f\t%.2f", kon, koff))
print(string.format("OF(Na+)\tOF(0)\t%.2f\t%.2f", kon, koff))
print(string.format("OF(0)\tOF(H+)\t%.2f\t%.2f", proton_on, koff))
print(string.format("OF(H+)\tIF(H+)\t%.2f\t%.2f", kon, koff))

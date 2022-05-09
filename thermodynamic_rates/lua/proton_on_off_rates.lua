#!/usr/bin/env lua

local chem = require("calc")

print("--- Regression ---")
chem.rates(7, 9, 5, "hemi") -- control, should be 17.588 for the koff
print(string.rep("-", 36))

print("--- Disk ---")
chem.rates(7, 9, 5, "disk")
print(string.rep("-", 36))

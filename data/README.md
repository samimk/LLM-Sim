# Simulation Data

Place simulation input files in this directory.

## Supported file types

- **MATPOWER `.m` files** — Network case definitions (bus, generator, branch data)
- **`.gic` files** — Geomagnetically Induced Current data
- **Contingency files** (`.cont`) — For security-constrained analysis (SCOPFLOW)
- **Load profiles** (`*_load_P.csv`, `*_load_Q.csv`) — Time-series load data for TCOPFLOW
- **Wind profiles** (`*_wind.csv`) — Wind generation profiles for TCOPFLOW (optional)
- **Scenario files** — Stochastic scenarios (SOPFLOW)

## File naming conventions

### Contingency files (SCOPFLOW)
Name the file to match the base case: `<casename>.cont`
- `case9mod.m` → `case9.cont`
- `case_ACTIVSg200.m` → `case_ACTIVSg200.cont`

### Load profiles (TCOPFLOW)
Name profile files using the case prefix: `<casename>_load_P.csv` and `<casename>_load_Q.csv`

The launcher auto-selects profiles matching the base case using layered fallback:
1. Exact prefix: `case9mod_load_P.csv`
2. Strip known suffixes (e.g., "mod"): `case9_load_P.csv`
3. Fallback: all available profiles

Example:
- `case9mod.m` → `case9_load_P.csv` + `case9_load_Q.csv`

### Wind profiles (TCOPFLOW, optional)
- `<casename>_wind.csv` — e.g., `case9_wind.csv`

## Example

The ACTIVSg200 synthetic test case is a good starting point:

```
data/
├── case_ACTIVSg200.m       # Network definition
├── case_ACTIVSg200.gic     # GIC data
└── case_ACTIVSg200.cont   # Contingency file (SCOPFLOW)
```

For TCOPFLOW, add load profiles:

```
data/
├── case9mod.m              # Network definition
├── case9_load_P.csv        # Active load profile
├── case9_load_Q.csv        # Reactive load profile
└── case9_wind.csv          # Wind profile (optional)
```

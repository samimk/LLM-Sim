# Simulation Data

Place simulation input files in this directory.

## Supported file types

- **MATPOWER `.m` files** — Network case definitions (bus, generator, branch data)
- **`.gic` files** — Geomagnetically Induced Current data
- **Contingency files** — For security-constrained analysis (SCOPFLOW)
- **Load profiles** — Time-series load data (TCOPFLOW)
- **Scenario files** — Stochastic scenarios (SOPFLOW)

## Example

The ACTIVSg200 synthetic test case is a good starting point:

```
data/
├── case_ACTIVSg200.m
├── case_ACTIVSg200.gic
└── ACTIVSg200_contingencies.con
```

These files can be obtained from the ExaGO test data or the ACTIVSg repository.

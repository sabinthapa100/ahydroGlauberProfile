# aHydro initial condition visualizer (Version 2)

Author: Sabin Thapa <sthapa3@kent.edu>

This folder contains a small reusable Python package (`src/`) + two notebooks:

- `notebooks/pPb_vs_dAu.ipynb` (small systems)
- `notebooks/PbPb_vs_AuAu.ipynb` (large systems)

## What’s implemented

**Optical Glauber (baseline):**
- AA, pA, dA wounded + binary densities
- overlap integral, `P_inel(b)`, and **b-percentile** centrality tables

**aHydro-style tilted initial energy density:**
- ρ(ς): plateau + Gaussian tails
- g(ς): tilt with y_N(√s)
- ε(x,y,ς) shape from (wounded + binary) mixing

**MC Glauber (small-system centrality):**
- pA / dA (and a basic AA option)
- black-disk collisions using σ_NN
- entropy proxy S with optional Gamma weights
- centrality by **S-quantiles** (multiplicity-like)

## How to use

Open either notebook and run top-to-bottom.

If you copy this into your repo, keep:
- `src/` in the same directory level as `notebooks/` **or**
- add `../` to `sys.path` in the notebooks.

## Scientific note (important)

In pA and dA, experimental centrality is typically multiplicity-based.
Optical b-binning is still a clean geometry baseline, but **S-based (MC) centrality**
is what you want for realistic comparisons.

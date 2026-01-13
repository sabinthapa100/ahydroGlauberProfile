"""src/initial_ahydro.py
Author: Sabin Thapa <sthapa3@kent.edu>

aHydro-style "tilted" initial energy density ε(x,y,ς) at τ0.

We implement the *shape* (up to a global constant normalization):
  ε ∝ (1-χ) ρ(ς) [ W_proj g(ς) + W_targ g(-ς) ] + χ ρ(ς) C

where:
- W_proj, W_targ are optical wounded densities in the transverse plane
- C is the optical binary collision density
- ρ(ς): plateau + Gaussian tails
- g(ς): tilt function determined by y_N(√s)

Centrality dependence enters only through (W_proj, W_targ, C), i.e. geometry.
If you peak-normalize each slice, you erase amplitude changes by construction.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .physics import y_nucleon


@dataclass(frozen=True)
class TiltedParams:
    sNN_GeV: float
    chi: float
    Delta_zeta: float
    sigma_zeta: float


def rho_longitudinal(zeta: np.ndarray, Delta: float, sigma: float) -> np.ndarray:
    """Plateau (ρ=1) for |ζ|≤Δ and Gaussian tails beyond."""
    x = np.maximum(np.abs(zeta) - float(Delta), 0.0)
    return np.exp(-(x * x) / (2.0 * float(sigma) ** 2))


def g_tilt(zeta: np.ndarray, yN: float) -> np.ndarray:
    """Piecewise-linear tilt function with saturation outside ±yN."""
    out = np.empty_like(zeta, dtype=float)
    out[zeta < -yN] = 0.0
    mid = (zeta >= -yN) & (zeta <= yN)
    out[mid] = (zeta[mid] + yN) / (2.0 * yN)
    out[zeta > yN] = 1.0
    return out


def epsilon_tilted_3d(
    zeta: np.ndarray,
    W_proj_xy: np.ndarray,
    W_targ_xy: np.ndarray,
    C_xy: np.ndarray,
    pars: TiltedParams,
) -> np.ndarray:
    """Return ε(ζ,y,x) up to an overall constant normalization.

    Shapes:
      - zeta: (Nz,)
      - W_proj_xy, W_targ_xy, C_xy: (Ny, Nx)
      - output: (Nz, Ny, Nx)
    """
    yN = y_nucleon(pars.sNN_GeV)
    rho = rho_longitudinal(zeta, pars.Delta_zeta, pars.sigma_zeta)
    gP = g_tilt(zeta, yN)
    gM = g_tilt(-zeta, yN)

    rho3 = rho[:, None, None]
    gP3 = gP[:, None, None]
    gM3 = gM[:, None, None]

    wounded = (1.0 - pars.chi) * rho3 * (W_proj_xy[None, :, :] * gP3 + W_targ_xy[None, :, :] * gM3)
    binary = pars.chi * rho3 * C_xy[None, :, :]
    return wounded + binary


def epsilon_tilted_slice(
    zeta0: float,
    zeta_grid: np.ndarray,
    W_proj_xy: np.ndarray,
    W_targ_xy: np.ndarray,
    C_xy: np.ndarray,
    pars: TiltedParams,
) -> np.ndarray:
    """Compute ε(x,y,ζ0) without building the full 3D array."""
    # nearest index
    iz = int(np.argmin(np.abs(zeta_grid - float(zeta0))))
    z = np.array([zeta_grid[iz]], dtype=float)

    eps = epsilon_tilted_3d(z, W_proj_xy, W_targ_xy, C_xy, pars)[0]
    return eps

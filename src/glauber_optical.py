"""src/glauber_optical.py
Author: Sabin Thapa <sthapa3@kent.edu>

Optical Glauber geometry for AA, pA, dA:
- thickness functions T_A(x,y)
- wounded densities W_A, W_B
- binary density C
- overlap T_AB(b) and inelastic probability P_inel(b)
- centrality binning using b percentiles (geometry-only baseline)

Important for interpretation:
- In small systems (pA, dA), experimental "centrality" is not purely geometric.
  Optical b-binning is still useful as a clean baseline, but matching data requires
  multiplicity/entropy-based selection (see src/glauber_mc.py).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from .physics import mb_to_fm2
from .geometry import SphericalWoodsSaxon, GeneralizedGaussianProton, DeuteronThickness


@dataclass(frozen=True)
class Grid2D:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray


def make_grid2d(*, xmax: float = 12.0, nx: int = 241, ymax: float = 12.0, ny: int = 241) -> Grid2D:
    x = np.linspace(-xmax, xmax, nx)
    y = np.linspace(-ymax, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return Grid2D(x=x, y=y, X=X, Y=Y)


def integrate_2d(fxy: np.ndarray, grid: Grid2D) -> float:
    return float(np.trapz(np.trapz(fxy, grid.x, axis=1), grid.y, axis=0))


def wounded_density(TA: np.ndarray, TB: np.ndarray, sigma_fm2: float) -> np.ndarray:
    """Optical wounded density: TA * (1 - exp(-σ TB))."""
    return TA * (1.0 - np.exp(-sigma_fm2 * TB))


def binary_density(TA: np.ndarray, TB: np.ndarray, sigma_fm2: float) -> np.ndarray:
    """Optical binary density: σ TA TB."""
    return sigma_fm2 * TA * TB


SystemKind = Literal["AA", "pA", "dA"]


class OpticalGlauber:
    """Optical Glauber model for AA, pA, dA."""

    def __init__(
        self,
        kind: SystemKind,
        *,
        sigmaNN_mb: float,
        nuc_target: SphericalWoodsSaxon,
        nuc_projectile: Optional[SphericalWoodsSaxon] = None,
        proton_projectile: Optional[GeneralizedGaussianProton] = None,
        deuteron_projectile: Optional[DeuteronThickness] = None,
    ):
        self.kind = kind
        self.sigma_fm2 = mb_to_fm2(sigmaNN_mb)
        self.nuc_t = nuc_target
        self.nuc_p = nuc_projectile
        self.proton = proton_projectile
        self.deuteron = deuteron_projectile

        if kind == "AA" and nuc_projectile is None:
            raise ValueError("AA requires nuc_projectile.")
        if kind == "pA" and proton_projectile is None:
            raise ValueError("pA requires proton_projectile.")
        if kind == "dA" and deuteron_projectile is None:
            raise ValueError("dA requires deuteron_projectile.")

    # ---- thickness helpers ----

    def _TA_TB(self, b: float, grid: Grid2D):
        """Return projectile/target thicknesses shifted by ±b/2 along x."""
        Xp = grid.X + 0.5 * b
        Xt = grid.X - 0.5 * b
        Y = grid.Y

        if self.kind == "AA":
            Tp = self.nuc_p.T(Xp, Y)
        elif self.kind == "pA":
            Tp = self.proton.T(Xp, Y)
        elif self.kind == "dA":
            Tp = self.deuteron.T(Xp, Y)
        else:
            raise ValueError("Unknown kind")

        Tt = self.nuc_t.T(Xt, Y)
        return Tp, Tt

    # ---- core fields ----

    def profiles_xy(self, b: float, grid: Grid2D):
        """Return (W_proj, W_targ, C) at impact parameter b."""
        Tp, Tt = self._TA_TB(b, grid)
        Wp = wounded_density(Tp, Tt, self.sigma_fm2)
        Wt = wounded_density(Tt, Tp, self.sigma_fm2)
        C = binary_density(Tp, Tt, self.sigma_fm2)
        return Wp, Wt, C

    def TAB(self, b: float, grid: Grid2D) -> float:
        """Overlap integral T_AB(b) = ∫ d^2x T_proj(x+b/2) T_targ(x-b/2)."""
        Tp, Tt = self._TA_TB(b, grid)
        return integrate_2d(Tp * Tt, grid)

    def P_inel(self, b: float, grid: Grid2D) -> float:
        """Inelastic probability at b (optical): 1 - exp(-σ T_AB(b))."""
        return float(1.0 - np.exp(-self.sigma_fm2 * self.TAB(b, grid)))

    def dSigma_db(self, b: float, grid: Grid2D) -> float:
        """Differential inelastic cross section weight: 2π b P_inel(b)."""
        return float(2.0 * np.pi * b * self.P_inel(b, grid))

    def Npart_Ncoll(self, b: float, grid: Grid2D) -> tuple[float, float]:
        Wp, Wt, C = self.profiles_xy(b, grid)
        Np = integrate_2d(Wp + Wt, grid)
        Nc = integrate_2d(C, grid)
        return float(Np), float(Nc)

    # ---- centrality by b-percentiles (geometry-only baseline) ----

    def centrality_edges_b(self, *, grid: Grid2D, bmax: float, db: float, cent_edges: np.ndarray) -> np.ndarray:
        """Return b-edges corresponding to requested centrality percentiles."""
        b_vals = np.arange(0.0, bmax + db, db)
        w = np.array([self.dSigma_db(b, grid) for b in b_vals])
        cum = np.cumsum(w) * db
        cum /= cum[-1]  # 0..1

        edges = []
        for c in cent_edges:
            f = float(c) / 100.0
            idx = int(np.searchsorted(cum, f))
            if idx <= 0:
                edges.append(b_vals[0])
            elif idx >= len(b_vals):
                edges.append(b_vals[-1])
            else:
                b0, b1 = b_vals[idx - 1], b_vals[idx]
                c0, c1 = cum[idx - 1], cum[idx]
                edges.append(b0 + (f - c0) * (b1 - b0) / (c1 - c0))
        return np.array(edges, dtype=float)

    def _avg_in_bin(self, b_vals: np.ndarray, w: np.ndarray, q: np.ndarray, b0: float, b1: float) -> float:
        m = (b_vals >= b0) & (b_vals <= b1)
        bb = b_vals[m]
        ww = w[m]
        qq = q[m]
        num = np.trapz(qq * ww, bb)
        den = np.trapz(ww, bb)
        return float(num / den)

    def centrality_table(
        self,
        *,
        grid: Grid2D,
        bmax: float = 20.0,
        db: float = 0.25,
        cent_edges: np.ndarray = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    ):
        """Compute b-edges and bin-averaged ⟨b⟩, ⟨Npart⟩, ⟨Ncoll⟩."""
        b_vals = np.arange(0.0, bmax + db, db)
        TAB = np.array([self.TAB(b, grid) for b in b_vals])
        w = 2.0 * np.pi * b_vals * (1.0 - np.exp(-self.sigma_fm2 * TAB))  # dσ/db

        Np = np.zeros_like(b_vals)
        Nc = np.zeros_like(b_vals)
        for i, b in enumerate(b_vals):
            Np[i], Nc[i] = self.Npart_Ncoll(b, grid)

        edges = self.centrality_edges_b(grid=grid, bmax=bmax, db=db, cent_edges=cent_edges)

        avg_b = []
        avg_Np = []
        avg_Nc = []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            avg_b.append(self._avg_in_bin(b_vals, w, b_vals, b0, b1))
            avg_Np.append(self._avg_in_bin(b_vals, w, Np, b0, b1))
            avg_Nc.append(self._avg_in_bin(b_vals, w, Nc, b0, b1))

        return {
            "cent_edges": cent_edges,
            "b_edges": edges,
            "b_mean": np.array(avg_b),
            "Npart_mean": np.array(avg_Np),
            "Ncoll_mean": np.array(avg_Nc),
            "b_vals": b_vals,
            "TAB": TAB,
            "dSigma_db": w,
            "Npart": Np,
            "Ncoll": Nc,
        }

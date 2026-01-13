"""src/geometry.py
Author: Sabin Thapa <sthapa3@kent.edu>

Geometry primitives:
- Spherical Woods–Saxon nucleus thickness T_A(x,y)
- Generalized Gaussian proton thickness T_p(x,y) (aHydro p+Pb paper)
- Deuteron transverse structure via Hulthén wavefunction:
  - event-by-event pn transverse offsets
  - orientation-averaged thickness T_d(x,y) for optical studies

Designed to be:
- fast (precomputed thickness spline)
- deterministic (seeded sampling where relevant)
- reusable across notebooks
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import gamma


# -------------------------
# Woods–Saxon nucleus
# -------------------------

@dataclass(frozen=True)
class WoodsSaxonParams:
    """Spherical Woods–Saxon parameters.

    rho(r) = rho0 / (1 + exp((r - R)/a))

    Typical:
      rho0 ~ 0.17 fm^-3
      R ~ 1.12 A^{1/3} - 0.86 A^{-1/3}
      a ~ 0.535 (Au) or 0.549 (Pb) fm
    """
    A: int
    rho0: float
    R: float
    a: float


def default_radius(A: int) -> float:
    """Empirical radius used in your notebooks."""
    A = float(A)
    return 1.12 * A ** (1.0 / 3.0) - 0.86 * A ** (-1.0 / 3.0)


class SphericalWoodsSaxon:
    """Spherical Woods–Saxon density and thickness (optical)."""

    def __init__(self, params: WoodsSaxonParams, *, zmax: float = 20.0, Nz: int = 4001, rmax: float = 20.0, Nr: int = 2001):
        self.p = params
        self._build_thickness_spline(zmax=zmax, Nz=Nz, rmax=rmax, Nr=Nr)

    def rho(self, r: np.ndarray) -> np.ndarray:
        p = self.p
        return p.rho0 / (1.0 + np.exp((r - p.R) / p.a))

    def _build_thickness_spline(self, *, zmax: float, Nz: int, rmax: float, Nr: int) -> None:
        r = np.linspace(0.0, rmax, Nr)
        z = np.linspace(-zmax, zmax, Nz)

        rr = r[:, None]
        zz = z[None, :]
        s = np.sqrt(rr * rr + zz * zz)  # radial distance
        rho = self.rho(s)
        T = np.trapz(rho, z, axis=1)  # fm^-2

        self._T_spline = InterpolatedUnivariateSpline(r, T, k=3, ext=1)

    def T(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Thickness T(x,y) = ∫ dz ρ(sqrt(x^2+y^2+z^2)) [fm^-2]."""
        s = np.sqrt(x * x + y * y)
        return self._T_spline(s)


@lru_cache(maxsize=16)
def nucleus_from_name(name: str) -> SphericalWoodsSaxon:
    """Convenience factory with cached splines."""
    name = name.strip()
    if name.lower() == "au":
        A = 197
        p = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.535)
        return SphericalWoodsSaxon(p)
    if name.lower() == "pb":
        A = 208
        p = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.549)
        return SphericalWoodsSaxon(p)
    raise ValueError(f"Unknown nucleus '{name}'. Add it in nucleus_from_name().")


# -------------------------
# Proton profile (generalized Gaussian)
# -------------------------

@dataclass(frozen=True)
class ProtonProfileParams:
    """Generalized Gaussian parameters used in aHydro p+Pb initial conditions."""
    n: float = 1.85
    rp: float = 0.975  # fm


class GeneralizedGaussianProton:
    r"""Proton thickness profile:
    T_p(b) = n/(2π r_p^2 Γ(2/n)) exp[-(b/r_p)^n].

    Normalized so ∫ d^2b T_p(b) = 1.
    """

    def __init__(self, params: ProtonProfileParams = ProtonProfileParams()):
        self.p = params
        self._norm = self.p.n / (2.0 * np.pi * self.p.rp ** 2 * gamma(2.0 / self.p.n))

    def T(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        b = np.sqrt(x * x + y * y)
        return self._norm * np.exp(- (b / self.p.rp) ** self.p.n)


# -------------------------
# Deuteron: Hulthén sampling
# -------------------------

@dataclass(frozen=True)
class HulthenParams:
    """Hulthén parameters (fm^-1)."""
    a: float = 0.228
    b: float = 1.18


class HulthenSampler:
    """Sample pn separation in a deuteron and random 3D orientation.

    We use the Hulthén wavefunction ψ(r) ∝ (e^{-ar} - e^{-br})/r.
    The radial probability density for r is ∝ (e^{-ar} - e^{-br})^2.

    Output: transverse offsets (px,py,nx,ny) for proton and neutron in the deuteron CM frame.
    """

    def __init__(self, params: HulthenParams = HulthenParams(), *, rmax: float = 30.0, Nr: int = 200000):
        self.p = params
        r = np.linspace(1e-5, rmax, Nr)
        u = (np.exp(-self.p.a * r) - np.exp(-self.p.b * r))
        P = u * u  # radial PDF up to normalization
        cdf = np.cumsum(P) * (r[1] - r[0])
        cdf /= cdf[-1]
        self._r = r
        self._cdf = cdf

    def sample_transverse_offsets(self, n: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample r from tabulated CDF
        u = rng.random(n)
        r = np.interp(u, self._cdf, self._r)

        # Random 3D direction
        cos_th = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        sin_th = np.sqrt(1.0 - cos_th * cos_th)

        # Transverse separation magnitude rT = r sin θ
        rT = r * sin_th
        dx = rT * np.cos(phi)
        dy = rT * np.sin(phi)

        # Nucleons at ±(dx,dy)/2 in transverse plane
        px, py = 0.5 * dx, 0.5 * dy
        nx, ny = -0.5 * dx, -0.5 * dy
        return px, py, nx, ny


class DeuteronThickness:
    """Orientation-averaged deuteron thickness for optical-style studies.

    T_d(x,y) ≈ ⟨ T_p(x-x_p, y-y_p) + T_p(x-x_n, y-y_n) ⟩ over random Hulthén orientations.
    """

    def __init__(self, proton: GeneralizedGaussianProton, *, n_orientations: int = 1200, seed: int = 7, hulthen: HulthenParams = HulthenParams()):
        self.proton = proton
        self.sampler = HulthenSampler(hulthen)
        self.n = int(n_orientations)
        self.rng = np.random.default_rng(int(seed))
        self.px, self.py, self.nx, self.ny = self.sampler.sample_transverse_offsets(self.n, rng=self.rng)

    def T(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Td = np.zeros_like(x, dtype=float)
        for i in range(self.n):
            Td += self.proton.T(x - self.px[i], y - self.py[i])
            Td += self.proton.T(x - self.nx[i], y - self.ny[i])
        return Td / self.n

    def sample_event_offsets(self, *, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """Return one event's transverse offsets (px,py,nx,ny)."""
        px, py, nx, ny = self.sampler.sample_transverse_offsets(1, rng=rng)
        return float(px[0]), float(py[0]), float(nx[0]), float(ny[0])

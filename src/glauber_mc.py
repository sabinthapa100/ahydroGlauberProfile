"""src/glauber_mc.py
Author: Sabin Thapa <sthapa3@kent.edu>

Minimal, publication-friendly MC Glauber for small systems centrality.

Why you need this:
- In pA/dA, experimental "centrality" is usually based on multiplicity.
- Geometry (impact parameter b) correlates only weakly with multiplicity because
  fluctuations dominate.
- Optical b-percentiles are a clean *baseline*, but not a data-matching centrality definition.

This module provides:
- event sampling for pA and dA (and AA in a basic form)
- black-disk collision criterion using σ_NN
- participant & collision counting
- entropy proxy S with optional Gamma (negative-binomial-like) fluctuations
- centrality binning by S quantiles

Note: This is intentionally simple. If you want a full TRENTo implementation
(reduced thickness, nucleon substructure, etc.), we can add it next.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

from .physics import mb_to_fm2
from .geometry import WoodsSaxonParams, SphericalWoodsSaxon, nucleus_from_name, DeuteronThickness, GeneralizedGaussianProton


SystemKind = Literal["AA", "pA", "dA"]


def _sample_ws_radius(rng: np.random.Generator, R: float, a: float, rmax: float) -> float:
    """Rejection sample radius with PDF ∝ r^2 / (1 + exp((r-R)/a))."""
    while True:
        r = rng.uniform(0.0, rmax)
        # envelope: max of 1/(1+exp(...)) is 1; r^2 is handled explicitly
        u = rng.random()
        f = 1.0 / (1.0 + np.exp((r - R) / a))
        # Use r^2 weighting by comparing u to (r/rmax)^2 * f
        if u < (r / rmax) ** 2 * f:
            return r


def sample_ws_nucleons_transverse(
    params: WoodsSaxonParams,
    *,
    rng: np.random.Generator,
    rmax: float = 20.0,
) -> np.ndarray:
    """Sample A nucleon transverse positions (x,y) from a spherical Woods–Saxon."""
    A = int(params.A)
    R, a = float(params.R), float(params.a)

    # Sample radii + angles
    r = np.array([_sample_ws_radius(rng, R, a, rmax) for _ in range(A)], dtype=float)
    cos_th = rng.uniform(-1.0, 1.0, size=A)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=A)
    sin_th = np.sqrt(1.0 - cos_th * cos_th)

    x = r * sin_th * np.cos(phi)
    y = r * sin_th * np.sin(phi)
    return np.stack([x, y], axis=1)  # (A,2)


@dataclass(frozen=True)
class MCGlauberConfig:
    kind: SystemKind
    sNN_GeV: float
    sigmaNN_mb: float

    projectile: str  # "p","d","Au","Pb"
    target: str      # "Au","Pb"

    # entropy / multiplicity proxy
    chi: float = 0.15
    use_gamma_weights: bool = True
    gamma_k: float = 1.0  # Gamma(k, scale=1/k) gives mean=1, var=1/k

    # sampling
    bmax: float = 20.0


def _gamma_weights(rng: np.random.Generator, n: int, k: float) -> np.ndarray:
    # shape=k, scale=1/k -> mean=1
    k = float(k)
    return rng.gamma(shape=k, scale=1.0 / k, size=n)


def run_mc_glauber(
    cfg: MCGlauberConfig,
    *,
    n_events: int = 20000,
    seed: int = 123,
) -> Dict[str, Any]:
    """Run MC Glauber and return event-level observables.

    Returns dict with arrays:
      b, Npart, Ncoll, S
    """
    rng = np.random.default_rng(int(seed))
    sig = mb_to_fm2(cfg.sigmaNN_mb)
    d0 = np.sqrt(sig / np.pi)  # black-disk radius in fm
    d0sq = d0 * d0

    # Build target Woods–Saxon *parameters* (not thickness spline)
    if cfg.target.lower() == "au":
        A = 197
        from .geometry import default_radius
        targ_params = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.535)
    elif cfg.target.lower() == "pb":
        A = 208
        from .geometry import default_radius
        targ_params = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.549)
    else:
        raise ValueError("Unsupported target nucleus. Add it.")

    # Projectile settings
    if cfg.kind == "AA":
        if cfg.projectile.lower() not in ("au", "pb"):
            raise ValueError("AA requires Au/Pb projectile.")
        if cfg.projectile.lower() == "au":
            A = 197
            from .geometry import default_radius
            proj_params = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.535)
        else:
            A = 208
            from .geometry import default_radius
            proj_params = WoodsSaxonParams(A=A, rho0=0.17, R=default_radius(A), a=0.549)
    else:
        proj_params = None

    b = rng.uniform(0.0, cfg.bmax, size=n_events)
    # correct geometric sampling for impact parameter: p(b) ∝ b
    b = np.sqrt(rng.random(n_events)) * cfg.bmax

    Npart = np.zeros(n_events, dtype=int)
    Ncoll = np.zeros(n_events, dtype=int)
    S = np.zeros(n_events, dtype=float)

    # Deuteron sampling (event-by-event pn offsets)
    deut = None
    if cfg.kind == "dA":
        deut = DeuteronThickness(GeneralizedGaussianProton(), n_orientations=1, seed=seed)  # sampler only

    for ievt in range(n_events):
        bi = b[ievt]

        # Sample target transverse nucleon positions
        XY_t = sample_ws_nucleons_transverse(targ_params, rng=rng)

        # Sample projectile positions
        if cfg.kind == "pA":
            XY_p = np.array([[0.0, 0.0]], dtype=float)
        elif cfg.kind == "dA":
            px, py, nx, ny = deut.sample_event_offsets(rng=rng)
            XY_p = np.array([[px, py], [nx, ny]], dtype=float)
        elif cfg.kind == "AA":
            XY_p = sample_ws_nucleons_transverse(proj_params, rng=rng)
        else:
            raise ValueError("Unknown kind")

        # Shift centers by ±b/2 along x (projectile at +b/2, target at -b/2)
        XY_p = XY_p + np.array([+0.5 * bi, 0.0])
        XY_t = XY_t + np.array([-0.5 * bi, 0.0])

        # Pairwise distance check (broadcast)
        dx = XY_p[:, None, 0] - XY_t[None, :, 0]
        dy = XY_p[:, None, 1] - XY_t[None, :, 1]
        dsq = dx * dx + dy * dy
        coll = dsq < d0sq

        ncoll = int(coll.sum())
        Ncoll[ievt] = ncoll

        # Participants: any nucleon with ≥1 collision
        part_p = coll.any(axis=1)
        part_t = coll.any(axis=0)
        npart = int(part_p.sum() + part_t.sum())
        Npart[ievt] = npart

        # Entropy proxy S
        if cfg.use_gamma_weights:
            w_p = _gamma_weights(rng, int(part_p.sum()), cfg.gamma_k) if part_p.any() else np.array([])
            w_t = _gamma_weights(rng, int(part_t.sum()), cfg.gamma_k) if part_t.any() else np.array([])
            wounded = 0.5 * (w_p.sum() + w_t.sum())
        else:
            wounded = 0.5 * float(part_p.sum() + part_t.sum())

        S[ievt] = (1.0 - cfg.chi) * wounded + cfg.chi * float(ncoll)

    return {"b": b, "Npart": Npart, "Ncoll": Ncoll, "S": S, "cfg": cfg}


def centrality_bins_from_S(S: np.ndarray, cent_edges: np.ndarray) -> Dict[str, Any]:
    """Return centrality bin edges in S and event indices for each bin."""
    S = np.asarray(S, dtype=float)
    # centrality: most central = largest S
    order = np.argsort(S)[::-1]
    S_sorted = S[order]

    edges = {}
    idx_bins = []
    for c0, c1 in zip(cent_edges[:-1], cent_edges[1:]):
        f0 = c0 / 100.0
        f1 = c1 / 100.0
        i0 = int(np.floor(f0 * len(S_sorted)))
        i1 = int(np.floor(f1 * len(S_sorted)))
        idx = order[i0:i1]
        idx_bins.append(idx)
        edges[(c0, c1)] = (S_sorted[i0], S_sorted[i1 - 1] if i1 > i0 else S_sorted[i0])

    return {"cent_edges": cent_edges, "bins": idx_bins, "S_edges": edges}

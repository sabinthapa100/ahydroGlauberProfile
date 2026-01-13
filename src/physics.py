"""src/physics.py
Author: Sabin Thapa <sthapa3@kent.edu>

Small, stable physics utilities used across notebooks.
Keep this file boring and well-tested.

Conventions:
- length: fm
- σ_NN inelastic: mb (input) and fm^2 (internal)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# --- constants (GeV) ---
M_P = 0.9382720813
M_N = 0.9395654133

MB_TO_FM2 = 0.1  # 1 mb = 0.1 fm^2


def mb_to_fm2(sigma_mb: float) -> float:
    """Convert millibarn to fm^2."""
    return MB_TO_FM2 * float(sigma_mb)


def y_nucleon(sNN_GeV: float) -> float:
    r"""Beam rapidity proxy used in aHydro papers:
    $$y_N = \log\left(\frac{2\sqrt{s_{NN}}}{m_p+m_n}\right).$$
    """
    return float(np.log((2.0 * sNN_GeV) / (M_P + M_N)))


@dataclass(frozen=True)
class SigmaNNTable:
    """Anchor points for σ_NN^inel(√s) used in your aHydro initializations.

    This is *not* a global PDG fit; it is a practical table taken from common
    heavy-ion initialization choices and the specific aHydro papers you're using.
    Override per project as needed.
    """
    anchors_mb: dict

    def sigma_mb(self, sNN_GeV: float) -> float:
        # Exact match -> return
        if sNN_GeV in self.anchors_mb:
            return float(self.anchors_mb[sNN_GeV])

        # Interpolate in log(s) vs σ for sanity across decades.
        s = np.array(sorted(self.anchors_mb.keys()), dtype=float)
        sig = np.array([self.anchors_mb[x] for x in s], dtype=float)

        # Guard: require within range
        if sNN_GeV < s.min() or sNN_GeV > s.max():
            raise ValueError(
                f"sNN={sNN_GeV} GeV outside sigma table range [{s.min()}, {s.max()}]. "
                "Provide sigmaNN_mb explicitly."
            )

        xs = np.log(s)
        x = np.log(float(sNN_GeV))
        return float(np.interp(x, xs, sig))


# Practical defaults consistent with your notebooks/papers.
DEFAULT_SIGMA_NN = SigmaNNTable(
    anchors_mb={
        # RHIC
        200.0: 42.0,
        # LHC (values used in many initial-condition setups; update if your run cards differ)
        2760.0: 62.0,
        5020.0: 67.6,
        8160.0: 71.0,
    }
)

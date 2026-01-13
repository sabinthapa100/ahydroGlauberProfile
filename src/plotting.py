"""src/plotting.py
Author: Sabin Thapa <sthapa3@kent.edu>

Publication-oriented matplotlib helpers.

Default choices (as you requested):
- no grid
- no figure titles by default
- clean spines
- consistent fonts/sizes
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_pub_style():
    mpl.rcParams.update({
        "figure.figsize": (6.5, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "mathtext.fontset": "stix",
        "font.family": "DejaVu Sans",
    })


def style_ax(ax):
    ax.grid(False)
    ax.set_title("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def add_panel_label(ax, label: str, *, x: float = 0.02, y: float = 0.98):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

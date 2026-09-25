"""One chart style for the whole project, so every figure reads as one system.

Colors come from a documented, pre-validated palette (the dataviz reference instance).
Only pairs that palette validates for colour-vision deficiency are used:
  - slots 1-3 (blue, orange, aqua) validate ALL pairs, so any two can share a chart
  - the "no sepsis" group is muted grey: sepsis is the story, so it is the one colour
    and everything else recedes (emphasis, not a second categorical hue)
Color follows the ENTITY, fixed here once: hospital A is blue in every chart, sepsis is
orange in every chart. Aqua sits below 3:1 contrast on the light surface, so charts that
use it carry direct labels and the notebook shows the numbers as a table beside them.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {
    "A": "#2a78d6",          # slot 1 blue   - hospital A
    "B": "#1baf7a",          # slot 3 aqua   - hospital B
    "sepsis": "#eb6834",     # slot 2 orange - sepsis
    "no_sepsis": MUTED,      # recessive grey
    "series": "#2a78d6",     # single-series charts use slot 1
}


def apply() -> None:
    """Thin marks, hairline solid grid, recessive axes, system sans."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "patch.linewidth": 0,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Write to reports/figures so the README can embed the same image the notebook shows."""
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")

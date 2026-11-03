# -*- coding: utf-8 -*-
"""Vaste stijl voor grafieken: titel, asnamen met eenheid, blauw/oranje/grijs, >= 150 dpi."""
import matplotlib.pyplot as plt

from .config import INK, INK2, DPI


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e2", "axes.axisbelow": True, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                         "axes.titlecolor": INK, "axes.titlelocation": "left", "font.size": 9,
                         "legend.frameon": False})


def save(fig, path):
    """Opslaan met vaste resolutie en strakke marges."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    print("grafiek:", path.name)

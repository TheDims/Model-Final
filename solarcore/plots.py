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


def to_english(fig):
    """Alle tekst in de grafiek naar het Engels (titels, assen, legenda's, labels, categorieën op de assen)."""
    from matplotlib.text import Text
    from matplotlib.ticker import FixedFormatter
    from .labels_en import translate
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, FixedFormatter):
                fmt.seq = [translate(s) for s in fmt.seq]
    for t in fig.findobj(Text):
        s = t.get_text()
        if s:
            t.set_text(translate(s))


def save(fig, path):
    """Opslaan met vaste resolutie en strakke marges; tekst in het Engels (de groep en docenten lezen Engels)."""
    to_english(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    print("grafiek:", path.name)

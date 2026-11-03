# -*- coding: utf-8 -*-
"""
HEATMAPS voor het verslag (Model Final)
=======================================
Alle heatmaps in één map: resultaten/heatmaps/

  1. Correlatie van alle features (sensor, ERA5-weer, natuurkunde, tijd/zon) met elkaar en met de opwek.
  2. Fout van het eindmodel (LR GHI, ERA5) per testdag en per plant.
  3. Fout per model x testdag, apart voor Plant 1 en Plant 2.
  4. Fout per model x plant.
Fouten = nRMSE (% van de capaciteit), overdag. Opzet: bevroren model, getraind op de trainingsdagen en getest op
de testdagen van de run (zelfde opzet als fase 3 en 4). Geen nieuwe keuzes: dit script laat alleen zien wat er al getest is.

Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase 0-4.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import RESULTS, PLANTS, MIN_COSZ, BLUE, ORANGE, INK, INK2   # noqa: E402
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.models import LADDER_FASE3, LADDER_FASE1   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "heatmaps"
OUT.mkdir(parents=True, exist_ok=True)
style()
BLUES = LinearSegmentedColormap.from_list("blauw", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])

f3 = json.loads((RESULTS / "fase3" / "samenvatting.json").read_text(encoding="utf-8"))
RAD_SHIFT, N_SETUP = int(f3["tijdafspraak_min"]), int(f3["opzet_dagen"].split("-")[1])
EIND = f3["keuze"]                                           # LR GHI (ERA5)
E6 = "e6: EFF_T x (a + b KT_POA), zonder intercept"
data = weather_frame(RAD_SHIFT)
N_DAYS = int(data.DAY.max())
SETUP, TEST = list(range(1, N_SETUP + 1)), list(range(N_SETUP + 1, N_DAYS + 1))
TE_TXT = f"dag {TEST[0]}-{TEST[-1]}"


def heatmap(M, rows, cols, title, fname, cmap=BLUES, vmin=None, vmax=None, fmt="{:.1f}", size=(8, 4.5),
            xlabel="", ylabel="", cbar="nRMSE (%) - lager is beter", fs=8):
    """Heatmap met de waarde in elk vak (witte tekst op donkere vakken)."""
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    long = max(len(str(c)) for c in cols) > 10             # alleen lange labels schuin zetten
    ax.set_xticks(range(len(cols)), cols, rotation=40 if long else 0, ha="right" if long else "center")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    lo = np.nanmin(M) if vmin is None else vmin
    hi = np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            dark = abs(v) > 0.6 if cmap is DIV else (v - lo) / (hi - lo + 1e-9) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="white" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar); cb.outline.set_visible(False)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    save(fig, OUT / fname)


# %% 1. Correlatie van alle features (overdag, beide plants, alle 34 dagen)
dd = add_target(data, capacity(data))                        # beschrijvend: capaciteit uit alle dagen is hier OK
FEATS = {   # kolom: (naam in de grafiek, groep)
    "FRAC": "opwek (FRAC)",
    "IRRADIATION": "sensor: instraling", "MODULE_TEMPERATURE": "sensor: moduletemp.",
    "AMBIENT_TEMPERATURE": "sensor: luchttemp.",
    "GHI_W": "ERA5: straling (GHI)", "KT_W": "ERA5: helderheid (KT)", "CLOUD": "ERA5: bewolking",
    "T": "ERA5: luchttemp.", "RH": "ERA5: luchtvochtigheid", "WS10": "ERA5: wind", "PRECIP": "ERA5: neerslag",
    "POA": "pvlib: straling paneel (POA)", "EFF_T": "pvlib: EFF_T", "KT_POA": "pvlib: KT_POA",
    "EFF_T_KT": "pvlib: EFF_T x KT_POA", "T_CELL": "pvlib: celtemp.", "PHYS": "pvlib: opwek",
    "COSZ": "zonnestand (cos zenit)", "HOUR_SIN": "uur (sin)", "HOUR_COS": "uur (cos)"}
C = dd[dd.OK][list(FEATS)].corr()
lab = list(FEATS.values())
heatmap(C.values, lab, lab, "Correlatie van alle features (Pearson, overdag, beide plants, 34 dagen)",
        "01_correlatie_alle_features.png", cmap=DIV, vmin=-1, vmax=1, fmt="{:.2f}", size=(13, 10.5),
        cbar="correlatie (Pearson)", fs=7)
C.round(3).to_csv(OUT / "01_correlatie_alle_features.csv")
print("Correlatie met de opwek (FRAC):")
print(C["FRAC"].drop("FRAC").sort_values(ascending=False).round(3).rename(FEATS).to_string())

# %% 2. Voorspellingen van alle modellen op de testdagen (bevroren, getraind op de trainingsdagen)
cap = capacity(data[data.DAY.isin(SETUP)])
d = add_target(data, cap)
d.loc[d.DIP, "FRAC"] = np.nan
tr, te = d[d.DAY.isin(SETUP) & d.OK], d[d.DAY.isin(TEST) & d.OK]
prof = tr.groupby(["PLANT", "SLOT"]).FRAC.mean()
s = d.set_index(["PLANT", "DAY", "SLOT"]).FRAC
s = s[~s.index.duplicated()]
k = dayahead(add_daily(add_clear(d, d[d.DAY.isin(SETUP)]))).loc[te.index]
pred = {"gemiddelde dag": np.nan_to_num(prof.reindex(list(zip(te.PLANT, te.SLOT))).values),
        "persistence (gisteren)": s.reindex(list(zip(te.PLANT, te.DAY - 1, te.SLOT))).fillna(0).values,
        "smart persistence (geen weer)": (k.KD_1.fillna(k.KD.mean()) * k.FRAC_CS).values}
EIND_LABEL = f"EINDMODEL: {EIND.split(' (')[0]}"
for name, label in dict.fromkeys([(EIND, EIND_LABEL), (E6, "e6 (ERA5 + pvlib)"), ("LR GHI (ERA5)", "LR GHI (ERA5)")]):
    if label != EIND_LABEL and name == EIND:
        continue
    cols, make, _ = LADDER_FASE3[name]
    pred[label] = make().fit(tr[cols], tr.FRAC).predict(te[cols])
cols, make, _ = LADDER_FASE1["LR straling"]
pred["sensor (bovengrens)"] = make().fit(tr[cols], tr.FRAC).predict(te[cols])
pred = {n: np.clip(v, 0, None) for n, v in pred.items()}
MODELS = list(pred)
err = pd.DataFrame({n: (v - te.FRAC.values) ** 2 for n, v in pred.items()}, index=te.index)
err[["PLANT", "DAY"]] = te[["PLANT", "DAY"]]
per_day = err.groupby(["PLANT", "DAY"])[MODELS].mean() ** 0.5 * 100            # nRMSE per plant-dag
per_day.round(3).to_csv(OUT / "fout_per_dag_per_plant.csv")

# %% 3. Fout van het eindmodel per testdag en per plant
E = per_day[EIND_LABEL].unstack("DAY")
E.loc["samen"] = (err.groupby("DAY")[EIND_LABEL].mean() ** 0.5 * 100)
heatmap(E.values, ["Plant 1", "Plant 2", "samen"], [str(x) for x in E.columns],
        f"Fout van het eindmodel ({EIND.split(' (')[0]}) per testdag en per plant", "02_fout_eindmodel_dag_plant.png",
        fmt="{:.0f}", size=(13, 2.8), xlabel=f"testdag ({TEST[0]} = {pd.Timestamp('2020-05-14') + pd.Timedelta(days=TEST[0]):%d-%m}, 34 = 17-06-2020)", ylabel="plant")

# %% 4. Fout per model x testdag, per plant
vmax = float(np.nanpercentile(per_day.values, 98))
for p in PLANTS:
    M = per_day.loc[p][MODELS].T
    heatmap(M.values, MODELS, [str(x) for x in M.columns], f"Plant {p}: fout per model en testdag",
            f"03_fout_model_dag_plant{p}.png", vmin=0, vmax=vmax, fmt="{:.0f}", size=(13, 3.8),
            xlabel="testdag", ylabel="model")

# %% 5. Fout per model x plant (hele testperiode)
rows = {}
for n in MODELS:
    rows[n] = {**{f"Plant {p}": float(err.loc[err.PLANT == p, n].mean() ** 0.5 * 100) for p in PLANTS},
               "samen": float(err[n].mean() ** 0.5 * 100)}
MP = pd.DataFrame(rows).T
heatmap(MP.values, MODELS, list(MP.columns), f"Fout per model en plant (test {TE_TXT})", "04_fout_model_plant.png",
        size=(8, 4), xlabel="plant", ylabel="model")
MP.round(3).to_csv(OUT / "04_fout_model_plant.csv")
print("\nnRMSE per model en plant (%):"); print(MP.round(2).to_string())
plt.show()
print(f"\nKlaar. Heatmaps in {OUT.relative_to(ROOT)}")

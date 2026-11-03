# -*- coding: utf-8 -*-
"""
LEERCURVE  -  helpt meer trainingsdata? (AGENTS.md §6.0, BESLISSINGEN #16)
=========================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  - Altijd DEZELFDE testdagen: dag 28-34 (= de laatste 20 %, de 80/20-split).
  - Bevroren modellen, getraind op de eerste 10 / 17 / 24 / 27 dagen (beide plants samen).
  - Modellen: gemiddelde dag, smart persistence, LR GHI, LR POA, e5, e6, Ridge e6. GEEN nieuwe keuze: alleen meten.
  - Per trainingsgrootte: nRMSE (+ 90 %-interval, bootstrap over de 7 testdagen), dagfout, bias, en a/b van e6.
  - Gepaarde bootstrap: is 27 dagen aantoonbaar beter dan 24 (en 24 dan 10), per model, op dezelfde dagen?
Capaciteit en heldere-hemelschaal komen per trainingsgrootte alleen uit die trainingsdagen. De ERA5-tijdafspraak
(+15 min) en de dip-vlag komen uit run 2 (dag 1-24); run 1 (dag 1-10) gaf precies dezelfde (BESLISSINGEN #12, #20).

Uitvoer: resultaten/leercurve/. Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase 0-3 van split70.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (RESULTS, LEARNING_SIZES, LEARNING_TEST, MIN_COSZ, RUN_NAME, BLUE, ORANGE, GREY,   # noqa: E402
                              INK2)
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.models import LADDER_FASE3   # noqa: E402
from solarcore.evaluate import metrics, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS.parent / "leercurve"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
f3 = json.loads((RESULTS / "fase3" / "samenvatting.json").read_text(encoding="utf-8"))
RAD_SHIFT = int(f3["tijdafspraak_min"])
TEST = list(range(LEARNING_TEST[0], LEARNING_TEST[1] + 1))
SP = "smart persistence (geen weer)"
E5, E6 = "e5: EFF_T, KT_POA, CS_POA, COSZ (LR met intercept)", "e6: EFF_T x (a + b KT_POA), zonder intercept"
MODELS = {"LR GHI (ERA5)": "LR GHI", "LR POA (straling op het paneel)": "LR POA", E5: "e5", E6: "e6",
          "Ridge e6 (zonder intercept)": "Ridge e6"}
NAMES = ["gemiddelde dag", SP] + list(MODELS.values())
print(f"Leercurve (gegevens uit {RUN_NAME}, tijdafspraak {RAD_SHIFT:+d} min): trainen op {LEARNING_SIZES} dagen, "
      f"testen op dag {TEST[0]}-{TEST[-1]}")

# %% 1. Per trainingsgrootte: trainen (bevroren) en voorspellen op dezelfde testdagen
data = weather_frame(RAD_SHIFT)
rows, coefs, all_pred, te_ref = [], [], {}, None
for n in LEARNING_SIZES:
    train = list(range(1, n + 1))
    d = add_target(data, capacity(data[data.DAY.isin(train)]))
    d.loc[d.DIP, "FRAC"] = np.nan
    tr, te = d[d.DAY.isin(train) & d.OK], d[d.DAY.isin(TEST) & d.OK]
    if te_ref is None:
        te_ref = te
    assert te.index.equals(te_ref.index), "testkwartieren moeten voor elke trainingsgrootte gelijk zijn"
    k = dayahead(add_daily(add_clear(d, d[d.DAY.isin(train)]))).loc[te.index]
    prof = tr.groupby(["PLANT", "SLOT"]).FRAC.mean()
    pr = {"gemiddelde dag": np.nan_to_num(prof.reindex(list(zip(te.PLANT, te.SLOT))).values),
          SP: (k.KD_1.fillna(k.KD.mean()) * k.FRAC_CS).values}
    for key, short in MODELS.items():
        cols, make, _ = LADDER_FASE3[key]
        m = make().fit(tr[cols], tr.FRAC)
        pr[short] = m.predict(te[cols])
        if short in ("e6", "Ridge e6"):
            est, sc = (m[-1], m[0]) if hasattr(m, "steps") else (m, None)
            c = est.coef_ / (sc.scale_ if sc is not None else 1)
            coefs.append({"trainingsdagen": n, "model": short, "a": c[0], "b": c[1]})
    for name, y in pr.items():
        all_pred[f"{name} | {n}"] = np.clip(y, 0, None) * (te.COSZ.values > MIN_COSZ)
    cap_used = capacity(data[data.DAY.isin(train)])
    all_pred[f"__cap | {n}"] = te.PLANT.map(cap_used).values                 # capaciteit van deze training
coef_tab = pd.DataFrame(coefs)

# %% 2. Alles op dezelfde schaal, bootstrap (90 %-interval) en gepaarde vergelijking van trainingsgroottes
# De capaciteit (noemer van FRAC) verschilt per trainingsgrootte. Voor een eerlijke vergelijking rekenen we alles
# om naar dezelfde schaal: de capaciteit uit 27 dagen (de grootste training). FRAC_ref = FRAC_n x cap_n / cap_ref.
n_max = max(LEARNING_SIZES)
cap_ref = all_pred[f"__cap | {n_max}"]
te_b = te_ref.copy()
te_b["FRAC"] = te_ref.AC_KW.values / cap_ref                                # echte waarde op de referentieschaal
preds_b = {k: v * all_pred[f"__cap | {k.split(' | ')[1]}"] / cap_ref for k, v in all_pred.items()
           if not k.startswith("__cap")}
rows = []
for k, y in preds_b.items():
    name, n = k.split(" | ")
    rows.append({"trainingsdagen": int(n), "model": name, **metrics(te_b.FRAC, y, te_b),
                 "bias_%": float(np.mean(y - te_b.FRAC.values) * 100)})
res = pd.DataFrame(rows)
pairs = [(f"{m} | {n_max}", f"{m} | 24") for m in NAMES] + [(f"{m} | 24", f"{m} | 10") for m in NAMES]
boot_m, boot_d = bootstrap_days(te_b, preds_b, pairs=pairs)
boot_m.index = pd.MultiIndex.from_tuples([(k.split(" | ")[0], int(k.split(" | ")[1])) for k in boot_m.index],
                                         names=["model", "trainingsdagen"])
tab = res.set_index(["model", "trainingsdagen"]).join(boot_m[["laag_%", "hoog_%"]])
tab = tab[["nRMSE_%", "laag_%", "hoog_%", "dagfout_%", "bias_%", "nMAE_%", "R2"]]
print("\n=== Leercurve: nRMSE (90 %-interval), dagfout en bias per model en trainingsgrootte (test dag "
      f"{TEST[0]}-{TEST[-1]}) ===")
print(tab.loc[NAMES].round(2).to_string())
print("\n=== Gepaarde bootstrap: verschil in nRMSE (pp) tussen trainingsgroottes, zelfde testdagen ===")
print(boot_d.round(2).to_string())
print("\na en b van e6 per trainingsgrootte:"); print(coef_tab.round(6).to_string(index=False))
tab.round(4).to_csv(OUT / "leercurve.csv"); boot_d.round(4).to_csv(OUT / "leercurve_verschillen.csv")
coef_tab.to_csv(OUT / "coefficienten_e6.csv", index=False)
(OUT / "samenvatting.json").write_text(json.dumps({
    "test_dagen": f"{TEST[0]}-{TEST[-1]}", "trainingsgroottes": list(LEARNING_SIZES), "tijdafspraak_min": RAD_SHIFT,
    "leercurve": {f"{m} | {n}": v for (m, n), v in tab.round(4).to_dict("index").items()},
    "verschillen": boot_d.round(4).to_dict("index"), "coefficienten_e6": coefs,
    "opmerking": "geen nieuwe keuze; alle fouten op dezelfde schaal (capaciteit uit de grootste training, 27 dagen)"},
    indent=2, ensure_ascii=False, default=float),
    encoding="utf-8")

# %% 3. Grafiek: nRMSE tegen het aantal trainingsdagen
style()
COL = {"gemiddelde dag": GREY, SP: ORANGE, "LR GHI": "#9ec5f4", "LR POA": "#6da7ec", "e5": BLUE,
       "e6": "#184f95", "Ridge e6": "#0d366b"}
fig, ax = plt.subplots(figsize=(8, 4.4))
for i, m in enumerate(NAMES):
    s = tab.loc[m]
    x = np.array(LEARNING_SIZES) + (i - len(NAMES) / 2) * 0.12               # iets uit elkaar, anders overlap
    ax.errorbar(x, s["nRMSE_%"], yerr=[s["nRMSE_%"] - s["laag_%"], s["hoog_%"] - s["nRMSE_%"]], marker="o", ms=6,
                lw=2, capsize=2, color=COL[m], ls="--" if m in ("gemiddelde dag", SP) else "-", label=m)
ax.set_xticks(LEARNING_SIZES, [f"{n}\n({n / 34:.0%})" for n in LEARNING_SIZES])
ax.set_xlabel("aantal trainingsdagen (aandeel van 34)"); ax.set_ylabel("nRMSE (% van capaciteit)")
ax.set_title(f"Leercurve: zelfde testdagen {TEST[0]}-{TEST[-1]}, meer trainingsdagen (90 %-interval)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
save(fig, OUT / "leercurve.png")
plt.show()
print(f"\nKlaar. Resultaten in {OUT.relative_to(ROOT)}")

# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 4  -  UK PV: horizon mengen met de gemiddelde dag (BESLISSINGEN #41, ACHTERAF)
=============================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Stap 3 liet zien: de weersvoorspelling helpt tot ~dag 4; vanaf dag 7 is de gemiddelde dag beter. Hier:
  voorspelling_d = w_d x model + (1 - w_d) x gemiddelde dag, per dag vooruit d,
  w_d met kleinste kwadraten op de TRAININGSPERIODE (runs vanaf 14 mrt, tijdstippen t/m 30 jun 2024).
Vergelijken op de test (jul 2024 - mrt 2025) met: alleen model, alleen gemiddelde dag en de vaste regel
"model t/m dag 4, daarna gemiddelde dag". ACHTERAF-test: de testperiode is in stap 3 al bekeken.
Runnen: F5 in Spyder of VS Code (Anaconda). Eerst taak5b_ukpv_tests.py (cache).
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import UKPV_RESULTS, UKPV_FC_START, UKPV_TRAIN_END, BLUE, ORANGE, GREY, INK2  # noqa: E402
from solarcore.ukpv import chosen_systems, model_frame, horizon_features, horizon_frame   # noqa: E402
from solarcore.models import _lr   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = UKPV_RESULTS / "stap4_horizon_mengen"
OUT.mkdir(parents=True, exist_ok=True)
style()
RULE_DAY = 4                                   # vaste regel: model t/m dag 4 (uit stap 3, skill > 0,1)
TOL = 0.2                                      # regel #41: mengen "beter" als <= min(model, gem. dag) + 0,2 pp

# %% 0. Data: zelfde als stap 3 (gekozen model LR POA, getraind op de voorspelling dag 1)
step3 = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
assert step3["keuze"] == "LR POA", "dit script gaat uit van de keuze LR POA (#39)"
chosen = chosen_systems()
data = model_frame(step3["tijdverschuiving_min"], chosen)
TR = data.DATE_TIME.between(pd.Timestamp(UKPV_FC_START), pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59"))
hz = horizon_frame(data, horizon_features(step3["tijdverschuiving_min"], chosen), TR, ["POA"], _lr, "F1")
train = hz[hz.DATE_TIME <= pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59")]
test = hz[hz.DATE_TIME >= "2024-07-01"].copy()
print(f"horizon-rijen: training {len(train)} ({train.RUN.nunique()} runs), test {len(test)} ({test.RUN.nunique()} runs)")

# %% 1. Gewicht per dag vooruit (alleen training)
w = {}
for d, g in train.groupby("DAG"):
    m, c, y = g.P_MODEL.values, g.P_GEMDAG.values, g.FRAC.values
    w[d] = float(np.clip(((y - c) * (m - c)).sum() / ((m - c) ** 2).sum(), 0, 1))
w = pd.Series(w, name="w_d")
print("Gewicht van het model per dag vooruit (training):", w.round(2).to_dict())
TRAINED = list(w.index)                         # runs in de training gaan maar 10 dagen vooruit
w = w.reindex(range(1, 16)).ffill()             # dag 11-15: gewicht van dag 10 doorgetrokken (gemeld)

# %% 2. Test
test["P_MENG"] = test.DAG.map(w) * test.P_MODEL + (1 - test.DAG.map(w)) * test.P_GEMDAG
test["P_REGEL"] = np.where(test.DAG <= RULE_DAY, test.P_MODEL, test.P_GEMDAG)
LAB = {"P_MODEL": "alleen model (LR POA)", "P_GEMDAG": "alleen gemiddelde dag",
       "P_REGEL": f"regel: model t/m dag {RULE_DAY}, daarna gem. dag", "P_MENG": "mengen (w_d uit training)"}
rows = []
for d, g in test.groupby("DAG"):
    rows.append({"dag": d, "w_d": w.get(d, np.nan),
                 **{LAB[c]: float(np.sqrt(((g[c] - g.FRAC) ** 2).mean()) * 100) for c in LAB}})
res = pd.DataFrame(rows).set_index("dag")
res["w uit training"] = res.index.isin(TRAINED)
best_single = res[[LAB["P_MODEL"], LAB["P_GEMDAG"]]].min(axis=1)
res["mengen beter (regel #41)"] = res[LAB["P_MENG"]] <= best_single + TOL
t10 = test[test.DAG.isin(TRAINED)]
overall = {LAB[c]: float(np.sqrt(((t10[c] - t10.FRAC) ** 2).mean()) * 100) for c in LAB}
print("\n=== Test jul 2024 - mrt 2025: nRMSE (%) per dag vooruit (ACHTERAF, #41) ===")
print(res.round(2).to_string())
print(f"Gemiddeld over dag 1-{max(TRAINED)} (alle methodes op dezelfde rijen):", {k: round(v, 2) for k, v in overall.items()})
print(f"Regel #41 gehaald op dag 1-{max(TRAINED)}:", bool(res.loc[TRAINED, "mengen beter (regel #41)"].all()),
      "| dag 11-15 (gewicht doorgetrokken):", bool(res.loc[~res["w uit training"], "mengen beter (regel #41)"].all()))
res.round(4).to_csv(OUT / "horizon_mengen.csv")
(OUT / "samenvatting.json").write_text(json.dumps(
    {"taak": "5b stap 4 - horizon mengen (ACHTERAF, BESLISSINGEN #41)", "gewichten": w.round(4).to_dict(),
     "per_dag": res.round(4).to_dict("index"), "gemiddeld": overall,
     "regel_gehaald_dag_1_10": bool(res.loc[TRAINED, "mengen beter (regel #41)"].all())}, indent=2, ensure_ascii=False, default=float),
    encoding="utf-8")

# %% 3. Grafiek
fig, ax = plt.subplots(figsize=(8, 4))
for c, col, ls in (("P_MODEL", BLUE, "--"), ("P_GEMDAG", GREY, "-"), ("P_REGEL", INK2, ":"), ("P_MENG", ORANGE, "-")):
    ax.plot(res.index, res[LAB[c]], color=col, ls=ls, marker="o", ms=3, lw=2, label=LAB[c])
ax.set(title="Horizon: voorspelling mengen met de gemiddelde dag (UK, testperiode)",
       xlabel="dag vooruit (dag 1 = 0-24 uur na de run van 00 UTC)", ylabel="nRMSE (%)", xticks=res.index)
ax.legend(fontsize=8)
save(fig, OUT / "01_horizon_mengen.png")
print("klaar:", OUT)

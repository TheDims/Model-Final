# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 5  -  UK PV: meerdere weermodellen voor morgen (ECMWF, ICON, GFS)  (BESLISSINGEN #52)
====================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  1. Previous Runs (dag 1 = voorspeld 24 uur eerder) ophalen voor DWD ICON en NOAA GFS (ECMWF hebben we al);
  2. zelfde verwerking (pvlib, echte dak) -> POA per weermodel;
  3. kandidaten: LR POA per model, LR op het gemiddelde van de 3 POA's, LR op de 3 POA's;
     kiezen met regel #17 op dezelfde 2 vouwen als stap 3; test jul 2024 - mrt 2025 + bootstrap tegen ECMWF.
Runnen: F5 in Spyder of VS Code (Anaconda). Eerst taak5b_ukpv_tests.py (cache). Internet de eerste keer.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (UKPV_RESULTS, UKPV_CLEAN, UKPV_PERIOD, UKPV_WEATHER, UKPV_FC_START, UKPV_TRAIN_END,  # noqa: E402
                              UKPV_FOLDS, CAPACITY_QUANTILE, MIN_COSZ, BLUE, GREY)
from solarcore.ukpv import chosen_systems, era5_system, model_frame, forecast_hourly, features_30min   # noqa: E402
from solarcore.forecast_api import previous_runs   # noqa: E402
from solarcore.models import _lr   # noqa: E402
from solarcore.evaluate import metrics, choose, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = UKPV_RESULTS / "stap5_multimodel"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
EXTRA = {"I": "icon_seamless", "G": "gfs_seamless"}          # achtervoegsel -> Open-Meteo-model
step3 = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
SHIFT = step3["tijdverschuiving_min"]
chosen = chosen_systems()

# %% 1. Ophalen + verwerken (gecachet)
cache = UKPV_CLEAN / f"ukpv_features_multimodel_shift{SHIFT}.parquet"
if cache.exists():
    extra = pd.read_parquet(cache)
else:
    parts = []
    for i, sid in enumerate(chosen.index, 1):
        lat, lon = chosen.latitude_rounded[sid], chosen.longitude_rounded[sid]
        elev = float(era5_system(chosen, sid)[1])
        cols = []
        for suf, model in EXTRA.items():
            c = UKPV_WEATHER / "dagvooruit" / f"{model}_{sid}.csv"
            new = not c.exists()
            fc = previous_runs(lat, lon, *UKPV_PERIOD, c, model=model)
            if new:
                time.sleep(2)
            x = features_30min(forecast_hourly(fc, 1), lat, lon, elev, SHIFT, float(chosen.tilt[sid]),
                               float(chosen.orientation[sid]))
            cols.append(x[["POA", "KT_W"]].add_suffix(f"_{suf}"))
        parts.append(pd.concat(cols, axis=1).assign(PLANT=sid).reset_index())
        print(f"  {i}/{len(chosen)}")
    extra = pd.concat(parts, ignore_index=True)
    extra.to_parquet(cache, index=False)
data = model_frame(SHIFT, chosen).merge(extra, on=["DATE_TIME", "PLANT"], how="left")
data["POA_F"] = data.POA_F1                                   # ECMWF (stap 3)
data["POA_MEAN"] = data[["POA_F", "POA_I", "POA_G"]].mean(axis=1)
OK = data.OK & data[["POA_I", "POA_G"]].notna().all(axis=1)
print(f"bruikbare halve uren: {int(OK.sum())} (stap 3: {int(data.OK.sum())})")

# %% 2. Kandidaten en keuze (regel #17)
CAND = {"LR POA ECMWF (stap 3)": (["POA_F"], (0, 1, 1)), "LR POA ICON": (["POA_I"], (0, 1, 1)),
        "LR POA GFS": (["POA_G"], (0, 1, 1)), "LR gemiddelde POA (3 modellen)": (["POA_MEAN"], (0, 3, 1)),
        "LR op 3 POA's": (["POA_F", "POA_I", "POA_G"], (0, 3, 3))}


def between(a, b):
    return data.DATE_TIME.between(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta("23:59:59"))


def run(tr, te):
    cap = data[tr & OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)
    d = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    trn, tst = d[tr & OK], d[te & OK]
    return tst, {n: np.clip(_lr().fit(trn[c], trn.FRAC).predict(tst[c]), 0, None) * (tst.COSZ.values > MIN_COSZ)
                 for n, (c, _) in CAND.items()}


rows = []
for a, b, c, d in UKPV_FOLDS:
    te, pr = run(between(a, b), between(c, d))
    rows += [{"model": n, "vouw": f"{a[5:]}..{b[5:]} -> {c[5:7]}", **metrics(te.FRAC, y, te)} for n, y in pr.items()]
val = pd.DataFrame(rows)
tab = val.groupby("model", sort=False)[["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]].mean()
tab = tab.join(val.pivot(index="model", columns="vouw", values="nRMSE_%").add_prefix("nRMSE_% ")).loc[list(CAND)]
CHOICE, WHY = choose(tab, {n: e for n, (_, e) in CAND.items()}, [c for c in tab.columns if c.startswith("nRMSE_% ")])
print("\n=== Validatie (2 vouwen) ==="); print(tab.round(3).to_string())
print("KEUZE:", CHOICE); print("Waarom:", WHY)

# %% 3. Test jul 2024 - mrt 2025
te, pr = run(between(UKPV_FC_START, UKPV_TRAIN_END), between("2024-07-01", UKPV_PERIOD[1]))
test = pd.DataFrame({n: metrics(te.FRAC, y, te) for n, y in pr.items()}).T
boot_m, boot_d = bootstrap_days(te, pr, pairs=[(n, "LR POA ECMWF (stap 3)") for n in CAND if n != "LR POA ECMWF (stap 3)"])
print("\n=== Test jul 2024 - mrt 2025 (dag 1) ==="); print(test[["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]].round(3).to_string())
print("\nBootstrap t.o.v. ECMWF (90 %):"); print(boot_d.round(2).to_string())
val.to_csv(OUT / "validatie_per_vouw.csv", index=False); tab.round(4).to_csv(OUT / "validatie.csv")
test.round(4).to_csv(OUT / "test.csv"); boot_d.round(4).to_csv(OUT / "bootstrap_verschillen.csv")
(OUT / "samenvatting.json").write_text(json.dumps({
    "taak": "5b stap 5 - meerdere weermodellen (BESLISSINGEN #52)", "keuze": CHOICE, "motivatie": WHY,
    "validatie": tab.round(4).to_dict("index"), "test": test.round(4).to_dict("index")},
    indent=2, ensure_ascii=False, default=float), encoding="utf-8")

fig, ax = plt.subplots(figsize=(8, 3.6))
v = test["nRMSE_%"]
ax.bar(range(len(v)), v.values, color=[BLUE if n == CHOICE else GREY for n in v.index])
for i, x in enumerate(v.values):
    ax.text(i, x + 0.1, f"{x:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(v)), [n.replace(" (", "\n(") for n in v.index], fontsize=7)
ax.set(title="UK, test jul 2024 - mrt 2025: welk weermodel voor morgen? (blauw = gekozen)", ylabel="nRMSE (%)",
       ylim=(v.min() - 1.5, v.max() + 0.8))
ax.grid(axis="x", visible=False)
save(fig, OUT / "01_weermodellen.png")
print("klaar:", OUT)

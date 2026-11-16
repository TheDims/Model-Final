# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 2  -  UK PV: weer achteraf (ERA5) en archief-weersvoorspellingen ophalen (BESLISSINGEN #37)
=========================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  1. ERA5 (Open-Meteo archief) per systeem: weer ACHTERAF = bovengrens;
  2. dag-vooruit: ECMWF IFS 0,25° via Previous Runs (voorspeld 24 en 48 uur vóór het tijdstip);
  3. horizon: ECMWF IFS 9 km via Single Runs, run 00 UTC elke 4e dag, 15 dagen vooruit, alle locaties per aanvraag;
  4. controle: dekking per maand en correlatie opwek <-> straling per uur (ERA5, dag 1, dag 2) per systeem.
Alles wordt gecachet in data/extern/uk_pv/weer/ (opnieuw draaien haalt alleen wat ontbreekt). Nog GEEN model.
Internet nodig (alleen de eerste keer). Runnen: F5 in Spyder of in VS Code (Anaconda-Python).
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (UKPV_CLEAN, UKPV_RESULTS, UKPV_PERIOD, UKPV_WEATHER, UKPV_DA_MODEL,  # noqa: E402
                              UKPV_HZ_MODEL, UKPV_HZ_RUNS, BLUE, ORANGE, GREY)
from solarcore.ukpv import read_metadata   # noqa: E402
from solarcore.weather import load_era5   # noqa: E402
from solarcore.forecast_api import previous_runs, single_run   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = UKPV_RESULTS / "stap2_weer"
OUT.mkdir(parents=True, exist_ok=True)
(UKPV_WEATHER / "era5").mkdir(parents=True, exist_ok=True)
style()
START, END = UKPV_PERIOD
chosen = pd.read_csv(UKPV_RESULTS / "stap1_systemen" / "gekozen_systemen.csv", index_col="ss_id")
LAT, LON = chosen.latitude_rounded, chosen.longitude_rounded

# %% 1 + 2. ERA5 en dag-vooruit per systeem
era, da = {}, {}
for i, sid in enumerate(chosen.index, 1):
    era[sid], _ = load_era5(LAT[sid], LON[sid], START, END, tz="GMT", folder=UKPV_WEATHER / "era5")
    cache = UKPV_WEATHER / "dagvooruit" / f"{UKPV_DA_MODEL}_{sid}.csv"
    new = not cache.exists()
    da[sid] = previous_runs(LAT[sid], LON[sid], START, END, cache, model=UKPV_DA_MODEL)
    print(f"  {i:2d}/{len(chosen)} systeem {sid}: ERA5 {len(era[sid])} uur, dag-vooruit {len(da[sid])} uur")
    if new:
        time.sleep(2)                                          # vriendelijk voor de gratis API

# %% 3. Horizon: Single Runs (alle 40 locaties per aanvraag)
runs = pd.date_range(UKPV_HZ_RUNS[0], UKPV_HZ_RUNS[1], freq=UKPV_HZ_RUNS[2])
hz = []
for j, run in enumerate(runs, 1):
    cache = UKPV_WEATHER / "horizon" / f"{UKPV_HZ_MODEL}_{run:%Y%m%d}00.csv"
    new = not cache.exists()
    hz.append(single_run(list(LAT), list(LON), list(chosen.index), f"{run:%Y-%m-%dT00:00}", cache, model=UKPV_HZ_MODEL))
    if new:
        print(f"  run {j}/{len(runs)}: {run:%Y-%m-%d} 00 UTC")
        time.sleep(10)
hz = pd.concat(hz, ignore_index=True)
print(f"horizon: {len(runs)} runs, {len(hz)} rijen")

# %% 4. Controle: dekking en correlatie met de opwek (per uur; PV-tijd = einde halfuur -> uur met label = einde uur)
pv = pd.read_parquet(UKPV_CLEAN / "ukpv_30min.parquet")
pv = pv[~pv.SLECHTE_DAG]
rows, cov = [], {}
for sid in chosen.index:
    p = pv[pv.ss_id == sid].set_index("datetime_GMT").P_KW.resample("1h", label="right", closed="right").mean()
    w = era[sid].set_index("DATE_TIME").shortwave_radiation.rename("ERA5")
    f = da[sid].set_index("DATE_TIME")[["shortwave_radiation_d1", "shortwave_radiation_d2"]]
    d = pd.concat([p, w, f], axis=1).dropna()
    d = d[d.ERA5 > 10]
    rows.append({"ss_id": sid, "r_ERA5": d.P_KW.corr(d.ERA5), "r_dag1": d.P_KW.corr(d.shortwave_radiation_d1),
                 "r_dag2": d.P_KW.corr(d.shortwave_radiation_d2), "n_uur": len(d)})
    cov[sid] = da[sid].set_index("DATE_TIME").shortwave_radiation_d1.notna().resample("MS").mean()
check = pd.DataFrame(rows).set_index("ss_id")
cov = pd.DataFrame(cov)
cov.index = cov.index.strftime("%Y-%m")
print("\nCorrelatie opwek <-> straling per uur (mediaan over 40 systemen):")
print(check[["r_ERA5", "r_dag1", "r_dag2"]].describe().loc[["min", "50%", "max"]].round(3).to_string())
print("\nDekking dag-vooruit (dag 1) per maand, gemiddeld over de systemen:")
print(cov.mean(axis=1).round(2).to_dict())
lead_cov = hz.assign(DAG=(hz.LEAD_H // 24).astype(int) + 1).groupby("DAG").shortwave_radiation.apply(lambda s: s.notna().mean())
print("\nDekking horizon per dag vooruit:", lead_cov.round(2).to_dict())
check.round(4).to_csv(OUT / "correlatie_per_systeem.csv")
cov.round(3).to_csv(OUT / "dekking_dagvooruit_per_maand.csv")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.boxplot([check.r_ERA5, check.r_dag1, check.r_dag2], tick_labels=["ERA5 (achteraf)", "voorspeld dag 1", "voorspeld dag 2"])
ax.set(title="UK PV: correlatie opwek met straling per uur (40 systemen)", ylabel="r (-)")
save(fig, OUT / "01_correlatie_weerbronnen.png")

summary = {"taak": "5b stap 2 - weer ophalen", "dagvooruit_model": UKPV_DA_MODEL, "horizon_model": UKPV_HZ_MODEL,
           "horizon_runs": [str(r.date()) for r in runs], "correlatie_mediaan": check.median().round(4).to_dict(),
           "dekking_dag1_per_maand": cov.mean(axis=1).round(3).to_dict(), "dekking_horizon_per_dag": lead_cov.round(3).to_dict()}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("klaar:", OUT)

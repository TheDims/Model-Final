# -*- coding: utf-8 -*-
"""
TAAK 5a - stap 2  -  HKUST: opschonen naar 15 minuten + check van wind en temperatuur
=====================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Wat dit script doet (nog GEEN model):
  1. de 5 gekozen stations (BESLISSINGEN #25): 5-min AC-vermogen -> 15 min (zelfde rooster als Kaggle);
  2. weerstation 1 min -> 15 min (GHI, temperatuur, wind, vochtigheid);
  3. vlaggen: DAG (cos zenit > 0,02), SENSOR_OK (BESLISSINGEN #26), STORING (bekende maanden), UITVAL;
  4. wind en temperatuur van het weerstation vergelijken met ERA5 (nodig voor de Faiman-fit, TAAK 6);
  5. opslaan: data/extern/hkust/clean/hkust_15min.csv + resultaten/taak5/hkust/stap2_data/.
Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Duurt een paar minuten.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (HKUST_RESULTS, HKUST_CLEAN, HKUST_TZ, HKUST_LOCATION, HKUST_YEARS,  # noqa: E402
                              HKUST_STATIONS, HKUST_FAULT_MONTHS, HKUST_SENSOR_OK, HKUST_OUTAGE_FRAC, MIN_COSZ,
                              DATA, FREQ, BLUE, ORANGE, GREY)
from solarcore.hkust import read_brick, systems_table, inverter_files, station_15min, weather_15min, norm_name  # noqa: E402
from solarcore.weather import load_era5   # noqa: E402
from solarcore.data_prep import check_df   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = HKUST_RESULTS / "stap2_data"
OUT.mkdir(parents=True, exist_ok=True)
HKUST_CLEAN.mkdir(parents=True, exist_ok=True)
style()
LAT, LON = HKUST_LOCATION
SUN_ERA5 = 300.0           # W/m²: ERA5 ziet duidelijk zon (voor de uitvalvlag; de sensor is niet overal betrouwbaar)

# %% 1. Weer: weerstation (15 min) en ERA5 (uur)
w15 = weather_15min(HKUST_YEARS)
era5, era_elev = load_era5(LAT, LON, f"{HKUST_YEARS[0]}-01-01", f"{HKUST_YEARS[-1]}-12-31", tz=HKUST_TZ,
                           folder=DATA / "extern" / "hkust")
era = era5.set_index("DATE_TIME")
grid = pd.date_range(f"{HKUST_YEARS[0]}-01-01", f"{HKUST_YEARS[-1]}-12-31 23:45", freq=FREQ)
sp = pvlib.solarposition.get_solarposition((grid + pd.Timedelta("450s")).tz_localize(HKUST_TZ), LAT, LON)
cosz = pd.Series(np.cos(np.radians(sp["apparent_zenith"].to_numpy())), index=grid)
# ERA5-straling van uur h = gemiddelde over (h-1, h]: kwartieren in dat uur krijgen die waarde (alleen voor de vlag)
era_ghi_15 = pd.Series(era["shortwave_radiation"].reindex((grid + pd.Timedelta("1h")).floor("h")).to_numpy(), index=grid)

# %% 2. Wind en temperatuur: weerstation vs ERA5 (uurgemiddelde, label = einde van het uur)
w1h = w15[["T_AIR", "WIND"]].resample("1h", label="right", closed="right").mean()
cmp = pd.concat([w1h, era[["temperature_2m", "wind_speed_10m"]]], axis=1, join="inner").dropna()
check_wind = {}
for y in HKUST_YEARS:
    c = cmp[cmp.index.year == y]
    check_wind[y] = {"wind_sensor_gem": round(c.WIND.mean(), 2), "wind_era5_gem": round(c.wind_speed_10m.mean(), 2),
                     "wind_r": round(c.WIND.corr(c.wind_speed_10m), 3),
                     "temp_sensor_gem": round(c.T_AIR.mean(), 2), "temp_era5_gem": round(c.temperature_2m.mean(), 2),
                     "temp_r": round(c.T_AIR.corr(c.temperature_2m), 3)}
print("wind/temperatuur per jaar:")
print(pd.DataFrame(check_wind).T.to_string())
monthly = cmp.groupby(cmp.index.to_period("M")).mean()
monthly.round(2).to_csv(OUT / "wind_temp_per_maand.csv")

# %% 3. Stations -> 15 min + vlaggen
brick = systems_table(read_brick()).set_index("key")
files, _ = inverter_files()
sensor_ok = (grid >= pd.Timestamp(HKUST_SENSOR_OK[0])) & (grid <= pd.Timestamp(HKUST_SENSOR_OK[1]))
parts, info = [], {}
for st in HKUST_STATIONS:
    key = norm_name(st)
    meta = brick.loc[key]
    p = station_15min(files[key]).reindex(grid)
    p = p.where(cosz > 0, 0.0)                                   # zon onder -> opwek 0 (er zijn geen nachtrijen)
    d = pd.DataFrame({"DATE_TIME": grid, "STATION": st, "P_KW": p.to_numpy()}, index=grid).join(w15)
    d["COSZ"] = cosz
    d["DAG"] = cosz > MIN_COSZ
    d["SENSOR_OK"] = sensor_ok
    d["STORING"] = grid.to_period("M").astype(str).isin(HKUST_FAULT_MONTHS.get(st, ()))
    d["UITVAL"] = d["DAG"] & (era_ghi_15 > SUN_ERA5) & (d["P_KW"] < HKUST_OUTAGE_FRAC * meta["kWp"])
    d["KWP_AC"], d["KWP_DC"] = meta["kWp"], meta["kWp_modules"]
    first = d["P_KW"].where(d["DAG"]).first_valid_index()
    d = d.loc[first.normalize():]                                # vanaf de eerste dag met data
    usable = d["DAG"] & d["P_KW"].notna() & ~d["STORING"] & ~d["UITVAL"]
    info[st] = {"kWp_AC": meta["kWp"], "kWp_DC": meta["kWp_modules"], "eerste_dag": str(first.date()),
                "daglicht_kwartieren": int(d["DAG"].sum()), "bruikbaar": int(usable.sum()),
                "bruikbaar_pct": round(100 * usable.sum() / d["DAG"].sum(), 1),
                "uitval": int(d["UITVAL"].sum()), "storing": int((d["DAG"] & d["STORING"]).sum()),
                "bruikbaar_per_jaar": {y: int(usable[d.index.year == y].sum()) for y in HKUST_YEARS}}
    parts.append(d)
    print(f"  {st:18s} {meta['kWp']:6.1f} kW AC  vanaf {first.date()}  bruikbaar {info[st]['bruikbaar_pct']} %"
          f"  uitval {info[st]['uitval']}  storing {info[st]['storing']}")
clean = pd.concat(parts, ignore_index=True)
info_df = check_df(clean, "hkust_15min")
clean.to_csv(HKUST_CLEAN / "hkust_15min.csv", index=False)
print("opgeslagen:", HKUST_CLEAN / "hkust_15min.csv")

# %% 4. Grafieken
fig, ax = plt.subplots(figsize=(9, 4))
for st, c in zip(HKUST_STATIONS, (BLUE, ORANGE, GREY, "#6b6a66", "#c9c8c3")):
    d = clean[clean.STATION == st].set_index("DATE_TIME")
    ok = d["P_KW"].where(~d["STORING"] & ~d["UITVAL"])
    daily = (ok.resample("D").sum() * 0.25) / d["KWP_AC"].iloc[0]          # kWh per kW AC
    daily = daily.where(ok.where(d["DAG"]).resample("D").count() > 30)     # dagen met te weinig data -> gat
    ax.plot(daily.rolling(7, min_periods=4).mean(), color=c, lw=1, label=st)
for y in HKUST_YEARS[1:]:
    ax.axvline(pd.Timestamp(f"{y}-01-01"), color="k", lw=0.8, ls="--")
ax.axvline(pd.Timestamp(HKUST_SENSOR_OK[1]), color=ORANGE, lw=1, ls=":", label="einde betrouwbare sensor")
ax.set(title="Dagopbrengst per station (7-daags gemiddelde); stippellijn = jaargrens (train / valideer / test)",
       xlabel="datum", ylabel="opbrengst (kWh per kW AC)")
ax.legend(ncol=3, fontsize=7)
save(fig, OUT / "01_dagopbrengst_stations.png")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
x = monthly.index.to_timestamp()
axs[0].plot(x, monthly["WIND"], color=BLUE, label="weerstation (10 m mast)")
axs[0].plot(x, monthly["wind_speed_10m"], color=ORANGE, label="ERA5 (10 m)")
axs[0].set(title="Windsnelheid per maand", ylabel="wind (m/s)", xlabel="maand")
axs[1].plot(x, monthly["T_AIR"], color=BLUE, label="weerstation")
axs[1].plot(x, monthly["temperature_2m"], color=ORANGE, label="ERA5 (2 m)")
axs[1].set(title="Luchttemperatuur per maand", ylabel="temperatuur (°C)", xlabel="maand")
for a in axs:
    a.legend()
    a.tick_params(axis="x", rotation=45)
save(fig, OUT / "02_wind_temp_vs_era5.png")

summary = {"taak": "5a stap 2 - opschonen HKUST", "stations": info, "wind_temp_check": check_wind,
           "era5_hoogte_m": era_elev, "rooster": "15 min, label = begin kwartier, HKT",
           "regels": {"pv_min_5min_per_kwartier": 3, "weer_min_1min_per_kwartier": 10,
                      "uitval": f"P < {HKUST_OUTAGE_FRAC:.0%} van kWp AC terwijl ERA5-GHI > {SUN_ERA5:.0f} W/m²",
                      "sensor_ok": HKUST_SENSOR_OK}}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("klaar:", OUT)

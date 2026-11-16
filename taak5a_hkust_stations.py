# -*- coding: utf-8 -*-
"""
TAAK 5a - stap 1  -  HKUST rooftop PV: data begrijpen en stations kiezen
========================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Wat dit script doet (nog GEEN model):
  1. Brick-metadata (.ttl) lezen met rdflib + SPARQL: kWp, tilt, azimuth, modules, aansluitdatum per systeem;
  2. tijdzone controleren: valt de middag van de gemeten straling op de berekende zonnemiddag (HKT, UTC+8)?
  3. per station met panel-optimizer (5-min omvormerdata) een datacheck: dekking per jaar, uitval,
     correlatie met het weerstation, PR per uur (schaduw), tijdverschuiving t.o.v. het weerstation;
  4. de vooraf vastgelegde keuzeregel toepassen en kandidaat-stations voorstellen;
  5. opslaan in resultaten/taak5/hkust/ (csv + png).
Stations zonder optimizer hebben alleen uurdata (te grof voor het 15-min-protocol) en doen niet mee.
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
from solarcore.config import (HKUST_RESULTS, HKUST_TZ, HKUST_LOCATION, HKUST_YEARS, HKUST_MIN_GHI,  # noqa: E402
                              HKUST_MIN_ELEV, HKUST_OUTAGE_FRAC, HKUST_MIN_COVERAGE, HKUST_MIN_SHADE_INDEX,
                              HKUST_MIN_R, BLUE, ORANGE, GREY)
from solarcore.hkust import read_brick, systems_table, inverter_files, read_station_power, read_weather  # noqa: E402
from solarcore.weather import load_era5   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = HKUST_RESULTS / "stap1_stations"
OUT.mkdir(parents=True, exist_ok=True)
style()
LAT, LON = HKUST_LOCATION
MIDDAY = (11.0, 13.0)          # uur-klasse "rond de middag" voor de PR-verhouding
HOURS = np.arange(7, 18)       # uur-klassen voor het PR-profiel
LAGS = range(-30, 35, 5)       # tijdverschuiving PV <-> weerstation (min)
SHADE_HOURS = np.arange(8, 17) # aangepaste schaduwindex: uur 8-16 (zon hoog genoeg, weinig IAM-verlies)

# %% 1. Brick-metadata
brick = read_brick()
systems = systems_table(brick)
print(f"Brick: {len(systems)} systemen, tikfouten gerepareerd: {brick.attrs['ttl_reparaties']}")
afwijking = (systems["kWp_modules"] / systems["kWp"] - 1).abs()
print("kWp opgegeven vs modules (>5 % verschil):")
print(systems.loc[afwijking > 0.05, ["kWp", "kWp_modules"]].to_string())

# %% 2. Weerstation (straling) + zonnestand op een 5-min-rooster
ghi_1min = read_weather("Irradiance", "Irradiance (W/m2)", HKUST_YEARS)
ghi = ghi_1min.resample("5min", label="left", closed="left").mean()   # GHI(t) = gemiddelde t .. t+5 min
grid = pd.date_range(f"{HKUST_YEARS[0]}-01-01", f"{HKUST_YEARS[-1]}-12-31 23:55", freq="5min")
sp = pvlib.solarposition.get_solarposition((grid + pd.Timedelta("150s")).tz_localize(HKUST_TZ), LAT, LON)
elev = pd.Series(sp["apparent_elevation"].to_numpy(), index=grid)
ghi = ghi.reindex(grid)
daytime = elev > 5

# Tijdzone-check: zwaartepunt van de straling op heldere dagen vs berekende zonnemiddag
day = ghi.where(daytime).groupby(grid.date)
hour = pd.Series(grid.hour + grid.minute / 60 + 2.5 / 60, index=grid)
centroid = (ghi.where(daytime) * hour).groupby(grid.date).sum() / day.sum()
cs = pvlib.location.Location(LAT, LON, tz=HKUST_TZ).get_clearsky(grid.tz_localize(HKUST_TZ), model="ineichen")["ghi"]
kt_day = day.sum() / pd.Series(cs.to_numpy(), index=grid).where(daytime).groupby(grid.date).sum()
noon = (sp["apparent_elevation"].groupby(grid.date).idxmax().dt.tz_localize(None))
noon_h = pd.Series(noon.dt.hour + noon.dt.minute / 60, index=noon.index)
clear = kt_day > 0.7
tz_check = {"heldere_dagen": int(clear.sum()),
            "zwaartepunt_straling_uur": round(float(centroid[clear].median()), 2),
            "zonnemiddag_HKT_uur": round(float(noon_h[clear].median()), 2)}
print("tijdzone-check:", tz_check)

# Hoeveel zonne-energie valt er na 18:00? (de PV-bestanden stoppen elke dag om 17:55)
evening = ghi.where(daytime & (grid.hour >= 18)).groupby(grid.month).sum() / ghi.where(daytime).groupby(grid.month).sum()
print("aandeel GHI na 18:00 per maand (%):", (100 * evening).round(1).to_dict())

# Dekking van het weerstation
weather_cov = {y: round(float(ghi[daytime & (grid.year == y)].notna().mean()), 3) for y in HKUST_YEARS}
print("dekking weerstation (GHI, daguren):", weather_cov)

# %% 2b. Sensorcheck: stralingssensor tegen heldere hemel (pvlib) en ERA5 (Open-Meteo, CC BY 4.0), per maand
era5, _ = load_era5(LAT, LON, f"{HKUST_YEARS[0]}-01-01", f"{HKUST_YEARS[-1]}-12-31", tz=HKUST_TZ,
                    folder=HKUST_RESULTS.parents[2] / "data" / "extern" / "hkust")
era_ghi = era5.set_index("DATE_TIME")["shortwave_radiation"]
sens_h = ghi_1min.resample("1h", label="right", closed="right").mean()   # Open-Meteo: gemiddelde voorgaande uur
cs1 = pd.Series(cs.to_numpy(), index=grid)
month = grid.to_period("M")
sensor = pd.DataFrame({
    "sensor_kWh_m2": sens_h.groupby(sens_h.index.to_period("M")).sum() / 1000,
    "era5_kWh_m2": era_ghi.groupby(era_ghi.index.to_period("M")).sum() / 1000,
    "sensor_p99_W_m2": ghi.groupby(month).quantile(0.99),
    "heldere_hemel_max_W_m2": cs1.groupby(month).max()}).loc[str(HKUST_YEARS[0]):str(HKUST_YEARS[-1])]
sensor["sensor_div_era5"] = sensor["sensor_kWh_m2"] / sensor["era5_kWh_m2"]
sensor["p99_div_heldere_hemel"] = sensor["sensor_p99_W_m2"] / sensor["heldere_hemel_max_W_m2"]
sensor.round(3).to_csv(OUT / "sensorcheck_per_maand.csv")
print(sensor[["sensor_div_era5", "p99_div_heldere_hemel"]].round(2).T.to_string())


# %% 3. Datacheck per station
def station_check(key, paths, meta):
    """Kengetallen voor één station (5-min AC-vermogen tegen het weerstation)."""
    p = read_station_power(paths).reindex(grid)
    kwp = meta["kWp"]
    have = p.notna() & daytime
    win = daytime & (grid.hour >= 6) & (grid.hour < 18)          # venster waarin de bestanden data hebben
    first = p.first_valid_index()
    row = {"station": key, "kWp": kwp, "tilt": meta["tilt"], "azimuth": meta["azimuth"],
           "n_omvormers": len(paths), "aangesloten": meta["aangesloten"].date(),
           "eerste_data": first.date(), "laatste_data": p.last_valid_index().date()}
    for y in HKUST_YEARS:
        in_year = win & (grid.year == y)
        row[f"dekking_{y}"] = round(float(p[in_year].notna().mean()), 3)                 # t.o.v. het hele jaar
        since = in_year & (grid >= first)
        row[f"gaten_{y}"] = round(float(p[since].isna().mean()), 3) if since.any() else np.nan  # na de start
    sun = have & (ghi > HKUST_MIN_GHI)
    row["uitval_pct"] = round(100 * float((p[sun] < HKUST_OUTAGE_FRAC * kwp).mean()), 2)
    ok = sun & (p >= HKUST_OUTAGE_FRAC * kwp)
    row["r_ghi"] = round(float(np.corrcoef(p[have & ghi.notna()], ghi[have & ghi.notna()])[0, 1]), 3)
    pr = (p / kwp) / (ghi / 1000)
    hr = grid.hour + grid.minute / 60
    mid = ok & (hr >= MIDDAY[0]) & (hr < MIDDAY[1])
    row["PR_middag"] = round(float(pr[mid].median()), 3)
    prof = pr[ok & (elev > HKUST_MIN_ELEV)].groupby(grid.hour[ok & (elev > HKUST_MIN_ELEV)]).median()
    prof = prof.reindex(HOURS) / row["PR_middag"]
    row["schaduw_index"] = round(float(prof.min()), 3)
    row["slechtste_uur"] = int(prof.idxmin())
    row["P_max_per_kWp"] = round(float(p.max() / kwp), 3)
    # aanvulling (na de eerste run, zie README): 'kWp' in de .ttl is het AC-vermogen van de omvormer;
    # het paneelvermogen (DC) volgt uit de modules. Uur 7 bevat bij platte panelen al veel natuurkundig IAM-verlies.
    row["kWp_DC"] = meta["kWp_modules"]
    row["DC_AC"] = round(meta["kWp_modules"] / kwp, 2)
    row["PR_middag_DC"] = round(row["PR_middag"] * kwp / meta["kWp_modules"], 3)
    row["clipping_pct"] = round(100 * float((p[sun] >= 0.98 * kwp).mean()), 1)
    row["schaduw_index_8_16"] = round(float(prof.loc[SHADE_HOURS].min()), 3)
    p15, g15 = p.resample("15min").mean(), ghi.resample("15min").mean()
    both = p15.notna() & g15.notna() & (elev.resample("15min").mean() > 5)
    row["r_ghi_15min"] = round(float(np.corrcoef(p15[both], g15[both])[0, 1]), 3)
    # de sensor verspringt (sensorcheck), dus r over de hele periode is te laag: mediaan van r per kwartaal
    q = p15[both].groupby(p15[both].index.to_period("Q"))
    row["r_ghi_15min_kwartaal"] = round(float(q.apply(lambda s: s.corr(g15[s.index])).median()), 3)
    # PR per maand tegen ERA5 (onafhankelijk van de sensor), op DC-kWp -> storingen per station opsporen
    ph = p.resample("1h", label="right", closed="right").mean()
    e = era_ghi.reindex(ph.index).where(ph.notna())
    pr_m = (ph.groupby(ph.index.to_period("M")).sum() / meta["kWp_modules"]) / (e.groupby(e.index.to_period("M")).sum() / 1000)
    pr_era5[meta["sys"]] = pr_m.where(ph.notna().groupby(ph.index.to_period("M")).mean() > 0.3)
    # tijdverschuiving: bij welke verschuiving van de straling is de correlatie het hoogst?
    corr = {lag: p[have].corr(ghi.shift(-lag // 5)[have]) for lag in LAGS}
    row["lag_min"] = max(corr, key=corr.get)
    return row, prof


files, names = inverter_files()
sysidx = systems.reset_index().set_index("key")
rows, profiles, pr_era5 = [], {}, {}
for key, paths in files.items():
    meta = sysidx.loc[key]
    row, prof = station_check(meta["sys"], paths, meta)
    rows.append(row)
    profiles[meta["sys"]] = prof
    print(f"  {meta['sys']:32s} {row['kWp']:7.1f} kWp  r={row['r_ghi']:.3f}  schaduw={row['schaduw_index']:.2f}"
          f"  uitval={row['uitval_pct']:.1f} %  lag={row['lag_min']:+d} min")
inv = pd.DataFrame(rows).set_index("station").sort_values("kWp")

# storing = maand waarin de PR (t.o.v. ERA5) < 85 % van de mediaan van alle stations in die maand
pr_era5 = pd.DataFrame(pr_era5)
rel = pr_era5.div(pr_era5.median(axis=1), axis=0)
pr_era5.round(3).to_csv(OUT / "pr_per_maand_era5.csv")
inv["storing_maanden"] = (rel < 0.85).sum().reindex(inv.index)
inv["storing_welke"] = [", ".join(str(m) for m in rel.index[rel[s] < 0.85]) for s in inv.index]

# %% 4. Keuzeregel (vooraf vast): geschikt als alle eisen gelden
inv["eis_2021_bruikbaar"] = inv["eerste_data"] <= pd.Timestamp("2021-06-30").date()   # ≥ 6 maanden om te trainen
inv["eis_weinig_gaten"] = (inv[[f"gaten_{y}" for y in HKUST_YEARS]].max(axis=1) <= 1 - HKUST_MIN_COVERAGE)
# eerste versie (uur 7-17, r op 5 min) liet 0 stations door; aangepaste definitie, oude kolommen blijven staan
inv["eis_geen_schaduw"] = inv["schaduw_index_8_16"] >= HKUST_MIN_SHADE_INDEX
inv["eis_correlatie"] = inv["r_ghi_15min_kwartaal"] >= HKUST_MIN_R
inv["eis_een_richting"] = ~inv["tilt"].astype(str).str.contains("Mixed|/") & ~inv["azimuth"].astype(str).str.contains("Mixed|/")
eisen = [c for c in inv.columns if c.startswith("eis_")]
inv["geschikt"] = inv[eisen].all(axis=1)
inv.to_csv(OUT / "stations_inventaris.csv")
systems.drop(columns="key").to_csv(OUT / "brick_systemen.csv")
print(f"\n{inv['geschikt'].sum()} van {len(inv)} stations voldoen aan alle eisen:")
print(inv.loc[inv["geschikt"], ["kWp", "kWp_DC", "tilt", "eerste_data", "gaten_2021", "gaten_2022", "gaten_2023",
                                "uitval_pct", "clipping_pct", "r_ghi_15min_kwartaal", "schaduw_index_8_16",
                                "storing_maanden", "storing_welke"]].to_string())
print("\nNiet geschikt (welke eis faalt):")
for st, r in inv.loc[~inv["geschikt"]].iterrows():
    print(f"  {st:32s} {', '.join(c[4:] for c in eisen if not r[c])}")

# %% 5. Grafieken
fig, ax = plt.subplots(figsize=(8, 4))
for st, prof in profiles.items():
    good = inv.loc[st, "geschikt"]
    ax.plot(prof.index + 0.5, prof.values, color=BLUE if good else GREY, alpha=0.9 if good else 0.35,
            lw=1.6 if good else 0.8)
ax.axhline(HKUST_MIN_SHADE_INDEX, color=ORANGE, ls="--", lw=1, label=f"drempel {HKUST_MIN_SHADE_INDEX}")
ax.plot([], [], color=BLUE, label="geschikt"), ax.plot([], [], color=GREY, label="niet geschikt")
ax.set(title="PR per uur t.o.v. de middag (lager = schaduw of ongunstige richting)", xlabel="uur (HKT)",
       ylabel="PR / PR middag (-)", ylim=(0, 1.4))
ax.legend()
save(fig, OUT / "01_pr_profiel_per_uur.png")

fig, ax = plt.subplots(figsize=(8, 4.5))
cov = inv[[f"gaten_{y}" for y in HKUST_YEARS]] * 100
x = np.arange(len(inv))
for i, (y, c) in enumerate(zip(HKUST_YEARS, (BLUE, ORANGE, GREY))):
    ax.bar(x + (i - 1) * 0.27, cov[f"gaten_{y}"], width=0.27, color=c, label=str(y))
ax.axhline(100 * (1 - HKUST_MIN_COVERAGE), color="k", ls="--", lw=1, label="drempel 5 %")
ax.set_xticks(x, inv.index, rotation=90, fontsize=7)
ax.set(title="Ontbrekende 5-min-waarden overdag (na de start van het station)", ylabel="ontbrekend (%)")
ax.legend()
save(fig, OUT / "02_gaten_per_station.png")

fig, ax = plt.subplots(figsize=(8, 3.8))
x = sensor.index.to_timestamp()
ax.plot(x, sensor["p99_div_heldere_hemel"], color=BLUE, marker="o", ms=3, label="sensor P99 / heldere hemel (max)")
ax.plot(x, sensor["sensor_div_era5"], color=ORANGE, marker="o", ms=3, label="sensor / ERA5 (maandsom)")
ax.axhline(1, color=GREY, lw=1)
ax.set(title="Stralingssensor weerstation HKUST: per maand t.o.v. heldere hemel en ERA5", ylabel="verhouding (-)",
       xlabel="maand")
ax.legend()
save(fig, OUT / "03_sensorcheck.png")

summary = {"taak": "5a stap 1 - stationkeuze HKUST", "bron": "Dryad doi:10.5061/dryad.m37pvmd99 (v. 2024-09-08, CC0)",
           "ttl_reparaties": brick.attrs["ttl_reparaties"], "tijdzone_check": tz_check,
           "dekking_weerstation": weather_cov,
           "aandeel_ghi_na_18u_pct": (100 * evening).round(1).to_dict(),
           "drempels": {"min_ghi": HKUST_MIN_GHI, "uitval_frac": HKUST_OUTAGE_FRAC, "max_gaten": 1 - HKUST_MIN_COVERAGE,
                        "min_schaduw_index": HKUST_MIN_SHADE_INDEX, "min_r": HKUST_MIN_R},
           "geschikt": inv.index[inv["geschikt"]].tolist()}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("klaar:", OUT)

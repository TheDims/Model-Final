# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 1  -  UK PV (Open Climate Fix): 40 systemen kiezen en alleen die ophalen
=======================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Doel van 5b: echte dag-vooruit-fout (TAAK 4b) en horizon dag 1-15 (TAAK 7) met archief-weersvoorspellingen
(Open-Meteo Previous Runs vanaf jan 2024, ECMWF IFS Single Runs vanaf 14 maart 2024). Het klimaat is anders dan
India: dit test het VOORSPELLEN, niet de locatie.

Wat dit script doet (nog GEEN model):
  1. metadata + bad_data.csv; filter op huisgrootte, dakhoek, richting, periode (BESLISSINGEN #33);
  2. per maand tellen hoeveel halve uren elk systeem heeft (alleen kolom ss_id, snel) -> complete systemen;
  3. een pool van 150 kandidaten ophalen (polars, alleen die ss_id's) en de kwaliteit checken (README-advies);
  4. 40 systemen kiezen, verspreid over 5 breedtebanden, max. 1 per locatie; opslaan als parquet.
Nodig: HF_TOKEN (zie README). Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Duurt enkele minuten.
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
from solarcore.config import (UKPV_CLEAN, UKPV_RESULTS, UKPV_PERIOD, UKPV_N_SYSTEMS, UKPV_KWP, UKPV_TILT,  # noqa: E402
                              UKPV_ORIENT, UKPV_MIN_ROWS, UKPV_MAX_BAD_DAYS, UKPV_POOL, UKPV_LAT_BANDS,
                              UKPV_YIELD, UKPV_PMAX, RANDOM_STATE, BLUE, GREY)
from solarcore.ukpv import read_metadata, read_bad, month_counts, stream_systems, add_sun, quality  # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = UKPV_RESULTS / "stap1_systemen"
OUT.mkdir(parents=True, exist_ok=True)
UKPV_CLEAN.mkdir(parents=True, exist_ok=True)
style()
rng = np.random.default_rng(RANDOM_STATE)
p0, p1 = pd.Timestamp(UKPV_PERIOD[0], tz="UTC"), pd.Timestamp(UKPV_PERIOD[1], tz="UTC") + pd.Timedelta("1D")

# %% 1. Metadata-filter
meta = read_metadata()
bad = read_bad()
bad_in = bad[(bad.start_datetime_GMT < p1) & (bad.end_datetime_GMT > p0)].ss_id.unique()
f = meta[(meta.start_datetime_GMT <= p0) & (meta.end_datetime_GMT >= p1 - pd.Timedelta("1D"))
         & meta.kWp.between(*UKPV_KWP) & meta.tilt.between(*UKPV_TILT) & meta.orientation.between(*UKPV_ORIENT)
         & ~meta.index.isin(bad_in)]
print(f"metadata: {len(meta)} systemen -> {len(f)} na filter (periode, kWp, dakhoek, richting, geen bad_data)")

# %% 2. Compleetheid per maand (aantal halve uren / verwacht)
counts = month_counts(UKPV_PERIOD).reindex(f.index)
# Verwacht = wat de dataset die maand maximaal heeft (99e percentiel van alle systemen), niet de kalender:
# in sommige maanden ontbreken hele dagen voor bijna alle systemen tegelijk (storing in de datastroom).
calendar = pd.Series({m: pd.Period(m).days_in_month * 48 for m in counts.columns})
expected = counts.quantile(0.99)
print("dataset-dekking per maand (P99 systemen / kalender):", (expected / calendar).round(3).to_dict())
frac = counts.div(expected, axis=1)
complete = frac.index[(frac >= UKPV_MIN_ROWS).all(axis=1)]
print(f"compleet (elke maand >= {UKPV_MIN_ROWS:.0%}): {len(complete)} systemen")
frac.round(4).to_csv(OUT / "compleetheid_per_maand.csv")

# %% 3. Pool ophalen + kwaliteit (max. 1 systeem per afgeronde locatie, gespreid over de breedtebanden)
cand = f.loc[complete].copy()
cand["band"] = pd.cut(cand.latitude_rounded, UKPV_LAT_BANDS, labels=False)
cand = cand.sample(frac=1, random_state=RANDOM_STATE).drop_duplicates(["latitude_rounded", "longitude_rounded"])
per_band = int(np.ceil(UKPV_POOL / (len(UKPV_LAT_BANDS) - 1)))
pool = cand.groupby("band", group_keys=False).head(per_band)
print(f"pool: {len(pool)} kandidaten; per band: {pool.band.value_counts().sort_index().to_dict()}")
raw = add_sun(stream_systems(pool.index, UKPV_PERIOD), meta)
q, bad_day = quality(raw, meta)
q = pool.join(q)
q["ok"] = (q.negatief == 0) & (q.te_hoog == 0) & (q.slechte_dagen_frac < UKPV_MAX_BAD_DAYS)
# plausibiliteit (#35): klopt de opgegeven kWp en werkt het systeem normaal?
q["plausibel"] = q.jaaropbrengst_kWh_per_kWp.between(*UKPV_YIELD) & q.P_max_per_kWp.between(*UKPV_PMAX)
q["ok"] &= q.plausibel
print(f"kwaliteit OK: {int(q.ok.sum())} van {len(q)}")

# %% 4. Kiezen: evenveel per band (vaste seed), aanvullen uit andere banden als een band te weinig heeft
good = q[q.ok]
n_band = UKPV_N_SYSTEMS // (len(UKPV_LAT_BANDS) - 1)
chosen = good.groupby("band", group_keys=False).head(n_band)
if len(chosen) < UKPV_N_SYSTEMS:
    chosen = pd.concat([chosen, good.drop(chosen.index).head(UKPV_N_SYSTEMS - len(chosen))])
chosen = chosen.sort_values("latitude_rounded")
cols = ["latitude_rounded", "longitude_rounded", "kWp", "tilt", "orientation", "band", "dagen_nachtopwek",
        "dagen_nul_bij_zon", "slechte_dagen_frac", "P_max_per_kWp", "jaaropbrengst_kWh_per_kWp"]
print(f"\nGEKOZEN: {len(chosen)} systemen")
print(chosen[cols].to_string())
q.to_csv(OUT / "pool_kwaliteit.csv")
chosen[cols].to_csv(OUT / "gekozen_systemen.csv")
sel = raw[raw.ss_id.isin(chosen.index)].assign(SLECHTE_DAG=bad_day[raw.ss_id.isin(chosen.index)].values)
sel.to_parquet(UKPV_CLEAN / "ukpv_30min.parquet", index=False)
print("opgeslagen:", UKPV_CLEAN / "ukpv_30min.parquet", sel.shape)

# %% 5. Grafieken
fig, ax = plt.subplots(figsize=(5, 6.5))
ax.scatter(f.longitude_rounded, f.latitude_rounded, s=2, color=GREY, alpha=0.3, label=f"na metadata-filter ({len(f)})")
ax.scatter(chosen.longitude_rounded, chosen.latitude_rounded, s=28, color=BLUE, edgecolor="white", label=f"gekozen ({len(chosen)})")
for b in UKPV_LAT_BANDS[1:-1]:
    ax.axhline(b, color=GREY, lw=0.6, ls="--")
ax.set(title="UK PV: gekozen systemen", xlabel="lengtegraad (°)", ylabel="breedtegraad (°)")
ax.legend(fontsize=7, loc="upper left")
save(fig, OUT / "01_kaart_systemen.png")

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
axs[0].hist(chosen.kWp, bins=12, color=BLUE); axs[0].set(title="Grootte", xlabel="kWp", ylabel="aantal")
axs[1].hist(chosen.orientation, bins=12, color=BLUE); axs[1].set(title="Richting (180 = zuid)", xlabel="graden", ylabel="aantal")
axs[2].hist(chosen.tilt, bins=12, color=BLUE); axs[2].set(title="Dakhoek", xlabel="graden", ylabel="aantal")
save(fig, OUT / "02_systeemgegevens.png")

summary = {"taak": "5b stap 1 - systemen kiezen UK PV", "bron": "huggingface.co/datasets/openclimatefix/uk_pv (CC BY 4.0)",
           "periode_UTC": UKPV_PERIOD, "na_metadata_filter": len(f), "compleet": len(complete), "pool": len(pool),
           "kwaliteit_ok": int(q.ok.sum()), "gekozen": [int(i) for i in chosen.index],
           "regels": {"kWp": UKPV_KWP, "tilt": UKPV_TILT, "orientation": UKPV_ORIENT, "min_rijen_per_maand": UKPV_MIN_ROWS,
                      "max_slechte_dagen": UKPV_MAX_BAD_DAYS, "opbrengst_kWh_per_kWp": UKPV_YIELD,
                      "Pmax_per_kWp": UKPV_PMAX, "banden": UKPV_LAT_BANDS, "seed": RANDOM_STATE}}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("klaar:", OUT)

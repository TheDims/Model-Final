# -*- coding: utf-8 -*-
"""
FASE 0  -  Data begrijpen en opschonen (CRISP-DM: data understanding + data preparation)
========================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Wat dit script doet:
  1. de 4 Kaggle-bestanden inlezen en controleren (check_df);
  2. opschonen volgens AGENTS.md §5.2 (datums, DC-fix, uitval, gemiddelde per omvormer, rooster);
  3. de 30/70-split in hele dagen bepalen en printen;
  4. sanity checks (DC/AC, % uitval, eenheid straling, NaN);
  5. opslaan: data/clean/<RUN>/plant_15min.csv + resultaten/<RUN>/fase0/samenvatting.json;
  6. grafieken in resultaten/fase0/.
Nog GEEN model. Runnen: F5 in Spyder of in VS Code (met de Anaconda-Python). Duurt < 1 minuut.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (RESULTS, CLEAN, CLEAN_FILE, PLANTS, PERIOD, SENSOR_COLS, PLANT_COLORS,  # noqa: E402
                              BLUE, ORANGE, GREY, INK, INK2, RUN_NAME, TRAIN_FRACTION)
from solarcore.data_prep import (read_generation, read_sensor, check_df, plant_15min, dc_ac_per_day,  # noqa: E402
                                 inverter_daily_kwh, flag_dips)
from solarcore.split import day_split, add_day_index   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

OUT = RESULTS / "fase0"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160)

# %% 1. Ruwe bestanden controleren
checks = {}
for p in PLANTS:
    raw_gen = pd.read_csv(ROOT / "data" / f"Plant_{p}_Generation_Data.csv")
    dc_ac_raw = raw_gen.loc[raw_gen.AC_POWER > 0, "DC_POWER"].sum() / raw_gen.loc[raw_gen.AC_POWER > 0, "AC_POWER"].sum()
    checks[f"Plant {p} generation"] = {**check_df(read_generation(p), f"Plant {p} generation"),
                                       "DC/AC_voor_fix": float(dc_ac_raw)}
    checks[f"Plant {p} sensor"] = check_df(read_sensor(p), f"Plant {p} sensor")

# %% 2. Opschonen -> één tabel per kwartier, beide plants
parts, gens, stats = [], {}, {}
for p in PLANTS:
    d, gen, st = plant_15min(p)
    parts.append(d); gens[p] = gen; stats[f"Plant {p}"] = st
clean = add_day_index(pd.concat(parts, ignore_index=True), PERIOD[0])
clean["HOUR"] = clean.DATE_TIME.dt.hour + clean.DATE_TIME.dt.minute / 60

# %% 3. 30/70-split in hele dagen
sp = day_split(clean.DATE_TIME)
print(f"\nRun {RUN_NAME} | split: {sp['n_dagen']} dagen -> training {sp['n_opzet']} dagen ({sp['opzet_van_tot'][0]} t/m "
      f"{sp['opzet_van_tot'][1]}), test {sp['n_test']} dagen ({sp['test_van_tot'][0]} t/m {sp['test_van_tot'][1]})")

# plant-brede dips (BESLISSINGEN #10, toegevoegd 28 okt na fase 1): verhouding alleen uit de opzetdagen
clean, dip_k = flag_dips(clean, range(1, sp["n_opzet"] + 1))
for p in PLANTS:
    stats[f"Plant {p}"]["dip_kwartieren"] = int(clean[clean.PLANT == p].DIP.sum())
    stats[f"Plant {p}"]["dip_verhouding_AC_per_IRR"] = float(dip_k[p])
print("Plant-brede dips:", {f"Plant {p}": int(clean[clean.PLANT == p].DIP.sum()) for p in PLANTS})

# %% 4. Sanity checks
for p in PLANTS:
    dcac = dc_ac_per_day(gens[p])
    stats[f"Plant {p}"]["DC/AC_na_fix_mediaan"] = float(dcac.median())
    stats[f"Plant {p}"]["DC/AC_voor_fix"] = checks[f"Plant {p} generation"]["DC/AC_voor_fix"]
print("\n=== Sanity checks per plant ===")
print(pd.DataFrame({k: {m: v for m, v in s.items() if not isinstance(v, dict)} for k, s in stats.items()}).round(3).to_string())
print("\nSensorgaten vóór/na interpolatie (<= 1 uur):")
for k, s in stats.items():
    print(f"  {k}: voor {s['sensor_gaten_voor']} -> na {s['sensor_gaten_na_interpolatie']}")

# %% 5. Opslaan
cols = ["DATE_TIME", "PLANT", "DAY", "AC_KW", "N_INV", "N_UITVAL", "UITVAL", "DIP"] + SENSOR_COLS
clean[cols].to_csv(CLEAN_FILE, index=False)
summary = {"fase": 0, "run": RUN_NAME, "bestand": str(CLEAN_FILE.relative_to(ROOT)), "rijen": len(clean),
           "split": {"n_dagen": sp["n_dagen"], "opzet_dagen": f"dag 1-{sp['n_opzet']}",
                     "opzet_datums": [str(x) for x in sp["opzet_van_tot"]],
                     "test_dagen": f"dag {sp['n_opzet'] + 1}-{sp['n_dagen']}",
                     "test_datums": [str(x) for x in sp["test_van_tot"]]},
           "per_plant": stats, "checks_ruwe_bestanden": checks,
           "eenheden": {"AC_KW": "kW per actieve omvormer", "IRRADIATION": "kW/m² (Kaggle; ERA5 is W/m²)",
                        "temperaturen": "°C"}}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False),
                                       encoding="utf-8")
print(f"\nOpgeslagen: {CLEAN_FILE.relative_to(ROOT)} ({len(clean)} rijen) en {(OUT / 'samenvatting.json').relative_to(ROOT)}")

# %% 6. Grafieken
style()
day = clean[clean.IRRADIATION > 0.01]             # overdag volgens de sensor
split_x = sp["n_opzet"] + 0.5                     # grens tussen dag 10 en 11

# 6.1 uurprofiel: gemiddeld vermogen per uur (met 25-75 %-band)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for p in PLANTS:
    g = clean[clean.PLANT == p].groupby("HOUR").AC_KW
    ax.fill_between(g.mean().index, g.quantile(0.25), g.quantile(0.75), color=PLANT_COLORS[p], alpha=0.15, lw=0)
    ax.plot(g.mean().index, g.mean(), color=PLANT_COLORS[p], lw=2, label=f"Plant {p} (gemiddelde, band = 25-75 %)")
ax.set_xlim(5, 19.5); ax.set_xticks(range(5, 20, 2))
ax.set_xlabel("tijd van de dag (uur, IST)"); ax.set_ylabel("AC-vermogen per omvormer (kW)")
ax.set_title("Dagprofiel van de opwek (alle 34 dagen)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
save(fig, OUT / "01_uurprofiel.png")

# 6.2 boxplot per omvormer: opwek per dag (kWh), ruwe data inclusief uitval
fig, axs = plt.subplots(2, 1, figsize=(10, 5.6))
for ax, p in zip(axs, PLANTS):
    e = inverter_daily_kwh(gens[p])
    order = e.groupby("SOURCE_KEY").kWh.median().sort_values().index
    bp = ax.boxplot([e[e.SOURCE_KEY == k].kWh for k in order], patch_artist=True, widths=0.6,
                    medianprops=dict(color=INK), flierprops=dict(marker="o", ms=3, mfc=GREY, mec=GREY))
    for b in bp["boxes"]:
        b.set(facecolor=PLANT_COLORS[p], alpha=0.35, edgecolor=PLANT_COLORS[p])
    ax.set_xticks(range(1, len(order) + 1), [f"{i + 1}" for i in range(len(order))])
    ax.set_title(f"Plant {p}: opwek per omvormer per dag (ruwe data, inclusief uitval)")
    ax.set_ylabel("kWh per dag"); ax.set_xlabel("omvormer (gesorteerd op mediaan)")
save(fig, OUT / "02_boxplot_per_omvormer.png")

# 6.3 straling tegen AC (na opschonen)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, p in zip(axs, PLANTS):
    s = clean[(clean.PLANT == p) & clean.AC_KW.notna()]
    ax.scatter(s.IRRADIATION, s.AC_KW, s=5, alpha=0.35, color=PLANT_COLORS[p], lw=0)
    r = s[["IRRADIATION", "AC_KW"]].corr().iloc[0, 1]
    ax.set_title(f"Plant {p}: straling tegen opwek (r = {r:.3f})")
    ax.set_xlabel("instraling plant-sensor (kW/m²)")
axs[0].set_ylabel("AC-vermogen per omvormer (kW)")
save(fig, OUT / "03_straling_vs_ac.png")

# 6.4 correlatiematrix (overdag, beide plants)
c = day.assign(**{"module - lucht": day.MODULE_TEMPERATURE - day.AMBIENT_TEMPERATURE})
names = {"AC_KW": "opwek (AC_KW)", "IRRADIATION": "instraling", "MODULE_TEMPERATURE": "moduletemp.",
         "AMBIENT_TEMPERATURE": "luchttemp.", "module - lucht": "module - lucht", "HOUR": "uur"}
C = c[list(names)].dropna().corr()
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", ORANGE])
fig, ax = plt.subplots(figsize=(6, 4.8))
im = ax.imshow(C.values, cmap=DIV, vmin=-1, vmax=1)
lab = list(names.values())
ax.set_xticks(range(len(lab)), lab, rotation=35, ha="right"); ax.set_yticks(range(len(lab)), lab)
ax.grid(False); ax.tick_params(length=0)
for i in range(len(lab)):
    for j in range(len(lab)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.7 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("correlatie (Pearson)"); cb.outline.set_visible(False)
ax.set_title("Correlatie overdag (plant-sensor, beide plants)")
save(fig, OUT / "04_correlatiematrix.png")

# 6.5 dagopbrengst per dag per plant, met de 30/70-grens
dy = clean.groupby(["PLANT", "DAY"]).agg(kWh=("AC_KW", lambda x: x.sum() * 0.25),
                                          nan=("AC_KW", lambda x: x[clean.loc[x.index, "IRRADIATION"] > 0.01].isna().sum()))
fig, ax = plt.subplots(figsize=(9, 3.8))
for p in PLANTS:
    s = dy.loc[p]
    ax.plot(s.index, s.kWh, marker="o", ms=5, mec="white", mew=1.5, lw=2, color=PLANT_COLORS[p], label=f"Plant {p}")
ax.axvspan(0.5, split_x, color=GREY, alpha=0.12, lw=0)
ax.axvline(split_x, color=INK2, ls="--", lw=1)
ax.text(split_x - 0.3, ax.get_ylim()[0] + 0.3, f"training ({TRAIN_FRACTION:.0%})", ha="right", va="bottom", color=INK2)
ax.text(split_x + 0.3, ax.get_ylim()[0] + 0.3, f"test ({1 - TRAIN_FRACTION:.0%})", ha="left", va="bottom", color=INK2)
ax.set_xlim(0.5, sp["n_dagen"] + 0.5)
ax.set_xlabel("dag (1 = 15 mei 2020, 34 = 17 juni 2020)"); ax.set_ylabel("opwek per omvormer (kWh/dag)")
ax.set_title(f"Dagopbrengst per plant met de split-grens ({RUN_NAME})"); ax.legend(loc="lower left")
save(fig, OUT / "05_dagopbrengst_split.png")

# 6.6 DC/AC-verhouding per dag (na de DC-fix van Plant 1)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for p in PLANTS:
    r = dc_ac_per_day(gens[p])
    x = (r.index - pd.Timestamp(PERIOD[0]).normalize()).days + 1             # dagnummer, zoals grafiek 5
    ax.plot(x, r.values, marker="o", ms=4, lw=1.8, color=PLANT_COLORS[p], label=f"Plant {p}")
ax.set_ylabel("DC / AC (-)"); ax.set_xlabel("dag (1 = 15 mei 2020, 34 = 17 juni 2020)")
ax.set_title("DC/AC-verhouding per dag (Plant 1 na de fix DC / 10)"); ax.legend()
save(fig, OUT / "06_dc_ac_per_dag.png")

# 6.7 moduletemperatuur min luchttemperatuur per uur
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for p in PLANTS:
    s = clean[clean.PLANT == p]
    g = (s.MODULE_TEMPERATURE - s.AMBIENT_TEMPERATURE).groupby(s.HOUR).mean()
    ax.plot(g.index, g.values, lw=2, color=PLANT_COLORS[p], label=f"Plant {p}")
ax.axhline(0, color=GREY, lw=1)
ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("tijd van de dag (uur, IST)")
ax.set_ylabel("module - lucht (°C)")
ax.set_title("Hoeveel warmer is het paneel dan de lucht? (gemiddeld per kwartier)"); ax.legend()
save(fig, OUT / "07_module_min_lucht.png")
plt.show()
print("\nKlaar met fase 0.")

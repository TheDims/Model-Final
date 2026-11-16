# -*- coding: utf-8 -*-
"""
Wolken via de satelliet  -  controle in India (Kaggle, run 2): Himawari i.p.v. ERA5 (BESLISSINGEN #48)
=====================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Grens vooraf: "echte impact" = LR GHI met satelliet op dag 25-34 >= 1 pp lager nRMSE dan met ERA5 (zelfde
kwartieren) EN aantoonbaar in de bootstrap; anders verwaarloosbaar -> laten.
De plantlocaties zijn GESCHAT (+-0,5 graad). Daarom twee varianten:
  (a) satelliet op de geschatte locatie; (b) beste roosterpunt binnen +-0,5 graad, gekozen op dag 1-24.
Runnen: F5 in Spyder of VS Code (Anaconda). Internet alleen de eerste keer (cache).
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
from solarcore.config import RESULTS, PLANTS, PLANT_LOCATIONS, TZ, DATA, MIN_COSZ, BLUE, ORANGE, GREY  # noqa: E402
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.split import day_split   # noqa: E402
from solarcore.forecast_api import satellite, satellite_15min   # noqa: E402
from solarcore.models import _lr   # noqa: E402
from solarcore.evaluate import metrics, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "satelliet"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DATA / "extern" / "kaggle_satelliet"
pd.set_option("display.width", 220)
style()
SHIFTS = [-20, -10, 0, 10, 20]
OFFSETS = np.arange(-0.5, 0.51, 0.25)
IMPACT_PP = 1.0
PERIOD = ("2020-05-14", "2020-06-18")

# %% 1. Data (zelfde tabel als fase 3) en split
kd = weather_frame(15)
sp = day_split(kd.DATE_TIME)
TRAIN, TEST = sp["train_dagnrs"], sp["test_dagnrs"]
kd = add_target(kd, capacity(kd[kd.DAY.isin(TRAIN)]))
base = kd.OK & ~kd.DIP
print(f"Run split70: train dag {TRAIN[0]}-{TRAIN[-1]}, test dag {TEST[0]}-{TEST[-1]}")


def sat_at(lat, lon):
    cache = CACHE / f"sat_{lat:.2f}_{lon:.2f}.csv"
    new = not cache.exists()
    s = satellite(lat, lon, *PERIOD, cache, tz=TZ)
    if new:
        time.sleep(1)
    return s


# %% 2. Locatie en tijdverschuiving per plant (alleen dag 1-24)
rows, choice = [], {}
for p in PLANTS:
    lat0, lon0 = PLANT_LOCATIONS[p]
    tr = kd[(kd.PLANT == p) & base & kd.DAY.isin(TRAIN)]
    for dlat in OFFSETS:
        for dlon in OFFSETS:
            s = sat_at(round(lat0 + dlat, 2), round(lon0 + dlon, 2))
            for sh in SHIFTS:
                g = tr.DATE_TIME.map(satellite_15min(s, sh))
                rows.append({"plant": p, "lat": lat0 + dlat, "lon": lon0 + dlon, "verschuiving_min": sh,
                             "r": tr.AC_KW.corr(g), "geschat": dlat == 0 and dlon == 0})
grid = pd.DataFrame(rows)
for p in PLANTS:
    g = grid[grid.plant == p]
    est = g[g.geschat].sort_values("r").iloc[-1]
    best = g.sort_values("r").iloc[-1]
    choice[p] = {"a": (est.lat, est.lon, int(est.verschuiving_min), est.r), "b": (best.lat, best.lon, int(best.verschuiving_min), best.r)}
    print(f"Plant {p}: (a) geschat {est.lat:.2f}/{est.lon:.2f} shift {est.verschuiving_min:+.0f} r {est.r:.3f} | "
          f"(b) beste {best.lat:.2f}/{best.lon:.2f} shift {best.verschuiving_min:+.0f} r {best.r:.3f} "
          f"| ERA5 r {kd[(kd.PLANT == p) & base & kd.DAY.isin(TRAIN)].pipe(lambda d: d.AC_KW.corr(d.GHI_W)):.3f}")
for v in ("a", "b"):
    kd[f"GHI_SAT_{v}"] = np.nan
    for p in PLANTS:
        lat, lon, sh, _ = choice[p][v]
        m = kd.PLANT == p
        kd.loc[m, f"GHI_SAT_{v}"] = kd.loc[m, "DATE_TIME"].map(satellite_15min(sat_at(round(lat, 2), round(lon, 2)), sh)).values

# %% 3. Controle tegen de plant-sensor (alle dagen; sensor is hier wel betrouwbaar)
sens = {p: {"r_sensor_ERA5": kd[(kd.PLANT == p) & base].pipe(lambda d: d.IRRADIATION.corr(d.GHI_W)),
            "r_sensor_satelliet_a": kd[(kd.PLANT == p) & base].pipe(lambda d: d.IRRADIATION.corr(d.GHI_SAT_a)),
            "r_sensor_satelliet_b": kd[(kd.PLANT == p) & base].pipe(lambda d: d.IRRADIATION.corr(d.GHI_SAT_b))}
        for p in PLANTS}
print("Correlatie per kwartier met de plant-sensor:", {p: {k: round(v, 3) for k, v in d.items()} for p, d in sens.items()})

# %% 4. LR GHI: ERA5 vs satelliet (a, b) vs sensor (bovengrens), zelfde kwartieren, bevroren op dag 1-24
MODELS = {"LR GHI (ERA5)": ["GHI_W"], "LR GHI (satelliet, geschatte locatie)": ["GHI_SAT_a"],
          "LR GHI (satelliet, beste punt)": ["GHI_SAT_b"], "LR straling (plant-sensor, bovengrens)": ["IRRADIATION"]}
ok = base & kd[["GHI_SAT_a", "GHI_SAT_b", "IRRADIATION"]].notna().all(axis=1)
trn, tst = kd[ok & kd.DAY.isin(TRAIN)], kd[ok & kd.DAY.isin(TEST)]
pr = {n: np.clip(_lr().fit(trn[c], trn.FRAC).predict(tst[c]), 0, None) * (tst.COSZ.values > MIN_COSZ) for n, c in MODELS.items()}
rows = []
for plant, m in [("samen", np.ones(len(tst), bool))] + [(f"Plant {p}", (tst.PLANT == p).values) for p in PLANTS]:
    rows += [{"model": n, "plant": plant, **metrics(tst.FRAC[m], y[m], tst[m])} for n, y in pr.items()]
res = pd.DataFrame(rows)
tab = res.pivot(index="model", columns="plant", values="nRMSE_%").loc[list(MODELS)]
print(f"\n=== Test dag {TEST[0]}-{TEST[-1]}: nRMSE % (bevroren op dag 1-24) ===")
print(tab.round(2).to_string())
boot_m, boot_d = bootstrap_days(tst, pr, pairs=[("LR GHI (satelliet, geschatte locatie)", "LR GHI (ERA5)"),
                                                 ("LR GHI (satelliet, beste punt)", "LR GHI (ERA5)")])
print("\nBootstrap (90 %):"); print(boot_d.round(2).to_string())
verdict = {}
for v, n in (("a", "LR GHI (satelliet, geschatte locatie)"), ("b", "LR GHI (satelliet, beste punt)")):
    d = boot_d.loc[f"{n} - LR GHI (ERA5)"]
    verdict[v] = bool((-d.verschil_pp >= IMPACT_PP) and d.hoog_pp < 0)
print(f"\nGrens #48 (>= {IMPACT_PP} pp beter en aantoonbaar): (a) {'IMPACT' if verdict['a'] else 'verwaarloosbaar'}, "
      f"(b) {'IMPACT' if verdict['b'] else 'verwaarloosbaar'}")

grid.round(4).to_csv(OUT / "locatie_rooster.csv", index=False)
tab.round(4).to_csv(OUT / "test_nrmse.csv"); boot_d.round(4).to_csv(OUT / "bootstrap_verschillen.csv")
(OUT / "samenvatting.json").write_text(json.dumps({
    "vraag": "satelliet vs ERA5 in India (BESLISSINGEN #48)", "keuze_locatie": {str(p): v for p, v in choice.items()},
    "sensor_correlatie": sens, "test_nRMSE": tab.round(4).to_dict("index"), "impact": verdict},
    indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 5. Grafiek
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(MODELS))
for i, (col, c) in enumerate(zip(["Plant 1", "Plant 2", "samen"], (BLUE, ORANGE, GREY))):
    ax.bar(x + (i - 1) * 0.27, tab[col], width=0.27, color=c, label=col)
ax.set_xticks(x, [n.replace(" (", "\n(") for n in MODELS], fontsize=7)
ax.set(title=f"India (Kaggle), test dag {TEST[0]}-{TEST[-1]}: straling uit ERA5, satelliet of sensor", ylabel="nRMSE (%)")
ax.grid(axis="x", visible=False)
ax.legend(fontsize=8)
save(fig, OUT / "01_satelliet_india.png")
print("klaar:", OUT)

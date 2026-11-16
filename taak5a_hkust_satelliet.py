# -*- coding: utf-8 -*-
"""
Wolken via de satelliet  -  HKUST: Himawari-straling i.p.v. ERA5 als "weer achteraf" (BESLISSINGEN #46)
=======================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Vraag: zit de satelliet (5 km, 10 min) dichter bij wat het dak ziet dan ERA5 (31 km, 1 uur)?
  1. Satellietstraling ophalen (Open-Meteo Satellite Radiation API, boven Hong Kong = JMA Himawari), 2021-2023;
  2. tijdverschuiving kiezen op 2021 (correlatie met de opwek); 10 min -> 15 min;
  3. controle tegen de stralingssensor van het weerstation (betrouwbare periode) naast ERA5;
  4. LR GHI met satelliet vs ERA5 (zelfde kwartieren): train 2021 -> val 2022, train 2021 + 2022 -> test 2023.
Satelliet is net als ERA5 weer ACHTERAF, geen voorspelling. Runnen: F5 in Spyder of VS Code (Anaconda).
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
from solarcore.config import HKUST_RESULTS, HKUST_LOCATION, HKUST_TZ, HKUST_STATIONS, DATA, BLUE, ORANGE, GREY  # noqa: E402
from solarcore.hkust import load_clean, load_era5_hk, add_era5, capacity   # noqa: E402
from solarcore.forecast_api import satellite, satellite_15min   # noqa: E402
from solarcore.models import _lr, _ridge   # noqa: E402
from solarcore.evaluate import metrics, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = HKUST_RESULTS / "stap5_satelliet"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
SHIFTS = [-20, -10, 0, 10, 20]                 # minuten; 0 = waarde op t is het gemiddelde van (t - 10, t]
BETTER_PP = 1.0                                # regel #46
MODELS = {"LR GHI (ERA5)": (["GHI_W"], _lr), "LR GHI (satelliet)": (["GHI_SAT"], _lr),
          "Ridge (satelliet + ERA5)": (["GHI_SAT", "GHI_W"], _ridge)}


sat_15min = satellite_15min


# %% 1. Data: HKUST + ERA5 (zelfde als stap 3) + satelliet
step3 = json.loads((HKUST_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
h, elev = load_era5_hk()
data = add_era5(load_clean(), h, elev, step3["tijdverschuiving_min"])
sat = satellite(*HKUST_LOCATION, "2021-01-01", "2023-12-31", DATA / "extern" / "hkust" / "satelliet_himawari_10min.csv",
                tz=HKUST_TZ)
print(f"satelliet: {len(sat)} waarden (10 min), {sat.GHI_SAT.notna().mean():.1%} gevuld, "
      f"{sat.DATE_TIME.min()} .. {sat.DATE_TIME.max()}")

# %% 2. Tijdverschuiving (alleen 2021, met de opwek)
y21 = data.OK & (data.DATE_TIME.dt.year == 2021)
rows = []
for sh in SHIFTS:
    g = data.DATE_TIME.map(sat_15min(sat, sh))
    rows += [{"verschuiving_min": sh, "station": st, "r": d.P_KW.corr(g[d.index])} for st, d in data[y21].groupby("PLANT")]
shift = pd.DataFrame(rows).pivot(index="verschuiving_min", columns="station", values="r")
SAT_SHIFT = int(shift.mean(axis=1).idxmax())
print("Tijdverschuiving satelliet (r met de opwek, 2021):", shift.mean(axis=1).round(4).to_dict(), f"-> {SAT_SHIFT:+d} min")
data["GHI_SAT"] = data.DATE_TIME.map(sat_15min(sat, SAT_SHIFT)).values

# %% 3. Controle tegen de stralingssensor (betrouwbare periode; sensor meet ~20 % te laag -> alleen r)
st0 = data[data.PLANT == HKUST_STATIONS[0]]
c = st0[st0.SENSOR_OK & st0.DAG & st0.GHI_SENSOR.notna() & st0.GHI_SAT.notna() & st0.GHI_W.notna()]
check = {"r_sensor_satelliet": float(c.GHI_SENSOR.corr(c.GHI_SAT)), "r_sensor_ERA5": float(c.GHI_SENSOR.corr(c.GHI_W)),
         "r_opwek_satelliet": float(data[data.OK].groupby("PLANT").apply(lambda d: d.P_KW.corr(d.GHI_SAT)).mean()),
         "r_opwek_ERA5": float(data[data.OK].groupby("PLANT").apply(lambda d: d.P_KW.corr(d.GHI_W)).mean()),
         "kwartieren": int(len(c))}
print("Correlatie per kwartier:", {k: round(v, 3) for k, v in check.items()})

# %% 4. Modellen: zelfde kwartieren voor alle bronnen
OK = data.OK & data.GHI_SAT.notna()
yr = data.DATE_TIME.dt.year


def run(tr, te):
    cap = capacity(data[tr & OK])
    d = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    trn, tst = d[tr & OK], d[te & OK]
    return tst, {n: np.clip(make().fit(trn[c], trn.FRAC).predict(tst[c]), 0, None) for n, (c, make) in MODELS.items()}


te_v, pr_v = run(yr == 2021, yr == 2022)
te_t, pr_t = run(yr.isin([2021, 2022]), yr == 2023)
val = pd.DataFrame({n: metrics(te_v.FRAC, p, te_v) for n, p in pr_v.items()}).T
test = pd.DataFrame({n: metrics(te_t.FRAC, p, te_t) for n, p in pr_t.items()}).T
season = pd.DataFrame({n: {s: metrics(te_t.FRAC[m], p[m], te_t[m])["nRMSE_%"]
                           for s, m in ((s, (te_t.SEASON == s).values) for s in te_t.SEASON.unique())}
                       for n, p in pr_t.items()}).T
station = pd.DataFrame({n: {s: metrics(te_t.FRAC[m], p[m], te_t[m])["nRMSE_%"]
                            for s, m in ((s, (te_t.PLANT == s).values) for s in HKUST_STATIONS)}
                        for n, p in pr_t.items()}).T
boot_m, boot_d = bootstrap_days(te_t, pr_t, pairs=[("LR GHI (satelliet)", "LR GHI (ERA5)"),
                                                    ("Ridge (satelliet + ERA5)", "LR GHI (ERA5)")])
COLS = ["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]
print("\n=== Validatie 2022 (train 2021) ==="); print(val[COLS].round(3).to_string())
print("\n=== Test 2023 (train 2021 + 2022), zelfde kwartieren ==="); print(test[COLS].round(3).to_string())
print("\nPer seizoen (nRMSE %):"); print(season.round(2).to_string())
print("\nPer station (nRMSE %):"); print(station.round(2).to_string())
print("\nBootstrap (90 %):"); print(boot_d.round(2).to_string())
gain = test.loc["LR GHI (ERA5)", "nRMSE_%"] - test.loc["LR GHI (satelliet)", "nRMSE_%"]
print(f"\nRegel #46 (satelliet >= {BETTER_PP} pp beter op 2023): {gain:.2f} pp -> {'JA' if gain >= BETTER_PP else 'NEE'}")

for name, df in (("tijdverschuiving", shift), ("validatie_2022", val), ("test_2023", test), ("test_2023_per_seizoen", season),
                 ("test_2023_per_station", station), ("bootstrap_verschillen", boot_d)):
    df.round(4).to_csv(OUT / f"{name}.csv")
(OUT / "samenvatting.json").write_text(json.dumps({
    "vraag": "satelliet (Himawari) vs ERA5 als weer achteraf (BESLISSINGEN #46)", "satelliet_verschuiving_min": SAT_SHIFT,
    "controle": check, "validatie_2022": val.round(4).to_dict("index"), "test_2023": test.round(4).to_dict("index"),
    "winst_pp": round(float(gain), 3), "regel_gehaald": bool(gain >= BETTER_PP)}, indent=2, ensure_ascii=False,
    default=float), encoding="utf-8")

# %% 5. Grafiek: voorbeelddag + residuen per bron
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
d0 = te_t[te_t.PLANT == HKUST_STATIONS[3]]
day = d0.groupby(d0.DATE_TIME.dt.date).FRAC.std().idxmax()                    # wisselvallige dag
m = (d0.DATE_TIME.dt.date == day).values
t = d0.DATE_TIME[m].dt.hour + d0.DATE_TIME[m].dt.minute / 60
idx = (te_t.PLANT == HKUST_STATIONS[3]).values
axs[0].plot(t, d0.FRAC[m] * 100, color="k", lw=1.5, label="gemeten")
for n, col in (("LR GHI (ERA5)", GREY), ("LR GHI (satelliet)", BLUE)):
    axs[0].plot(t, pr_t[n][idx][m] * 100, color=col, lw=1.6, label=n)
axs[0].set(title=f"{HKUST_STATIONS[3]}, {day} (wisselvallig)", xlabel="uur (HKT)", ylabel="opwek (% van capaciteit)")
axs[0].legend(fontsize=7)
v = test.loc[list(MODELS), "nRMSE_%"]
axs[1].bar(range(len(v)), v.values, color=[GREY, BLUE, ORANGE])
axs[1].set_xticks(range(len(v)), [n.replace(" (", "\n(") for n in v.index], fontsize=8)
for i, x in enumerate(v.values):
    axs[1].text(i, x + 0.2, f"{x:.1f}", ha="center", fontsize=8)
axs[1].set(title="Test 2023: nRMSE per bron", ylabel="nRMSE (%)")
axs[1].grid(axis="x", visible=False)
save(fig, OUT / "01_satelliet_vs_era5.png")
print("klaar:", OUT)

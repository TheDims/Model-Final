# -*- coding: utf-8 -*-
"""
TAAK 6  -  Faiman-fit: paneeltemperatuur afstemmen en vergelijken met de standaardwaarden (BESLISSINGEN #44)
==========================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Faiman: T_cel = T_lucht + straling / (u0 + u1 x wind). Standaard (PARAMS, aanname): u0 = 25, u1 = 6,84.
  1. Kaggle (direct): u0, u1 fitten op de gemeten moduletemperatuur (dag 1-24), wind = ERA5 (geen windmeter);
     test dag 25-34: RMSE temperatuur; daarna het vermogensmodel (LR EFF_T, e6) met standaard vs gefit.
  2. HKUST (indirect, geen moduletemperatuur): rooster u0/u1, kiezen op LR EFF_T (train 2021 -> val 2022),
     met de GEMETEN wind van het weerstation en met ERA5-wind; test 2023.
Regel: gefit "beter" als de nRMSE >= 0,3 pp lager is dan standaard. Runnen: F5 in Spyder of VS Code (Anaconda).
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import ROOT as PROJ, HKUST_RESULTS, MIN_COSZ, BLUE, ORANGE, GREY  # noqa: E402
from solarcore.physics import PARAMS, refaiman   # noqa: E402
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.split import day_split   # noqa: E402
from solarcore.models import LADDER_FASE3   # noqa: E402
from solarcore.evaluate import metrics   # noqa: E402
from solarcore.hkust import load_clean, load_era5_hk, add_era5, capacity as hk_capacity   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJ / "resultaten" / "taak6"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
U0, U1 = PARAMS["u0"], PARAMS["u1"]
WH = PARAMS["wind_height_factor"]
TOL_PP = 0.3
MODELS = {"LR EFF_T": LADDER_FASE3["LR EFF_T (POA + reflectie, spectrum, stof, temp.)"],
          "e6": LADDER_FASE3["e6: EFF_T x (a + b KT_POA), zonder intercept"]}


def faiman_t(g, ta, ws, u0, u1):
    return ta + g / (u0 + u1 * ws * WH)


# %% 1a. Kaggle: fit op de gemeten moduletemperatuur (sensor-straling, ERA5-wind)
kd = weather_frame(15)
sp = day_split(kd.DATE_TIME)
TRAIN, TEST = sp["train_dagnrs"], sp["test_dagnrs"]
m = kd[(kd.IRRADIATION * 1000 > 50) & kd.MODULE_TEMPERATURE.notna() & kd.AMBIENT_TEMPERATURE.notna() & kd.WS10.notna()]
G, TA, TM, WS = m.IRRADIATION * 1000, m.AMBIENT_TEMPERATURE, m.MODULE_TEMPERATURE, m.WS10
tr, te = m.DAY.isin(TRAIN).values, m.DAY.isin(TEST).values
fit_w = least_squares(lambda u: faiman_t(G[tr], TA[tr], WS[tr], *u) - TM[tr], [U0, U1], bounds=([1, 0], [100, 50])).x
fit_0 = least_squares(lambda u: faiman_t(G[tr], TA[tr], WS[tr], u[0], 0) - TM[tr], [U0], bounds=([1], [100])).x
VARIANTS = {"standaard (u0 25, u1 6,84)": (U0, U1), f"gefit met ERA5-wind": tuple(fit_w),
            "gefit zonder wind (u1 = 0)": (fit_0[0], 0.0)}
rows = []
for name, (u0, u1) in VARIANTS.items():
    e = faiman_t(G[te], TA[te], WS[te], u0, u1) - TM[te]
    rows.append({"variant": name, "u0": u0, "u1": u1, "RMSE_°C": float(np.sqrt((e ** 2).mean())),
                 "bias_°C": float(e.mean())})
k_temp = pd.DataFrame(rows).set_index("variant")
print(f"=== Kaggle: moduletemperatuur, test dag {TEST[0]}-{TEST[-1]} (fit op dag {TRAIN[0]}-{TRAIN[-1]}) ===")
print(k_temp.round(3).to_string())

# %% 1b. Kaggle: vermogensmodel met de ERA5-keten (standaard vs gefit)
cap = capacity(kd[kd.DAY.isin(TRAIN)])
kd = add_target(kd, cap)
ok = kd.OK & ~kd.DIP
rows = []
for name, (u0, u1) in VARIANTS.items():
    t_cell, eff_t, eff_t_kt = refaiman(kd, kd.WS10, u0, u1)
    x = kd.assign(EFF_T=eff_t, EFF_T_KT=eff_t_kt)
    trn, tst = x[ok & x.DAY.isin(TRAIN)], x[ok & x.DAY.isin(TEST)]
    for mname, (cols, make, _) in MODELS.items():
        p = np.clip(make().fit(trn[cols], trn.FRAC).predict(tst[cols]), 0, None) * (tst.COSZ.values > MIN_COSZ)
        rows.append({"model": mname, "variant": name, **metrics(tst.FRAC, p, tst)})
k_pow = pd.DataFrame(rows).pivot(index="variant", columns="model", values="nRMSE_%").loc[list(VARIANTS)]
print("\nKaggle: vermogen (ERA5-keten), nRMSE % test:"); print(k_pow.round(3).to_string())

# %% 2. HKUST: indirect, rooster op de vermogensfout; gemeten wind vs ERA5-wind
step3 = json.loads((HKUST_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
h, elev = load_era5_hk()
hd = add_era5(load_clean(), h, elev, step3["tijdverschuiving_min"])
hd["T"] = hd["T"].astype(float)
WIND = {"gemeten wind (weerstation)": hd.WIND.interpolate(limit=8), "ERA5-wind": hd.WS10}
ok_h = hd.OK & WIND["gemeten wind (weerstation)"].notna()


def hk_nrmse(u0, u1, wind, tr_mask, te_mask, model="LR EFF_T"):
    """LR EFF_T (of e6) met deze Faiman-waarden en windbron: train op tr_mask, nRMSE op te_mask."""
    _, eff_t, eff_t_kt = refaiman(hd, wind, u0, u1)
    cap = hk_capacity(hd[tr_mask & ok_h])
    x = hd.assign(EFF_T=eff_t, EFF_T_KT=eff_t_kt, FRAC=hd.P_KW / hd.PLANT.map(cap))
    trn, tst = x[tr_mask & ok_h], x[te_mask & ok_h]
    cols, make, _ = MODELS[model]
    p = np.clip(make().fit(trn[cols], trn.FRAC).predict(tst[cols]), 0, None)
    return float(np.sqrt(((p - tst.FRAC) ** 2).mean()) * 100)


yr = hd.DATE_TIME.dt.year
GRID_U0, GRID_U1 = range(10, 45, 5), range(0, 14, 2)
grid, best = [], {}
for wname, wind in WIND.items():
    for u0 in GRID_U0:
        for u1 in GRID_U1:
            grid.append({"wind": wname, "u0": u0, "u1": u1, "nRMSE_val_%": hk_nrmse(u0, u1, wind, yr == 2021, yr == 2022)})
    g = pd.DataFrame(grid)
    g = g[g.wind == wname]
    best[wname] = tuple(g.loc[g["nRMSE_val_%"].idxmin(), ["u0", "u1"]])
grid = pd.DataFrame(grid)
print("\n=== HKUST: beste u0/u1 op de validatie (train 2021 -> val 2022, LR EFF_T) ===")
for wname, (u0, u1) in best.items():
    v = grid[(grid.wind == wname)]
    print(f"  {wname}: u0 {u0:.0f}, u1 {u1:.0f} -> {v['nRMSE_val_%'].min():.3f} % "
          f"(spreiding over het rooster {v['nRMSE_val_%'].min():.3f}-{v['nRMSE_val_%'].max():.3f} %)")
rows = []
TR_H, TE_H = yr.isin([2021, 2022]), yr == 2023
for wname, wind in WIND.items():
    for label, (u0, u1) in (("standaard", (U0, U1)), ("gefit", best[wname])):
        rows += [{"wind": wname, "Faiman": label, "u0": u0, "u1": u1, "model": mname,
                  "nRMSE_test_2023_%": hk_nrmse(u0, u1, wind, TR_H, TE_H, mname)} for mname in MODELS]
h_test = pd.DataFrame(rows)
h_piv = h_test.pivot_table(index=["wind", "Faiman"], columns="model", values="nRMSE_test_2023_%")
print("\nHKUST test 2023 (train 2021 + 2022), nRMSE %:"); print(h_piv.round(3).to_string())

# %% 3. Regel en opslaan
verdict = {"Kaggle": {mn: bool(k_pow.loc["gefit met ERA5-wind", mn] <= k_pow.iloc[0][mn] - TOL_PP) for mn in MODELS}}
for wname in WIND:
    verdict[f"HKUST {wname}"] = {mn: bool(h_piv.loc[(wname, "gefit"), mn] <= h_piv.loc[(wname, "standaard"), mn] - TOL_PP)
                                 for mn in MODELS}
print(f"\nRegel (gefit >= {TOL_PP} pp beter dan standaard):", verdict)
k_temp.round(4).to_csv(OUT / "kaggle_moduletemperatuur.csv"); k_pow.round(4).to_csv(OUT / "kaggle_vermogen.csv")
grid.round(4).to_csv(OUT / "hkust_rooster_validatie.csv", index=False); h_piv.round(4).to_csv(OUT / "hkust_test_2023.csv")
(OUT / "samenvatting.json").write_text(json.dumps({
    "taak": "6 - Faiman-fit (BESLISSINGEN #44)", "standaard": {"u0": U0, "u1": U1, "wind_hoogtefactor": WH},
    "kaggle_fit": {k: list(map(float, v)) for k, v in VARIANTS.items()},
    "kaggle_temperatuur": k_temp.round(4).to_dict("index"), "kaggle_vermogen": k_pow.round(4).to_dict("index"),
    "hkust_beste": {k: list(map(float, v)) for k, v in best.items()},
    "hkust_test_2023": {f"{a} | {b}": v for (a, b), v in h_piv.round(4).to_dict("index").items()},
    "regel_gehaald": verdict}, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 4. Grafieken
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
tst = m[te]
for name, col in zip(VARIANTS, (GREY, BLUE, ORANGE)):
    u0, u1 = VARIANTS[name]
    axs[0].scatter(tst.MODULE_TEMPERATURE, faiman_t(tst.IRRADIATION * 1000, tst.AMBIENT_TEMPERATURE, tst.WS10, u0, u1),
                   s=3, alpha=0.25, color=col, lw=0, label=f"{name}: RMSE {k_temp.loc[name, 'RMSE_°C']:.1f} °C")
lim = [tst.MODULE_TEMPERATURE.min(), tst.MODULE_TEMPERATURE.max()]
axs[0].plot(lim, lim, color="k", lw=1)
axs[0].set(title="Kaggle: moduletemperatuur, test dag 25-34", xlabel="gemeten (°C)", ylabel="Faiman (°C)")
axs[0].legend(fontsize=7, markerscale=4)
for wname, col in zip(WIND, (BLUE, ORANGE)):
    g = grid[(grid.wind == wname) & (grid.u1 == best[wname][1])]
    axs[1].plot(g.u0, g["nRMSE_val_%"], marker="o", color=col, label=f"{wname} (u1 = {best[wname][1]:.0f})")
axs[1].axvline(U0, color=GREY, ls="--", lw=1, label="standaard u0 = 25")
axs[1].set(title="HKUST: vermogensfout vs u0 (validatie 2022)", xlabel="u0 (W/m²K)", ylabel="nRMSE LR EFF_T (%)")
axs[1].legend(fontsize=7)
save(fig, OUT / "01_faiman.png")
print("klaar:", OUT)

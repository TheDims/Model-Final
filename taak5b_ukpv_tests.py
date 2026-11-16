# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 3  -  UK PV: echte dag-vooruit-fout (TAAK 4b) en horizon dag 1-15 (TAAK 7)  (BESLISSINGEN #38)
============================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  0. Tijdverschuiving straling (ERA5, alleen jan-jun 2024) en featuretabel per halfuur: opwek + weer uit 3 bronnen
     (_E = ERA5 achteraf, _F1 / _F2 = ECMWF-voorspelling van 1 / 2 dagen eerder), pvlib op het echte dak.
  1. Keuze op 2 vouwen binnen de training (6 mrt - 30 jun 2024), regel #17. Input = voorspelling dag 1.
  ---- STOP: test 1 jul 2024 - 31 mrt 2025 pas na akkoord (config.UKPV_TEST_VRIJGEGEVEN) ----
  2. Test: bevroren + walk-forward per maand; dag 2; bovengrens ERA5; per maand/seizoen/systeem; bootstrap.
  3. Horizon dag 1-15 (Single Runs) met het gekozen bevroren model.
Klimaat VK != India: dit test het VOORSPELLEN, niet de locatie. Runnen: F5 in Spyder of VS Code (Anaconda).
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (UKPV_CLEAN, UKPV_RESULTS, UKPV_PERIOD, UKPV_FC_START,  # noqa: E402
                              UKPV_TRAIN_END, UKPV_FOLDS, UKPV_TEST_VRIJGEGEVEN, CAPACITY_QUANTILE,
                              MIN_COSZ, BLUE, ORANGE, GREY, INK2)
from solarcore.ukpv import (chosen_systems, era5_system, model_frame, horizon_features, horizon_frame,  # noqa: E402
                            pv_30min)
from solarcore.physics import hourly_to_15min   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.models import _lr, _ridge, _gb   # noqa: E402
from solarcore.evaluate import metrics, choose, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = UKPV_RESULTS / "stap3_tests"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 230)
style()
SHIFTS = [-60, -45, -30, -15, 0]
START, END = UKPV_PERIOD
chosen = chosen_systems()
sel = pd.read_parquet(UKPV_CLEAN / "ukpv_30min.parquet")


def era5(sid):
    return era5_system(chosen, sid)


# %% 0a. Tijdverschuiving straling: alleen ERA5 in jan-jun 2024 (BESLISSINGEN #38)
rows = []
for sid in chosen.index:
    h, elev = era5(sid)
    p = pv_30min(sel, sid)
    p = p[p.index <= UKPV_TRAIN_END]
    for sh in SHIFTS:
        w = hourly_to_15min(h, chosen.latitude_rounded[sid], chosen.longitude_rounded[sid], elev, rad_shift_min=sh, tz="UTC")
        g = w.set_index("DATE_TIME").GHI_W.resample("30min").mean().reindex(p.index)
        ok = p.notna() & (g > 10)
        rows.append({"ss_id": sid, "verschuiving_min": sh, "r": p[ok].corr(g[ok])})
shift = pd.DataFrame(rows).pivot(index="verschuiving_min", columns="ss_id", values="r")
RAD_SHIFT = int(shift.mean(axis=1).idxmax())
print("Tijdverschuiving (gemiddelde r over 40 systemen, jan-jun 2024):", shift.mean(axis=1).round(4).to_dict(),
      f"-> {RAD_SHIFT:+d} min")
shift.round(4).to_csv(OUT / "tijdverschuiving.csv")

# %% 0b. Featuretabel (gecachet per verschuiving)
data = model_frame(RAD_SHIFT, chosen)
print(f"Bruikbare halve uren overdag: {int(data.OK.sum())} (training vanaf {UKPV_FC_START}: "
      f"{int((data.OK & data.DATE_TIME.between(UKPV_FC_START, UKPV_TRAIN_END + ' 23:59')).sum())})")

# %% 1. Modellen en run()
MEAN_DAY, PERS, SMART, ZS = ("Baseline: gemiddelde dag (K x heldere hemel)", "Baseline: persistence (gisteren)",
                             "Smart persistence (K blijft gelijk)", "Zero-shot pvlib (kWp, dakhoek, richting)")
E6, E6X = ["EFF_T", "EFF_T_KT"], ["EFF_T", "EFF_T_KT", "CLOUD", "T", "WS10"]
LADDER = {  # naam: (inputs, fabriek, eenvoud) - inputs worden aangevuld met het bron-achtervoegsel
    "LR POA": (["POA"], _lr, (0, 1, 1)),
    "e6 (zonder intercept)": (E6, lambda: LinearRegression(fit_intercept=False), (0, 2, 2)),
    "Ridge e6 + bewolking, temp., wind": (E6X, _ridge, (0, 5, 5)),
    "Gradient Boosting (e6 + extra)": (E6X, _gb, (1, 5, 5)),
}
ERA_TRAINED = "LR POA getraind op ERA5"
EENVOUD = {**{n: e for n, (_, _, e) in LADDER.items()}, ERA_TRAINED: (0, 1, 1)}
BASES = [MEAN_DAY, PERS, SMART, ZS]


def between(a, b):
    return data.DATE_TIME.between(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta("23:59:59"))


def run(tr_mask, te_mask, src="F1", train_src=None):
    """Train op tr_mask (inputs uit train_src, standaard = src), voorspel te_mask met inputs uit src.
    Capaciteit, heldere-hemelschaal en gemiddelde K alleen uit de training."""
    train_src = train_src or src
    cap = data[tr_mask & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)
    dd = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    tr, te = dd[tr_mask & dd.OK], dd[te_mask & dd.OK]
    kc = add_clear(dd, tr)
    k_tr = kc.loc[tr.index].groupby("PLANT").K.mean()
    da = dayahead(add_daily(kc)).loc[te.index]
    s = dd.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    pr = {MEAN_DAY: (kc.loc[te.index].FRAC_CS * te.PLANT.map(k_tr)).values,
          PERS: s.reindex(list(zip(te.PLANT, te.DAY - 1, te.SLOT))).fillna(0).values,
          SMART: (da.KD_1.fillna(float(k_tr.mean())) * da.FRAC_CS).values,
          ZS: (te[f"PHYS_{src}"] * te.KWP / te.PLANT.map(cap)).values}
    for name, (cols, make, _) in LADDER.items():
        m = make().fit(tr[[f"{c}_{train_src}" for c in cols]].values, tr.FRAC)
        pr[name] = m.predict(te[[f"{c}_{src}" for c in cols]].values)
    m = _lr().fit(tr[["POA_E"]].values, tr.FRAC)
    pr[ERA_TRAINED] = m.predict(te[[f"POA_{src}"]].values)
    pr = {n: np.clip(np.nan_to_num(v), 0, None) * (te.COSZ.values > MIN_COSZ) for n, v in pr.items()}
    return te, pr


COLS = ["nRMSE_%", "nMAE_%", "RMSE_kW", "MAE_kW", "R2", "dagfout_%"]


def add_skill(tab):
    for ref, col in ((PERS, "skill_pers"), (MEAN_DAY, "skill_gemdag")):
        tab[col] = 1 - tab["nRMSE_%"] / tab.loc[ref, "nRMSE_%"]
    return tab


# %% 2. Keuze op de vouwen (input = voorspelling dag 1)
rows = []
for a, b, c, d in UKPV_FOLDS:
    te, pr = run(between(a, b), between(c, d))
    rows += [{"model": n, "vouw": f"{a[5:]}..{b[5:]} -> {c[5:7]}", **metrics(te.FRAC, y, te)} for n, y in pr.items()]
val = pd.DataFrame(rows)
NAMES = BASES + list(LADDER) + [ERA_TRAINED]
tab = val.groupby("model", sort=False)[COLS].mean()
tab = add_skill(tab.join(val.pivot(index="model", columns="vouw", values="nRMSE_%").add_prefix("nRMSE_% ")).loc[NAMES])
CHOICE, WHY = choose(tab, EENVOUD, [c for c in tab.columns if c.startswith("nRMSE_% ")])
print("\n=== Validatie (6 mrt - 30 jun 2024, 2 vouwen, input = voorspelling dag 1) - hier wordt gekozen ===")
print(tab.round(3).to_string())
print("\nKEUZE:", CHOICE); print("Waarom:", WHY)
val.to_csv(OUT / "validatie_per_vouw.csv", index=False); tab.round(4).to_csv(OUT / "validatie.csv")
summary = {"taak": "5b stap 3 - dag-vooruit en horizon UK PV", "tijdverschuiving_min": RAD_SHIFT,
           "tijdverschuiving_r": shift.mean(axis=1).round(4).to_dict(), "keuze": CHOICE, "motivatie": WHY,
           "vouwen": [list(f) for f in UKPV_FOLDS], "validatie": tab.round(4).to_dict("index")}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

if not UKPV_TEST_VRIJGEGEVEN:
    print("\nSTOP: de keuze staat vast. De testperiode (jul 2024 - mrt 2025) is nog NIET bekeken.")
    sys.exit(0) if __name__ == "__main__" else None

# %% 3. Test 1 jul 2024 - 31 mrt 2025: bevroren (dag 1, dag 2, bovengrens ERA5) + walk-forward per maand
TR, TE = between(UKPV_FC_START, UKPV_TRAIN_END), between("2024-07-01", END)
DAY2, UPPER = f"{CHOICE} met voorspelling dag 2", f"Bovengrens: {CHOICE} met ERA5 (perfect weer)"
te_a, pr_a = run(TR, TE, "F1")
_, pr_2 = run(TR, TE, "F2")
_, pr_e = run(TR, TE, "E")
pr_a[DAY2], pr_a[UPPER] = pr_2[CHOICE], pr_e[CHOICE]
ALL = NAMES + [DAY2, UPPER]
test_a = add_skill(pd.DataFrame({n: metrics(te_a.FRAC, pr_a[n], te_a) for n in ALL}).T)
wt, wp = [], {n: [] for n in NAMES}
for start in pd.date_range("2024-07-01", END, freq="MS"):
    t1, p1 = run(between(UKPV_FC_START, start - pd.Timedelta("1D")), between(start, start + pd.offsets.MonthEnd(0)))
    wt.append(t1)
    for n in NAMES:
        wp[n].append(p1[n])
te_w = pd.concat(wt)
walk = pd.DataFrame({n: metrics(te_w.FRAC, np.concatenate(v), te_w) for n, v in wp.items()}).T
print("\n=== TEST jul 2024 - mrt 2025 (bevroren op 6 mrt - 30 jun 2024; input = voorspelling dag 1) ===")
print(test_a[COLS + ["skill_pers", "skill_gemdag"]].round(3).to_string())
print("\nWalk-forward per maand (nRMSE %):"); print(walk["nRMSE_%"].loc[NAMES].round(2).to_string())
KEY = [MEAN_DAY, SMART, ZS, CHOICE, DAY2, UPPER]


def by(col):
    rows = []
    for g, idx in te_a.groupby(col).indices.items():
        rows += [{"model": n, col: g, **metrics(te_a.FRAC.iloc[idx], pr_a[n][idx], te_a.iloc[idx])} for n in KEY]
    return pd.DataFrame(rows).pivot(index=col, columns="model", values="nRMSE_%")[KEY]


by_month, by_season, by_sys = by("MONTH"), by("SEASON"), by("PLANT")
by_month = by_month.loc[[7, 8, 9, 10, 11, 12, 1, 2, 3]]
print("\nPer maand (nRMSE %):"); print(by_month.round(2).to_string())
print("\nPer seizoen (nRMSE %):"); print(by_season.round(2).to_string())
print("\nPer systeem (nRMSE %, samenvatting):"); print(by_sys.describe().loc[["min", "50%", "max"]].round(2).to_string())
boot_m, boot_d = bootstrap_days(te_a, pr_a, pairs=[(CHOICE, n) for n in (MEAN_DAY, SMART, ZS, DAY2, UPPER)])
print(f"\n=== Bootstrap over {te_a.DAY.nunique()} testdagen (90 %-interval) ===")
print(boot_m.loc[ALL].round(2).to_string()); print(boot_d.round(2).to_string())
for name, df in (("test_bevroren", test_a), ("test_walkforward", walk), ("test_per_maand", by_month),
                 ("test_per_seizoen", by_season), ("test_per_systeem", by_sys), ("bootstrap_modellen", boot_m),
                 ("bootstrap_verschillen", boot_d)):
    df.round(4).to_csv(OUT / f"{name}.csv")
summary["test"] = {"periode": "2024-07-01 .. 2025-03-31", "bevroren": test_a.round(4).to_dict("index"),
                   "walkforward": walk.round(4).to_dict("index"), "per_maand": by_month.round(4).to_dict("index"),
                   "per_seizoen": by_season.round(4).to_dict("index")}

# %% 4. Horizon dag 1-15 (TAAK 7): gekozen model, bevroren, met de Single Runs als input (alleen testperiode)
hzf = horizon_features(RAD_SHIFT, chosen)
cols, make, _ = LADDER[CHOICE] if CHOICE in LADDER else (["POA"], _lr, None)
hz = horizon_frame(data, hzf, TR, cols, make, "E" if CHOICE == ERA_TRAINED else "F1")
hz = hz[hz.DATE_TIME >= "2024-07-01"]                                                   # alleen testperiode
rows = []
for dag, g in hz.groupby("DAG"):
    r = {"dag": dag, "runs": g.RUN.nunique(), "n": len(g)}
    for lab, c in (("model", "P_MODEL"), ("gemiddelde dag", "P_GEMDAG"), ("persistence", "P_PERS"), ("smart persistence", "P_SMART")):
        r[f"nRMSE_% {lab}"] = float(np.sqrt(((g[c] - g.FRAC) ** 2).mean()) * 100)
    rows.append(r)
horizon = pd.DataFrame(rows).set_index("dag")
horizon["skill t.o.v. gemiddelde dag"] = 1 - horizon["nRMSE_% model"] / horizon["nRMSE_% gemiddelde dag"]
print("\n=== Horizon (TAAK 7): gekozen model met ECMWF Single Runs, testperiode ===")
print(horizon.round(3).to_string())
horizon.round(4).to_csv(OUT / "horizon_dag1_15.csv")
summary["horizon"] = horizon.round(4).to_dict("index")
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 5. Grafieken
v = test_a.loc[ALL, "nRMSE_%"][::-1]
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.barh(range(len(v)), v.values, height=0.6,
        color=[BLUE if n == CHOICE else (ORANGE if n in (DAY2, UPPER, ZS) else GREY) for n in v.index])
for i, x in enumerate(v.values):
    ax.text(x + 0.2, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(v)), v.index, fontsize=8)
ax.grid(axis="y", visible=False)
ax.set(xlabel="nRMSE (% van capaciteit) - lager is beter",
       title="UK PV, test jul 2024 - mrt 2025: echte dag-vooruit-voorspelling (40 huizen)")
save(fig, OUT / "01_vergelijking_test.png")

fig, ax = plt.subplots(figsize=(8, 3.8))
x = [f"{m:02d}" for m in by_month.index]
for n, c, ls in zip(KEY, (GREY, INK2, ORANGE, BLUE, BLUE, ORANGE), ("-", "-", "-", "-", "--", ":")):
    ax.plot(x, by_month[n], color=c, ls=ls, marker="o", ms=3, lw=1.6, label=n)
ax.set(title="nRMSE per maand (jul 2024 - mrt 2025)", xlabel="maand", ylabel="nRMSE (%)")
ax.legend(fontsize=7, ncol=2)
save(fig, OUT / "02_nrmse_per_maand.png")

fig, ax1 = plt.subplots(figsize=(8, 4))
for lab, c in (("model", BLUE), ("gemiddelde dag", GREY), ("smart persistence", INK2), ("persistence", ORANGE)):
    ax1.plot(horizon.index, horizon[f"nRMSE_% {lab}"], color=c, marker="o", ms=3, lw=2 if lab == "model" else 1.2,
             label=CHOICE if lab == "model" else lab)
ax1.set(title="Horizon: hoe ver vooruit helpt de weersvoorspelling? (ECMWF, testperiode)",
        xlabel="dag vooruit (dag 1 = 0-24 uur na de run van 00 UTC)", ylabel="nRMSE (%)", xticks=horizon.index)
ax1.legend(fontsize=8)
save(fig, OUT / "03_horizon.png")
print("\nKlaar met TAAK 5b stap 3.")

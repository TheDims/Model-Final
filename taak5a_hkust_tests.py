# -*- coding: utf-8 -*-
"""
TAAK 5a - stap 3  -  HKUST: dezelfde tests als Kaggle op een ander klimaat (BESLISSINGEN #28)
==============================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  0. Data (stap 2) + ERA5 met pvlib (platte daken, tilt 0°). Tijdverschuiving ERA5 alleen uit 2021, met de opwek.
  1. Bovengrens met de eigen weersensor: train 2021, test 2022 (sensor alleen betrouwbaar t/m 14 feb 2023, #26).
  3. Retrain (ERA5): ladder uit fase 3 (#18), kiezen op 2022 met regel #17 (vouwen: 2021 -> 2022 H1;
     2021 + 2022 H1 -> 2022 H2).
  ---- STOP: testjaar 2023 pas na akkoord (config.HKUST_TEST_VRIJGEGEVEN) ----
  2. Zero-shot op 2023: 2a Kaggle-eindmodel (LR POA) en Kaggle-e6; 2b pvlib-natuurkunde met de echte systemen.
  3. Test 2023: A = gekozen model op 2021 + 2022 (hoofdtest), ernaast alleen 2021; B = walk-forward per maand.
  Per station, per maand, per seizoen (droog okt-apr, moesson mei-sep), bootstrap over testdagen, residuen.
Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst taak5a_hkust_data.py.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (HKUST_RESULTS, HKUST_LOCATION, HKUST_STATIONS, HKUST_TZ,  # noqa: E402
                              HKUST_WET_MONTHS, HKUST_TEST_VRIJGEGEVEN, MIN_COSZ, MODELS,
                              BLUE, ORANGE, GREY, INK, INK2)
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.physics import hourly_to_15min, PARAMS   # noqa: E402
from solarcore.hkust import (load_clean, load_era5_hk, add_era5, capacity, mean_day, pvlib_ac_kw)  # noqa: E402
from solarcore.models import LADDER_FASE3, LADDER_FASE1, E6, _ridge   # noqa: E402
from solarcore.evaluate import metrics, choose, bootstrap_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = HKUST_RESULTS / "stap3_tests"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 230)
style()
LAT, LON = HKUST_LOCATION
SHIFTS = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
MEAN_DAY, PERS, SMART = "Baseline: gemiddelde dag", "Baseline: persistence (gisteren)", "Smart persistence (K blijft gelijk)"
BASES = [MEAN_DAY, PERS, SMART]
SENSOR_MODELS = {"Sensor: LR straling": (["GHI_SENSOR"], LADDER_FASE1["LR straling"][1]),
                 "Sensor: Ridge (straling, temp., tijd)": (["GHI_SENSOR", "T_AIR", "HOUR_SIN", "HOUR_COS", "COSZ"], _ridge)}
FOLDS = [("2021 -> 2022 H1", ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-06-30")),
         ("2021 + 2022 H1 -> 2022 H2", ("2021-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"))]


def between(df, a, b):
    """Rijen met DATE_TIME in [a, b] (hele dagen)."""
    return (df.DATE_TIME >= pd.Timestamp(a)) & (df.DATE_TIME < pd.Timestamp(b) + pd.Timedelta("1D"))


# %% 0. Data + ERA5 + tijdverschuiving (alleen 2021)
clean = load_clean()
meta = clean.groupby("PLANT")[["KWP_AC", "KWP_DC"]].first()
h, elev = load_era5_hk()
y2021 = between(clean, "2021-01-01", "2021-12-31") & clean.DAG & clean.P_KW.notna() & ~clean.STORING & ~clean.UITVAL
rows = []
for sh in SHIFTS:
    w = hourly_to_15min(h, LAT, LON, elev, rad_shift_min=sh, tz=HKUST_TZ)
    m = clean[y2021].merge(w[["DATE_TIME", "GHI_W"]], on="DATE_TIME")
    rows += [{"station": st, "verschuiving_min": sh, "r": g.P_KW.corr(g.GHI_W)} for st, g in m.groupby("PLANT")]
shift = pd.DataFrame(rows).pivot(index="verschuiving_min", columns="station", values="r")
RAD_SHIFT = int(shift.mean(axis=1).idxmax())
print("Tijdverschuiving ERA5 (r opwek <-> ERA5-GHI, 2021):")
print(shift.round(3).to_string())
print(f"-> gekozen: {RAD_SHIFT:+d} min")
shift.round(4).to_csv(OUT / "tijdverschuiving_era5.csv")

# Natuurkunde: alle 5 stations zijn plat (tilt 0°, BESLISSINGEN #25) -> één weertabel voor alle stations
data = add_era5(clean, h, elev, RAD_SHIFT)
print(f"Bruikbare kwartieren overdag: {int(data.OK.sum())} (2021: {int(data.OK[data.DATE_TIME.dt.year == 2021].sum())}, "
      f"2022: {int(data.OK[data.DATE_TIME.dt.year == 2022].sum())}, 2023: {int(data.OK[data.DATE_TIME.dt.year == 2023].sum())})")


def raw_coef(model):
    """Coëfficiënten per eenheid (voor de stabiliteit van a en b bij e6)."""
    if hasattr(model, "steps"):
        return model[-1].coef_ / model[0].scale_
    return model.coef_


def run(tr_mask, te_mask, ladder=LADDER_FASE3, extra=None, require=()):
    """Train op tr_mask, voorspel te_mask (OK-rijen). Capaciteit en heldere-hemelschaal alleen uit de training.
    extra: {naam: (inputs, fabriek)} (bv. sensormodellen); require: kolommen die in train én test bekend moeten zijn."""
    ok = data.OK & data[list(require)].notna().all(axis=1) if require else data.OK
    cap = capacity(data[tr_mask & ok])
    dd = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    dd.loc[dd.STORING | dd.UITVAL, "FRAC"] = np.nan
    tr, te = dd[tr_mask & ok], dd[te_mask & ok]
    k = dayahead(add_daily(add_clear(dd, tr))).loc[te.index]
    k_fill = float(add_daily(add_clear(tr, tr)).KD.mean())
    s = dd.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    pr = {MEAN_DAY: mean_day(tr, te),
          PERS: s.reindex(list(zip(te.PLANT, te.DAY - 1, te.SLOT))).fillna(0).values,
          SMART: (k.KD_1.fillna(k_fill) * k.FRAC_CS).values}
    coefs, fitted = {}, {}
    for name, (cols, make, *_) in {**ladder, **(extra or {})}.items():
        m = make().fit(tr[cols], tr.FRAC)
        pr[name], fitted[name] = m.predict(te[cols]), m
        if name.startswith(("e6", "Ridge e6 (")):
            coefs[name] = dict(zip(["a", "b"], raw_coef(m)))
    pr = {n: np.clip(v, 0, None) * (te.COSZ.values > MIN_COSZ) for n, v in pr.items()}
    return te, pr, coefs, cap, fitted


def table(te, pr, by=None):
    """Metrics per model (en optioneel per groep: PLANT, MONTH, SEASON)."""
    if by is None:
        return pd.DataFrame({n: metrics(te.FRAC, y, te) for n, y in pr.items()}).T
    rows = []
    for g, idx in te.groupby(by).indices.items():
        rows += [{"model": n, by: g, **metrics(te.FRAC.iloc[idx], y[idx], te.iloc[idx])} for n, y in pr.items()]
    return pd.DataFrame(rows)


def add_skill(tab):
    for ref, col in ((PERS, "skill_pers"), (MEAN_DAY, "skill_gemdag")):
        tab[col] = 1 - tab["nRMSE_%"] / tab.loc[ref, "nRMSE_%"]
    return tab


COLS = ["nRMSE_%", "nMAE_%", "RMSE_kW", "MAE_kW", "R2", "dagfout_%", "skill_pers", "skill_gemdag"]

# %% 1. Bovengrens met de eigen weersensor: train 2021, test 2022 (BESLISSINGEN #26)
tr1, te1 = between(data, "2021-01-01", "2021-12-31") & data.SENSOR_OK, between(data, "2022-01-01", "2022-12-31")
ref_era5 = {n: LADDER_FASE3[n] for n in ("LR GHI (ERA5)", "LR POA (straling op het paneel)")}
te_s, pr_s, *_ = run(tr1, te1, ladder=ref_era5, extra=SENSOR_MODELS, require=["GHI_SENSOR", "T_AIR"])
sens = add_skill(table(te_s, pr_s))
sens_station = table(te_s, pr_s, "PLANT").pivot(index="model", columns="PLANT", values="nRMSE_%")
print("\n=== (1) Bovengrens eigen sensor: train 2021, test 2022 (zelfde kwartieren voor alle modellen) ===")
print(sens[COLS].round(3).to_string())
print("nRMSE per station:"); print(sens_station.round(2).to_string())
sens.round(4).to_csv(OUT / "test1_sensor_2022.csv"); sens_station.round(4).to_csv(OUT / "test1_sensor_per_station.csv")

# %% 3a. Retrain (ERA5): keuze op 2022 (regel #17), alleen ladder-modellen kunnen gekozen worden
rows, coef_rows = [], []
for fold, (a, b), (c, d) in FOLDS:
    te, pr, coefs, *_ = run(between(data, a, b), between(data, c, d))
    rows += [{"model": n, "vouw": fold, **metrics(te.FRAC, y, te)} for n, y in pr.items()]
    coef_rows += [{"model": n, "vouw": fold, **v} for n, v in coefs.items()]
val = pd.DataFrame(rows)
NAMES = BASES + list(LADDER_FASE3)
tab = val.groupby("model", sort=False)[COLS[:6]].mean()
tab = add_skill(tab.join(val.pivot(index="model", columns="vouw", values="nRMSE_%").add_prefix("nRMSE_% ")).loc[NAMES])
CHOICE, WHY = choose(tab, {n: e for n, (_, _, e) in LADDER_FASE3.items()},
                     [c for c in tab.columns if c.startswith("nRMSE_% 2021")])
coef_tab = pd.DataFrame(coef_rows)
print("\n=== (3) Validatie op 2022 (gemiddelde van 2 vouwen) - hier wordt gekozen ===")
print(tab.round(3).to_string())
print("\nCoëfficiënten e6 per vouw:"); print(coef_tab.round(6).to_string(index=False))
print("\nKEUZE:", CHOICE); print("Waarom:", WHY)
val.to_csv(OUT / "validatie_per_vouw.csv", index=False); tab.round(4).to_csv(OUT / "validatie.csv")
coef_tab.to_csv(OUT / "coefficienten_per_vouw.csv", index=False)
summary = {"taak": "5a stap 3 - tests HKUST", "stations": list(HKUST_STATIONS), "tijdverschuiving_min": RAD_SHIFT,
           "tijdverschuiving_r": shift.round(4).to_dict(),
           "test1_sensor": {"train": "2021 (sensor OK)", "test": "2022", "tabel": sens.round(4).to_dict("index")},
           "keuze": CHOICE, "inputs": LADDER_FASE3[CHOICE][0], "motivatie": WHY,
           "vouwen": [f[0] for f in FOLDS], "validatie": tab.round(4).to_dict("index"), "coefficienten": coef_rows}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(shift.index, shift.mean(axis=1), marker="o", color=BLUE, lw=2, label="gemiddelde 5 stations")
for st in shift.columns:
    ax.plot(shift.index, shift[st], color=GREY, lw=0.8, alpha=0.6)
ax.axvline(RAD_SHIFT, color=INK2, ls="--", lw=1)
ax.set(title="Tijdverschuiving ERA5 (alleen 2021)", xlabel="verschuiving ERA5-straling (minuten)",
       ylabel="correlatie met de opwek (r)")
ax.legend()
save(fig, OUT / "01_tijdverschuiving.png")

if not HKUST_TEST_VRIJGEGEVEN:
    print("\nSTOP: de keuze staat vast. 2023 is nog NIET bekeken. Na akkoord: HKUST_TEST_VRIJGEGEVEN = True in config.")
    sys.exit(0) if __name__ == "__main__" else None

# %% 2. Zero-shot op 2023: niets geleerd van Hong Kong
TR, TE = between(data, "2021-01-01", "2022-12-31"), between(data, "2023-01-01", "2023-12-31")
ZS_LR, ZS_E6, ZS_PHYS = "Zero-shot 2a: Kaggle-eindmodel (LR POA)", "Zero-shot 2a: Kaggle-e6", "Zero-shot 2b: pvlib-natuurkunde"
kaggle = joblib.load(MODELS / "model_final_split70.joblib")


def kaggle_e6():
    """e6 op Kaggle-dag 1-34 (zoals het eindmodel; FRAC met de Kaggle-capaciteit)."""
    from solarcore.dataset import weather_frame
    from solarcore.features import capacity as kcap, add_target
    kd = weather_frame(kaggle["rad_shift_min"])
    kd = add_target(kd, kcap(kd[kd.OK]))
    kd = kd[kd.OK & ~kd.DIP]
    return LinearRegression(fit_intercept=False).fit(kd[E6], kd.FRAC)


def zero_shot(te, cap):
    """Voorspellingen in FRAC van Hong Kong (kW / capaciteit uit de training, alleen om te normaliseren)."""
    kwp_ac = te.PLANT.map(meta.KWP_AC).values
    kwp_dc = te.PLANT.map(meta.KWP_DC).values
    c = te.PLANT.map(cap).values
    lr = kaggle["model"].predict(te[kaggle["features"]]) * kwp_ac
    e6 = kaggle_e6().predict(te[E6]) * kwp_ac
    pac = pvlib_ac_kw(te, kwp_dc, kwp_ac, PARAMS)
    day = te.COSZ.values > MIN_COSZ
    return {n: np.clip(np.nan_to_num(np.asarray(v, float)) / c, 0, None) * day
            for n, v in ((ZS_LR, lr), (ZS_E6, e6), (ZS_PHYS, pac))}


# %% 3b. Test 2023: A (2021 + 2022, hoofdtest), A' (alleen 2021), B (walk-forward per maand)
te_a, pr_a, coef_a, cap_a, fit_a = run(TR, TE)
pr_a.update(zero_shot(te_a, cap_a))
ZS = [ZS_LR, ZS_E6, ZS_PHYS]
ALL = BASES + list(LADDER_FASE3) + ZS
test_a = add_skill(table(te_a, pr_a).loc[ALL])
te_a1, pr_a1, *_ = run(between(data, "2021-01-01", "2021-12-31"), TE)
test_a1 = add_skill(table(te_a1, pr_a1))
wt, wp = [], {n: [] for n in BASES + list(LADDER_FASE3)}
for mth in range(1, 13):
    start = pd.Timestamp(f"2023-{mth:02d}-01")
    t1, p1, *_ = run(data.DATE_TIME < start, between(data, start, start + pd.offsets.MonthEnd(0)))
    wt.append(t1)
    for n in wp:
        wp[n].append(p1[n])
te_w = pd.concat(wt)
walk = add_skill(pd.DataFrame({n: metrics(te_w.FRAC, np.concatenate(v), te_w) for n, v in wp.items()}).T)
print("\n=== TEST 2023 - A: getraind op 2021 + 2022 (hoofdtest), met zero-shot ===")
print(test_a[COLS].round(3).to_string())
print("\nA' (alleen 2021) en B (walk-forward per maand), nRMSE %:")
print(pd.DataFrame({"A (2021+2022)": test_a["nRMSE_%"], "A' (2021)": test_a1["nRMSE_%"],
                    "B (walk-forward)": walk["nRMSE_%"]}).loc[ALL].round(2).to_string())
KEY = [MEAN_DAY, SMART, CHOICE] + ZS
by_station = table(te_a, pr_a, "PLANT").pivot(index="model", columns="PLANT", values="nRMSE_%").loc[ALL]
by_month = table(te_a, {n: pr_a[n] for n in KEY}, "MONTH").pivot(index="MONTH", columns="model", values="nRMSE_%")[KEY]
by_season = table(te_a, pr_a, "SEASON").pivot(index="model", columns="SEASON", values="nRMSE_%").loc[ALL]
print("\nPer station (nRMSE %):"); print(by_station.round(2).to_string())
print("\nPer seizoen (nRMSE %):"); print(by_season.round(2).to_string())
print("\nPer maand (nRMSE %):"); print(by_month.round(2).to_string())
pairs = [(CHOICE, n) for n in [MEAN_DAY, SMART, "LR GHI (ERA5)"] + ZS if n != CHOICE]
boot_m, boot_d = bootstrap_days(te_a, pr_a, pairs=pairs)
print(f"\n=== Bootstrap over {te_a.DAY.nunique()} testdagen (90 %-interval) ===")
print(boot_m.loc[ALL].round(2).to_string()); print(boot_d.round(2).to_string())
# Diagnose (achteraf toegevoegd, na het zien van 2023): is de fout een niveau- of een vormprobleem?
# voorspeld/werkelijk per station en de nRMSE na de beste schaalfactor (= alleen de vorm). Geen nieuw model.
rows = []
for st, idx in te_a.groupby("PLANT").indices.items():
    y = te_a.FRAC.values[idx]
    for n in [CHOICE] + ZS:
        p = pr_a[n][idx]
        s = (p * y).sum() / (p * p).sum()
        rows.append({"station": st, "model": n, "DC_AC": round(meta.loc[st, "KWP_DC"] / meta.loc[st, "KWP_AC"], 2),
                     "capaciteit_div_kWpAC": cap_a[st] / meta.loc[st, "KWP_AC"], "voorspeld_div_werkelijk": p.sum() / y.sum(),
                     "nRMSE_%": np.sqrt(((p - y) ** 2).mean()) * 100,
                     "nRMSE_na_schaal_%": np.sqrt(((s * p - y) ** 2).mean()) * 100})
level = pd.DataFrame(rows)
print("\nDiagnose niveau vs vorm (achteraf):"); print(level.round(3).to_string(index=False))
ok = data[data.OK]
pr_year = (ok.groupby(["PLANT", ok.DATE_TIME.dt.year, "SEASON"]).P_KW.sum()
           / ok.groupby(["PLANT", ok.DATE_TIME.dt.year, "SEASON"]).GHI_W.sum()).unstack("PLANT")
pr_year = pr_year / pd.Series(meta.KWP_AC) * 1000          # kWh per kW AC per kWh/m² ERA5-GHI
print("\nOpbrengst per kW AC per eenheid ERA5-straling, per jaar en seizoen:"); print(pr_year.round(3).to_string())
level.round(4).to_csv(OUT / "diagnose_niveau_vs_vorm.csv", index=False)
pr_year.round(4).to_csv(OUT / "diagnose_opbrengst_per_era5_per_jaar.csv")
summary["diagnose_achteraf"] = {"niveau_vs_vorm": level.round(4).to_dict("records"),
                                "opbrengst_per_era5": {str(k): v for k, v in pr_year.round(4).to_dict("index").items()}}

for name, df in (("test_2023_A", test_a), ("test_2023_A_alleen2021", test_a1), ("test_2023_walkforward", walk),
                 ("test_2023_per_station", by_station), ("test_2023_per_maand", by_month),
                 ("test_2023_per_seizoen", by_season), ("bootstrap_modellen", boot_m), ("bootstrap_verschillen", boot_d)):
    df.round(4).to_csv(OUT / f"{name}.csv")
summary.update({"test_2023": {"A": test_a.round(4).to_dict("index"), "A_alleen_2021": test_a1.round(4).to_dict("index"),
                              "B_walkforward": walk.round(4).to_dict("index"),
                              "per_station": by_station.round(4).to_dict("index"),
                              "per_seizoen": by_season.round(4).to_dict("index"),
                              "per_maand": by_month.round(4).to_dict("index"), "coef_A": coef_a,
                              "capaciteit_kW": cap_a.round(3).to_dict()}})
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 4. Grafieken: vergelijking, per maand, residuen
v = test_a.loc[ALL, "nRMSE_%"][::-1]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
cols = [BLUE if n == CHOICE else (ORANGE if n.startswith("Zero") else GREY) for n in v.index]
ax.barh(range(len(v)), v.values, color=cols, height=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.2, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(v)), v.index, fontsize=8)
ax.grid(axis="y", visible=False)
ax.set(xlabel="nRMSE (% van capaciteit) - lager is beter",
       title="HKUST test 2023: retrain (blauw = gekozen), zero-shot (oranje) en baselines")
save(fig, OUT / "02_vergelijking_2023.png")

fig, ax = plt.subplots(figsize=(8, 3.8))
for n, c, ls in zip(KEY, (GREY, INK2, BLUE, ORANGE, ORANGE, ORANGE), ("-", "-", "-", "-", "--", ":")):
    ax.plot(by_month.index, by_month[n], color=c, ls=ls, marker="o", ms=3, lw=1.6, label=n)
for m in HKUST_WET_MONTHS:
    ax.axvspan(m - 0.5, m + 0.5, color="#eef3fb", zorder=0)
ax.set(title="nRMSE per maand in 2023 (lichtblauw = moesson)", xlabel="maand", ylabel="nRMSE (%)", xticks=range(1, 13))
ax.legend(fontsize=7, ncol=2)
save(fig, OUT / "03_nrmse_per_maand.png")

from statsmodels.graphics.tsaplots import plot_acf   # noqa: E402
res = (pr_a[CHOICE] - te_a.FRAC.values) * 100
hour = te_a.DATE_TIME.dt.hour + te_a.DATE_TIME.dt.minute / 60
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
axs[0, 0].scatter(hour, res, s=2, alpha=0.15, color=BLUE, lw=0)
axs[0, 1].scatter(pr_a[CHOICE] * 100, res, s=2, alpha=0.15, color=BLUE, lw=0)
axs[0, 2].scatter(te_a.KT_W, res, s=2, alpha=0.15, color=BLUE, lw=0)
for ax in axs[0]:
    ax.axhline(0, color=INK2, lw=1)
axs[1, 0].hist(res, bins=80, color=BLUE)
axs[0, 0].set(title="Residu tegen tijd van de dag", xlabel="uur (HKT)", ylabel="residu (% van capaciteit)")
axs[0, 1].set(title="Residu tegen voorspelde waarde", xlabel="voorspeld (% van capaciteit)", ylabel="residu (%)")
axs[0, 2].set(title="Residu tegen helderheid ERA5 (KT)", xlabel="KT ERA5 (-)", ylabel="residu (%)")
axs[1, 0].set(title="Verdeling van de residuen", xlabel="residu (% van capaciteit)", ylabel="aantal kwartieren")
season_res = pd.Series(res).groupby(te_a.SEASON.values)
axs[1, 1].boxplot([g.values for _, g in season_res], tick_labels=list(season_res.groups), showfliers=False)
axs[1, 1].set(title="Residu per seizoen", ylabel="residu (%)")
one = (te_a.PLANT == HKUST_STATIONS[3]).values
plot_acf(res[one], lags=40, ax=axs[1, 2], color=BLUE, vlines_kwargs={"colors": BLUE})
axs[1, 2].set(title=f"Autocorrelatie residu ({HKUST_STATIONS[3]})", xlabel="lag (kwartieren)", ylabel="ACF (-)")
fig.suptitle(f"Residuen van '{CHOICE}' (HKUST, test 2023, model op 2021 + 2022): voorspeld - werkelijk",
             x=0.01, ha="left", color=INK)
save(fig, OUT / "04_residuen.png")
print("\nKlaar met TAAK 5a stap 3.")

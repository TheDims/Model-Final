# -*- coding: utf-8 -*-
"""
TAAK 5b - stap 6  -  UK PV: band P10-P90 (FR9) met ECHTE voorspellingen  (BESLISSINGEN #51)
==========================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Zelfde bandmethode als fase 4 (gemeten / voorspeld per bewolkingsklasse), nu met de ECMWF-voorspelling:
  1. dag 1: kalibreren op out-of-sample-voorspellingen binnen de training (vouwen mei, juni), klasse = KT_W_F1;
     dekking op de test (jul 2024 - mrt 2025), per seizoen en per klasse;
  2. horizon: dekking van de dag-1-band per dag vooruit, en een band per dag vooruit (gekalibreerd op de
     horizonrijen t/m 30 jun 2024; model daar in-sample -> iets te smal). Klasse = KT_POA (helderheid op het paneel).
Regel: band "goed" bij een dekking van 75-85 % (doel 80 %). Runnen: F5 in Spyder of VS Code (Anaconda).
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
from solarcore.config import (UKPV_RESULTS, UKPV_PERIOD, UKPV_FC_START, UKPV_TRAIN_END, UKPV_FOLDS,  # noqa: E402
                              CAPACITY_QUANTILE, MIN_COSZ, BLUE, ORANGE, GREY)
from solarcore.ukpv import chosen_systems, model_frame, horizon_features, horizon_frame   # noqa: E402
from solarcore.models import _lr   # noqa: E402
from solarcore.evaluate import fit_band, apply_band, coverage   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = UKPV_RESULTS / "stap6_band"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
GOOD = (75.0, 85.0)
step3 = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
SHIFT = step3["tijdverschuiving_min"]
chosen = chosen_systems()
data = model_frame(SHIFT, chosen)
data["KT_POA_F1"] = (data.EFF_T_KT_F1 / data.EFF_T_F1).where(data.EFF_T_F1 > 0)


def between(a, b):
    return data.DATE_TIME.between(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta("23:59:59"))


def predict(tr, te):
    """LR POA (voorspelling dag 1), capaciteit uit de training -> testrijen met FRAC en pred."""
    cap = data[tr & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)
    d = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    trn, tst = d[tr & d.OK], d[te & d.OK].copy()
    tst["pred"] = np.clip(_lr().fit(trn[["POA_F1"]], trn.FRAC).predict(tst[["POA_F1"]]), 0, None) * (tst.COSZ > MIN_COSZ)
    return tst


# %% 1. Dag 1: kalibreren out-of-sample binnen de training, dekking op de test
oos = pd.concat([predict(between(a, b), between(c, d)) for a, b, c, d in UKPV_FOLDS])
BAND = fit_band(oos, kt_col="KT_W_F1")
TR, TE = between(UKPV_FC_START, UKPV_TRAIN_END), between("2024-07-01", UKPV_PERIOD[1])
test = predict(TR, TE)
p10, p90 = apply_band(BAND, test, kt_col="KT_W_F1")
test["p10"], test["p90"] = p10, p90
cov = coverage(test, p10, p90)
klasse = (np.digitize(test.KT_W_F1, [0, 0.45, 0.75, 9]) - 1).astype(int)
per_season = {s: coverage(g, g.p10, g.p90) for s, g in test.groupby("SEASON")}
per_class = {["bewolkt", "half bewolkt", "helder"][k]: coverage(g, g.p10, g.p90) for k, g in test.groupby(klasse)}
width = float(((p90 - p10) * 100).mean())
print("Band (P10, P90 als factor op de voorspelling) per klasse:", {k: tuple(round(x, 2) for x in v) for k, v in BAND.items()})
print(f"\n=== Dag 1, test jul 2024 - mrt 2025: dekking {cov:.1f} % (doel 80 %), gemiddelde breedte {width:.1f} % van de capaciteit ===")
print("per seizoen:", {k: round(v, 1) for k, v in per_season.items()})
print("per klasse:", {k: round(v, 1) for k, v in per_class.items()})

# %% 2. Horizon: dag-1-band per dag vooruit, en een band per dag vooruit
hz = horizon_frame(data, horizon_features(SHIFT, chosen), TR, ["POA"], _lr, "F1")
hz["KT_POA"] = (hz.EFF_T_KT / hz.EFF_T).where(hz.EFF_T > 0)
hz["pred"] = hz.P_MODEL
oos["KT_POA"] = oos.KT_POA_F1
band_d1 = fit_band(oos, kt_col="KT_POA")
h_tr = hz[hz.DATE_TIME <= pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59")]
h_te = hz[hz.DATE_TIME >= "2024-07-01"]
rows, bands = [], {}
for d, g in h_te.groupby("DAG"):
    c1 = coverage(g, *apply_band(band_d1, g, kt_col="KT_POA"))
    gt = h_tr[h_tr.DAG == d]
    if len(gt):
        bands[int(d)] = fit_band(gt, kt_col="KT_POA")
    b = bands.get(int(d), bands.get(max(bands)) if bands else band_d1)
    lo, hi = apply_band(b, g, kt_col="KT_POA")
    rows.append({"dag": d, "dekking dag-1-band %": c1, "dekking band per dag %": coverage(g, lo, hi),
                 "breedte band per dag (% cap.)": float(((hi - lo) * 100).mean()), "band uit training": len(gt) > 0})
horizon = pd.DataFrame(rows).set_index("dag")
print("\n=== Horizon: dekking per dag vooruit (test) ==="); print(horizon.round(1).to_string())
verdict = {"dag 1": GOOD[0] <= cov <= GOOD[1],
           "band per dag (alle dagen)": bool(horizon["dekking band per dag %"].between(*GOOD).all())}
print("Regel #51 (75-85 %):", verdict)
horizon.round(3).to_csv(OUT / "band_horizon.csv")
(OUT / "samenvatting.json").write_text(json.dumps({
    "taak": "5b stap 6 - band met echte voorspellingen (BESLISSINGEN #51)", "band_dag1": BAND,
    "dekking_dag1_%": cov, "breedte_dag1_%": width, "per_seizoen": per_season, "per_klasse": per_class,
    "band_per_dag": bands, "horizon": horizon.round(3).to_dict("index"), "regel": verdict},
    indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 3. Grafieken
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
s = test[test.PLANT == chosen.index[len(chosen) // 2]]
days = sorted(s.DATE_TIME.dt.date.unique())[-5:]
s = s[s.DATE_TIME.dt.date.isin(days)]
x = np.arange(len(s))
axs[0].fill_between(x, s.p10 * 100, s.p90 * 100, color=BLUE, alpha=0.2, label="band P10-P90")
axs[0].plot(x, s.pred * 100, color=BLUE, lw=1.5, label="voorspelling dag 1")
axs[0].plot(x, s.FRAC * 100, color="k", lw=1, label="gemeten")
axs[0].set(title=f"Systeem {s.PLANT.iloc[0]}: laatste 5 testdagen (overdag)", xlabel="halfuur (alleen overdag)",
           ylabel="opwek (% van capaciteit)")
axs[0].legend(fontsize=7)
axs[1].plot(horizon.index, horizon["dekking dag-1-band %"], marker="o", color=GREY, label="dag-1-band")
axs[1].plot(horizon.index, horizon["dekking band per dag %"], marker="o", color=ORANGE, label="band per dag vooruit")
axs[1].axhspan(*GOOD, color=BLUE, alpha=0.1, label="75-85 %")
axs[1].set(title="Dekking van de band per dag vooruit (test)", xlabel="dag vooruit", ylabel="dekking (%)",
           xticks=horizon.index)
axs[1].legend(fontsize=7)
save(fig, OUT / "01_band.png")
print("klaar:", OUT)

# %% 4. Verbetering (BESLISSINGEN #54, ACHTERAF): band elke week op de laatste 4 weken; factor (A) of optellen (B)
PRED_BINS = [-1, 0.05, 0.2, 0.5, 9]
full = predict(TR, between("2024-06-01", UKPV_PERIOD[1]))          # frozen model; juni = in-sample (gemeld)
full["week"] = full.DATE_TIME.dt.to_period("W-SUN")
full["pbin"] = np.digitize(full.pred, PRED_BINS) - 1
weeks = sorted(full.week[full.DATE_TIME >= "2024-07-01"].unique())
parts = []
for w in weeks:
    hist = full[(full.week < w) & (full.week >= w - 4)]
    cur = full[full.week == w].copy()
    ba = fit_band(hist, kt_col="KT_W_F1")
    cur["a10"], cur["a90"] = apply_band(ba, cur, kt_col="KT_W_F1")
    res = (hist.FRAC - hist.pred).groupby(hist.pbin).quantile([0.1, 0.9]).unstack()
    lo = cur.pbin.map(res[0.1]).fillna(-0.1).values
    hi = cur.pbin.map(res[0.9]).fillna(0.1).values
    cur["b10"], cur["b90"] = np.clip(cur.pred + lo, 0, None), cur.pred + hi
    parts.append(cur)
roll = pd.concat(parts)
cov_a, cov_b = coverage(roll, roll.a10, roll.a90), coverage(roll, roll.b10, roll.b90)
season_ab = {s: {"A factor": coverage(g, g.a10, g.a90), "B optellen": coverage(g, g.b10, g.b90)} for s, g in roll.groupby("SEASON")}
width_ab = {"A factor": float(((roll.a90 - roll.a10) * 100).mean()), "B optellen": float(((roll.b90 - roll.b10) * 100).mean())}
print(f"\n=== Verbetering (#54), test jul 2024 - mrt 2025: A factor {cov_a:.1f} % | B optellen {cov_b:.1f} % ===")
print("per seizoen:", {s: {k: round(v, 1) for k, v in d.items()} for s, d in season_ab.items()})
print("gemiddelde breedte (% cap.):", {k: round(v, 1) for k, v in width_ab.items()})
verdict54 = {"A factor": GOOD[0] <= cov_a <= GOOD[1], "B optellen": GOOD[0] <= cov_b <= GOOD[1]}
print("Regel 75-85 %:", verdict54)
s = json.loads((OUT / "samenvatting.json").read_text(encoding="utf-8"))
s["verbetering_54"] = {"dekking": {"A factor": cov_a, "B optellen": cov_b}, "per_seizoen": season_ab,
                       "breedte_%": width_ab, "regel": verdict54}
(OUT / "samenvatting.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

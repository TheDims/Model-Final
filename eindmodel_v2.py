# -*- coding: utf-8 -*-
"""
Eindmodel v2  -  "nieuw huis" (pvlib + doorlopend kalibreren + mengen + band), test en opslaan (BESLISSINGEN #53)
================================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Test op de UK-data (echte ECMWF-voorspellingen), naast LR POA uit stap 3 (die WEL een trainingsperiode nodig heeft):
  1. weekelijkse kalibratie per huis (f, heldere-hemelschaal c, gemiddelde helderheid K) uit de laatste 4 weken;
  2. dag-vooruit (voorspelling van 24 uur eerder) en horizon dag 1-15 (Single Runs);
  3. menggewichten per dag vooruit, geschat op de trainingsperiode (t/m 30 jun 2024);
  4. band variant B (#54), elke week op de laatste 4 weken van de eigen v2-voorspellingen;
  5. opslaan: modellen/model_final_v2.joblib (sleutels zoals model_v5.joblib + v2-onderdelen) en model_info_v2.json.
Runnen: F5 in Spyder of VS Code (Anaconda). Eerst de UK-scripts (cache).
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

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (UKPV_RESULTS, UKPV_FC_START, UKPV_TRAIN_END, MODELS, CAPACITY_QUANTILE,  # noqa: E402
                              MIN_COSZ, BLUE, ORANGE, GREY, INK2)
from solarcore.ukpv import chosen_systems, model_frame, horizon_features, horizon_frame   # noqa: E402
from solarcore.models import _lr   # noqa: E402
from solarcore.physics import PARAMS   # noqa: E402
from solarcore.evaluate import metrics, bootstrap_days, KT_BINS   # noqa: E402
from solarcore import model_v2 as v2   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = UKPV_RESULTS / "eindmodel_v2"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
step3 = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
SHIFT = step3["tijdverschuiving_min"]
chosen = chosen_systems()
END_TR = pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59")

# %% 1. Data + weekelijkse kalibratie per huis (alleen data van vóór de week)
data = model_frame(SHIFT, chosen)
TR = data.DATE_TIME.between(pd.Timestamp(UKPV_FC_START), END_TR)
cap = data[TR & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)       # alleen om te normaliseren (nRMSE)
data["FRAC"] = data.P_KW / data.PLANT.map(cap)
data["PV_KW"] = data.PHYS_F1 * data.KWP                                           # pvlib met de voorspelling van gisteren
data["WEEK"] = data.DATE_TIME.dt.to_period("W-SUN")
hist = data[data.OK][["PLANT", "WEEK", "P_KW", "PV_KW", "CS_GHI", "COSZ"]]
stats = v2.rolling_stats(hist)
d = data[data.OK].merge(stats, on=["PLANT", "WEEK"], how="left")
d["V2_RAW"], d["V2_GEMDAG"] = v2.predict(d.PV_KW, d.CS_GHI, d.f, d.c, d.K, w=1.0)
print(f"kalibratie: {len(stats)} huis-weken; mediaan f = {stats.f[stats.n > 0].median():.3f}")

# %% 2. Dag-vooruit: menggewicht op de training, test jul 2024 - mrt 2025
in_tr = d.DATE_TIME.between(pd.Timestamp(UKPV_FC_START), END_TR) & (d.n > 0)
W_DA = v2.mix_weight(d.V2_RAW[in_tr], d.V2_GEMDAG[in_tr], d.P_KW[in_tr])
d["V2"] = W_DA * d.V2_RAW + (1 - W_DA) * d.V2_GEMDAG
te = d[d.DATE_TIME >= "2024-07-01"].copy()
lr = _lr().fit(data[TR & data.OK][["POA_F1"]], data[TR & data.OK].FRAC)
te["LR_POA"] = np.clip(lr.predict(te[["POA_F1"]]), 0, None)
NAMES = {"LR POA (stap 3, trainingsperiode nodig)": te.LR_POA.values,
         "v2: pvlib x f (4 weken)": te.V2_RAW.values / te.PLANT.map(cap).values,
         f"v2: + mengen met gemiddelde dag (w = {W_DA:.2f})": te.V2.values / te.PLANT.map(cap).values,
         "v2: alleen gemiddelde dag (4 weken)": te.V2_GEMDAG.values / te.PLANT.map(cap).values}
NAMES = {n: np.clip(np.nan_to_num(p), 0, None) * (te.COSZ.values > MIN_COSZ) for n, p in NAMES.items()}
da = pd.DataFrame({n: metrics(te.FRAC, p, te) for n, p in NAMES.items()}).T
V2N = list(NAMES)[2]
boot_m, boot_d = bootstrap_days(te, NAMES, pairs=[(V2N, list(NAMES)[0])])
print("\n=== Dag-vooruit, test jul 2024 - mrt 2025 (40 huizen) ===")
print(da[["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]].round(3).to_string())
print(boot_d.round(2).to_string())

# %% 3. Band variant B op de eigen v2-voorspellingen (fractie van kWp), elke week op de laatste 4 weken
d["P_KWP"], d["V2_KWP"] = d.P_KW / d.KWP, d.V2 / d.KWP
bands, parts = {}, []
for w in sorted(d.WEEK[d.DATE_TIME >= "2024-07-01"].unique()):
    h = d[(d.WEEK < w) & (d.WEEK >= w - v2.WEEKS)]
    cur = d[d.WEEK == w].copy()
    bq = v2.band_quantiles(h.V2_KWP, h.P_KWP)
    cur["lo"], cur["hi"] = v2.apply_band(cur.V2_KWP.values, bq)
    bands[str(w)] = bq
    parts.append(cur)
bt = pd.concat(parts)
cov = float(np.mean((bt.P_KWP >= bt.lo) & (bt.P_KWP <= bt.hi)) * 100)
cov_season = {s: float(np.mean((g.P_KWP >= g.lo) & (g.P_KWP <= g.hi)) * 100) for s, g in bt.groupby("SEASON")}
print(f"\nBand v2 (variant B), dekking test: {cov:.1f} % (doel 80 %); per seizoen:",
      {k: round(v, 1) for k, v in cov_season.items()})

# %% 4. Horizon dag 1-15 met v2 (kalibratie uit de 4 weken vóór de run), gewicht per dag uit de training
hz = horizon_frame(data, horizon_features(SHIFT, chosen), TR, ["POA"], _lr, "F1")
hz["WEEK"] = hz.RUN.dt.to_period("W-SUN")                                        # week van de run: data daarvoor
hz = hz.merge(stats, on=["PLANT", "WEEK"], how="left").merge(data[["PLANT", "DATE_TIME", "CS_GHI", "KWP", "P_KW"]],
                                                              on=["PLANT", "DATE_TIME"], how="left")
hz["RAW"], hz["GEMDAG_V2"] = v2.predict(hz.PHYS * hz.KWP, hz.CS_GHI, hz.f, hz.c, hz.K, w=1.0)
h_tr = hz[(hz.DATE_TIME <= END_TR) & (hz.n > 0)]
W_H = {int(dd): v2.mix_weight(g.RAW, g.GEMDAG_V2, g.P_KW) for dd, g in h_tr.groupby("DAG")}
W_H = pd.Series(W_H).reindex(range(1, 16)).ffill().to_dict()
h_te = hz[hz.DATE_TIME >= "2024-07-01"].copy()
h_te["V2"] = h_te.DAG.map(W_H) * h_te.RAW + (1 - h_te.DAG.map(W_H)) * h_te.GEMDAG_V2
c_h = h_te.PLANT.map(cap)
rows = []
for dd, g in h_te.groupby("DAG"):
    cc = g.PLANT.map(cap)
    rows.append({"dag": dd, "w_d": W_H[int(dd)],
                 "v2 (gemengd)": float(np.sqrt((((g.V2 / cc).clip(0) - g.FRAC) ** 2).mean()) * 100),
                 "LR POA (zonder mengen)": float(np.sqrt(((g.P_MODEL - g.FRAC) ** 2).mean()) * 100),
                 "gemiddelde dag (trainingsperiode)": float(np.sqrt(((g.P_GEMDAG - g.FRAC) ** 2).mean()) * 100)})
horizon = pd.DataFrame(rows).set_index("dag")
mix4 = pd.read_csv(UKPV_RESULTS / "stap4_horizon_mengen" / "horizon_mengen.csv", index_col="dag")
horizon["LR POA gemengd (stap 4)"] = mix4["mengen (w_d uit training)"]
print("\n=== Horizon, test: nRMSE % per dag vooruit ==="); print(horizon.round(2).to_string())

# %% 5. Opslaan: model_final_v2.joblib (sleutels zoals model_v5) + model_info_v2.json
kaggle = joblib.load(MODELS / "model_final_split70.joblib")
bundle = {"versie": "Model Final v2 (nieuw huis: pvlib + doorlopend kalibreren + mengen + band)",
          # sleutels zoals model_v5.joblib (de app kan die blijven lezen)
          "model": kaggle["model"], "features": kaggle["features"], "factor": kaggle["factor"],
          "band_method": "optellen_per_voorspelklasse", "band": bands[max(bands)], "kt_bins": KT_BINS,
          "params": PARAMS, "rad_shift_min": SHIFT, "min_cosz": MIN_COSZ,
          # v2-onderdelen
          "v2": {"start": "pvlib (kWp, dakhoek, richting), physics.panel_inputs -> PHYS (kW per kWp)",
                 "kalibratie": {"methode": "f = som gemeten / som pvlib", "weken": v2.WEEKS, "bijwerken": "elke week"},
                 "gemiddelde_dag": {"methode": "K x c x heldere hemel (GHI)", "weken": v2.WEEKS},
                 "menggewicht_dagvooruit": W_DA, "menggewichten_per_dag": W_H,
                 "band": {"methode": "voorspelling + kwantielen (gemeten - voorspeld), fractie van kWp",
                          "klassen": v2.PRED_BINS, "weken": v2.WEEKS, "voorbeeld_laatste_week": bands[max(bands)]},
                 "code": "solarcore/model_v2.py"}}
joblib.dump(bundle, MODELS / "model_final_v2.joblib")
info = {"model": bundle["versie"], "getest_op": "UK PV, 40 huizen, jul 2024 - mrt 2025, ECMWF-voorspelling 24 uur eerder",
        "dagvooruit": da.round(4).to_dict("index"), "bootstrap_v2_vs_LR": boot_d.round(4).to_dict("index"),
        "band_dekking_%": cov, "band_per_seizoen_%": cov_season, "horizon": horizon.round(4).to_dict("index"),
        "aannames": ["pvlib-parameters (PARAMS) en DC/AC 1,1 zijn aannames", "menggewichten uit de UK-trainingsperiode",
                     "HKUST: doorlopend kalibreren 18,1 % (BESLISSINGEN #43)"]}
(OUT / "model_info_v2.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
da.round(4).to_csv(OUT / "dagvooruit.csv"); horizon.round(4).to_csv(OUT / "horizon.csv")
print("opgeslagen:", MODELS / "model_final_v2.joblib")

# %% 6. Grafiek
fig, ax = plt.subplots(figsize=(8, 4))
for col, c, ls in (("v2 (gemengd)", BLUE, "-"), ("LR POA gemengd (stap 4)", ORANGE, "--"),
                   ("LR POA (zonder mengen)", GREY, ":"), ("gemiddelde dag (trainingsperiode)", INK2, "-")):
    ax.plot(horizon.index, horizon[col], color=c, ls=ls, marker="o", ms=3, lw=1.8, label=col)
ax.set(title="Eindmodel v2 tegen LR POA: nRMSE per dag vooruit (UK, test)", xlabel="dag vooruit",
       ylabel="nRMSE (%)", xticks=horizon.index)
ax.legend(fontsize=8)
save(fig, OUT / "01_v2_horizon.png")
print("klaar:", OUT)

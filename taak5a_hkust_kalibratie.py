# -*- coding: utf-8 -*-
"""
TAAK 5a - stap 4  -  Kalibratietest "nieuw huis": hoeveel weken eigen meetdata zijn nodig? (BESLISSINGEN #31)
==============================================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Een nieuw huis heeft geen historie. Zero-shot zat in stap 3 vooral op het verkeerde NIVEAU (vorm was goed).
Hier: één niveaufactor f = Σ gemeten / Σ voorspeld over de eerste N weken na installatie (N = 0, 1, 2, 4, 8, 13).
  - 30 "nieuwe huizen": 5 stations x 6 installatiedata (1 jan ... 1 jun 2023)
  - evaluatie: altijd weken 13-26 na installatie (zelfde kwartieren voor elke N, geen overlap met de kalibratie)
  - modellen: pvlib x f, Kaggle-eindmodel x f, LR GHI alleen op de eerste N weken
  - referenties: LR GHI op 2021 + 2022 (stap 3, test A), smart persistence, persistence, gemiddelde dag,
    pvlib met 'orakel'-f (gefit op de evaluatieweken zelf = het beste wat één factor kan)
ACHTERAF-test: 2023 is in stap 3 al bekeken. Runnen: F5 in Spyder of VS Code (Anaconda). Eerst stap 3.
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
from solarcore.config import HKUST_RESULTS, HKUST_STATIONS, MIN_COSZ, MODELS, BLUE, ORANGE, GREY, INK2, TIE_PP  # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.physics import PARAMS   # noqa: E402
from solarcore.models import LADDER_FASE3   # noqa: E402
from solarcore.hkust import load_clean, load_era5_hk, add_era5, capacity, mean_day, pvlib_ac_kw   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = HKUST_RESULTS / "stap4_kalibratie"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
style()
WEEKS = [0, 1, 2, 4, 8, 13]                                   # kalibratieperiode na installatie
STARTS = pd.date_range("2023-01-01", "2023-06-01", freq="MS")  # installatiedata
EVAL = (13, 26)                                               # evaluatie: weken 13-26 na installatie
PV, KG, LRN = "pvlib x f", "Kaggle-eindmodel x f", "LR GHI alleen op de eerste N weken"
REFS = ["LR GHI op 2021 + 2022 (stap 3)", "Smart persistence", "Persistence", "Gemiddelde dag (2021-2022)",
        "pvlib x orakel-f"]

# %% 0. Data: zelfde tabel als stap 3 (tijdverschuiving uit stap 3, alleen bepaald op 2021)
step3 = json.loads((HKUST_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))
RAD_SHIFT = step3["tijdverschuiving_min"]
h, elev = load_era5_hk()
data = add_era5(load_clean(), h, elev, RAD_SHIFT)
meta = data.groupby("PLANT")[["KWP_AC", "KWP_DC"]].first()
hist = data.OK & (data.DATE_TIME < "2023-01-01")
cap = capacity(data[hist])                                     # alleen om te normaliseren (nRMSE), zoals stap 3
dd = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
dd.loc[dd.STORING | dd.UITVAL, "FRAC"] = np.nan
day = dd.COSZ.values > MIN_COSZ
kaggle = joblib.load(MODELS / "model_final_split70.joblib")
kwp_ac, kwp_dc, c = (dd.PLANT.map(meta.KWP_AC).values, dd.PLANT.map(meta.KWP_DC).values, dd.PLANT.map(cap).values)
dd["P_PV"] = pvlib_ac_kw(dd, kwp_dc, kwp_ac, PARAMS) / c * day
dd["P_KG"] = np.clip(kaggle["model"].predict(dd[kaggle["features"]]) * kwp_ac / c, 0, None) * day

# referenties met 2 jaar historie (zoals stap 3)
tr = dd[hist]
cols, make, _ = LADDER_FASE3["LR GHI (ERA5)"]
dd["R_LR"] = np.clip(make().fit(tr[cols], tr.FRAC).predict(dd[cols]), 0, None) * day
k = dayahead(add_daily(add_clear(dd, tr)))
dd["R_SMART"] = (k.KD_1.fillna(float(add_daily(add_clear(tr, tr)).KD.mean())) * k.FRAC_CS).values
s = dd.set_index(["PLANT", "DAY", "SLOT"]).FRAC
dd["R_PERS"] = s.reindex(list(zip(dd.PLANT, dd.DAY - 1, dd.SLOT))).fillna(0).values


def nrmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 0, None) - y) ** 2)) * 100)


def factor(obs, pred):
    """Niveaufactor: som gemeten / som voorspeld (1 als er niets te kalibreren is)."""
    return float(obs.sum() / pred.sum()) if pred.sum() > 0 else 1.0


# %% 1. 30 nieuwe huizen
rows = []
for st in HKUST_STATIONS:
    for start in STARTS:
        ev = dd[(dd.PLANT == st) & dd.OK & (dd.DATE_TIME >= start + pd.Timedelta(weeks=EVAL[0]))
                & (dd.DATE_TIME < start + pd.Timedelta(weeks=EVAL[1]))]
        y = ev.FRAC.values
        base = {"station": st, "installatie": start.date(), "n_eval": len(ev)}
        refs = {REFS[0]: ev.R_LR, REFS[1]: ev.R_SMART, REFS[2]: ev.R_PERS,
                REFS[3]: mean_day(tr[tr.PLANT == st], ev), REFS[4]: ev.P_PV * factor(ev.FRAC, ev.P_PV)}
        rows += [{**base, "weken": n, "model": m, "nRMSE_%": nrmse(np.asarray(p), y), "f": np.nan}
                 for m, p in refs.items() for n in WEEKS]
        for n in WEEKS:
            cal = dd[(dd.PLANT == st) & dd.OK & (dd.DATE_TIME >= start) & (dd.DATE_TIME < start + pd.Timedelta(weeks=n))]
            # geen bruikbare data in de kalibratieweken (storing/gaten) -> huis blijft ongekalibreerd (f = 1)
            f_pv, f_kg = (factor(cal.FRAC, cal.P_PV), factor(cal.FRAC, cal.P_KG)) if len(cal) else (1.0, 1.0)
            base_n = {**base, "weken": n, "n_kalibratie": len(cal)}
            rows.append({**base_n, "model": PV, "nRMSE_%": nrmse(ev.P_PV * f_pv, y), "f": f_pv})
            rows.append({**base_n, "model": KG, "nRMSE_%": nrmse(ev.P_KG * f_kg, y), "f": f_kg})
            if len(cal):
                lr = make().fit(cal[cols], cal.FRAC)
                rows.append({**base_n, "model": LRN, "nRMSE_%": nrmse(lr.predict(ev[cols]), y), "f": np.nan})
res = pd.DataFrame(rows)
leeg = res[(res.model == PV) & (res.weken > 0) & (res.n_kalibratie == 0)]
print("Huizen zonder bruikbare data in de kalibratieweken (blijven ongekalibreerd):")
print(leeg[["station", "installatie", "weken"]].to_string(index=False) if len(leeg) else "geen")
res.round(4).to_csv(OUT / "kalibratie_per_huis.csv", index=False)

# %% 2. Samenvatting over de 30 huizen + vooraf vastgelegde regel
agg = res.groupby(["model", "weken"])["nRMSE_%"].agg(gemiddeld="mean", P10=lambda v: v.quantile(0.1),
                                                     P90=lambda v: v.quantile(0.9)).round(2)
tab = agg["gemiddeld"].unstack("weken")
print(f"=== Kalibratietest: gemiddelde nRMSE (%) over {res.groupby(['station', 'installatie']).ngroups} huizen, "
      f"evaluatie weken {EVAL[0]}-{EVAL[1]} na installatie ===")
print(tab.loc[[PV, KG, LRN] + REFS].to_string())
print("\nSpreiding over de huizen (P10-P90):")
print(agg.loc[[PV, KG, LRN]].to_string())
oracle = tab.loc[REFS[4], 0]
enough = {m: next((n for n in WEEKS if tab.loc[m, n] <= oracle + TIE_PP), None) for m in (PV, KG, LRN)}
print(f"\nRegel (#31): genoeg = binnen {TIE_PP} pp van het orakel ({oracle:.2f} %): {enough}")
ferr = res[res.model == PV].pivot_table(index=["station", "installatie"], columns="weken", values="f")
print("\nNiveaufactor f (pvlib) per station, gemiddeld over installatiedata:")
print(ferr.groupby("station").mean().round(3).to_string())
agg.to_csv(OUT / "kalibratie_samenvatting.csv")
ferr.round(4).to_csv(OUT / "kalibratie_factor_pvlib.csv")
summary = {"taak": "5a stap 4 - kalibratietest nieuw huis (ACHTERAF, BESLISSINGEN #31)", "tijdverschuiving_min": RAD_SHIFT,
           "installatiedata": [str(d.date()) for d in STARTS], "evaluatie_weken": EVAL, "kalibratie_weken": WEEKS,
           "gemiddelde_nRMSE": {m: tab.loc[m].round(3).to_dict() for m in tab.index},
           "orakel_nRMSE": round(oracle, 3), "genoeg_weken": enough}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# %% 3. Grafiek
fig, ax = plt.subplots(figsize=(8, 4.2))
for m, col, ls in ((PV, BLUE, "-"), (KG, ORANGE, "-"), (LRN, GREY, "--")):
    a = agg.loc[m]
    ax.plot(a.index, a.gemiddeld, color=col, ls=ls, marker="o", lw=2, label=m)
    if m == PV:
        ax.fill_between(a.index, a.P10, a.P90, color=BLUE, alpha=0.15, label="pvlib x f: P10-P90 over huizen")
for m, ls in ((REFS[0], ":"), (REFS[1], "-."), (REFS[4], "--")):
    ax.axhline(tab.loc[m, 0], color=INK2, ls=ls, lw=1, label=f"{m} ({tab.loc[m, 0]:.1f} %)")
ax.set(title="Nieuw huis: hoeveel weken eigen meetdata zijn nodig? (HKUST 2023, 30 huizen)",
       xlabel="weken eigen meetdata na installatie", ylabel="nRMSE in week 13-26 (%)", xticks=WEEKS)
ax.legend(fontsize=7)
save(fig, OUT / "01_kalibratie_weken.png")
print("klaar:", OUT)

# %% 4. Doorlopend kalibreren (BESLISSINGEN #42, ACHTERAF): elke evaluatieweek f uit de laatste N weken
ROLL = [2, 4, 8, None]
ROLLN = {n: f"pvlib x f, doorlopend (laatste {n} wk)" if n else "pvlib x f, doorlopend (alles sinds installatie)"
         for n in ROLL}
rows2 = []
for st in HKUST_STATIONS:
    own = dd[(dd.PLANT == st) & dd.OK]
    for start in STARTS:
        ys, preds = [], {n: [] for n in ROLL}
        for k in range(EVAL[0], EVAL[1]):
            w0 = start + pd.Timedelta(weeks=k)
            ev = own[(own.DATE_TIME >= w0) & (own.DATE_TIME < w0 + pd.Timedelta(weeks=1))]
            if not len(ev):
                continue
            ys.append(ev.FRAC.values)
            for n in ROLL:
                a = start if n is None else max(start, w0 - pd.Timedelta(weeks=n))
                cal = own[(own.DATE_TIME >= a) & (own.DATE_TIME < w0)]       # alleen data van vóór deze week
                preds[n].append(ev.P_PV.values * (factor(cal.FRAC, cal.P_PV) if len(cal) else 1.0))
        y = np.concatenate(ys)
        rows2 += [{"station": st, "installatie": start.date(), "model": ROLLN[n],
                   "nRMSE_%": nrmse(np.concatenate(preds[n]), y)} for n in ROLL]
roll = pd.DataFrame(rows2)
roll_tab = roll.groupby("model")["nRMSE_%"].agg(gemiddeld="mean", P10=lambda v: v.quantile(0.1),
                                                 P90=lambda v: v.quantile(0.9)).loc[list(ROLLN.values())]
ref = {"vaste f uit 13 weken": tab.loc[PV, 13], "orakel-f": oracle, REFS[0]: tab.loc[REFS[0], 0],
       "pvlib zonder kalibratie": tab.loc[PV, 0]}
print("\n=== Doorlopend kalibreren (ACHTERAF, #42): gemiddelde nRMSE (%) over 30 huizen, week 13-26 ===")
print(roll_tab.round(2).to_string())
print("Referenties:", {k: round(v, 2) for k, v in ref.items()})
works = {m: bool(roll_tab.loc[m, "gemiddeld"] <= ref["vaste f uit 13 weken"] - 0.3) for m in roll_tab.index}
print("Regel (#42) 'doorlopend werkt' (<= vaste f 13 wk - 0,3 pp):", works)
roll.round(4).to_csv(OUT / "doorlopend_per_huis.csv", index=False)
roll_tab.round(3).to_csv(OUT / "doorlopend_samenvatting.csv")
summary["doorlopend_achteraf"] = {"gemiddelde_nRMSE": roll_tab.gemiddeld.round(3).to_dict(),
                                  "referenties": {k: round(float(v), 3) for k, v in ref.items()}, "werkt": works}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

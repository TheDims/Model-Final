# -*- coding: utf-8 -*-
"""
FASE 2  -  Forecasting perfectioneren (alleen Kaggle-data, echt vooruit)
========================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Vraag: wat kunnen we voorspellen met alleen informatie die er op het voorspelmoment t0 al is?
  - horizons: +15 min, +1 uur, +3 uur en day-ahead (morgen per kwartier, gemaakt om 21:00);
  - inputs alleen <= t0: eerdere opwek, eerdere sensorwaarden, klok, zonnestand, heldere-hemelprofiel;
  - we voorspellen de heldere-hemelindex K (1 = helder) en vermenigvuldigen met de opwek bij heldere hemel.
  Deel A (altijd): keuze per horizon op de validatievouwen binnen de trainingsdagen.
  Deel B (pas na akkoord, config.TEST_VRIJGEGEVEN): de testdagen (bevroren + walk-forward), per plant, horizon-grafiek.
Hypothese (nog niet getest): kort vooruit werkt goed; day-ahead zonder weer is nauwelijks beter dan de
gemiddelde dag -> motivatie voor fase 3 (weermodel).

Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase0_data.py.
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
from solarcore.config import (RESULTS, CLEAN_FILE, PLANTS, MIN_COSZ, HOUSE_KWP, BLUE, ORANGE, GREY, INK, INK2,  # noqa: E402
                              RUN_NAME, toon_test)
from solarcore.features import add_time, add_solar, add_clearsky, capacity, add_target   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, intraday, dayahead, K_CLIP   # noqa: E402
from solarcore.models import ladder_fase2, BASELINES_FASE2   # noqa: E402
from solarcore.evaluate import metrics, choose   # noqa: E402
from solarcore.split import day_split, fold_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

TOON_TEST = toon_test(2)   # per run vrijgegeven in config.TEST_VRIJGEGEVEN (na akkoord Dimme, AGENTS.md §6.2)
warnings.filterwarnings("ignore")
OUT = RESULTS / "fase2"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
HORIZONS = {"+15 min": 1, "+1 uur": 4, "+3 uur": 12, "day-ahead (21:00)": None}   # kwartieren vooruit


# %% 1. Data (volledig kwartier-rooster per plant, nodig voor de lags)
base = pd.read_csv(CLEAN_FILE, parse_dates=["DATE_TIME"])
base = add_clearsky(add_solar(add_time(base))).sort_values(["PLANT", "DATE_TIME"]).reset_index(drop=True)
sp = day_split(base.DATE_TIME)
N_SETUP, N_DAYS = sp["n_opzet"], sp["n_dagen"]
VAL = [(range(1, last + 1), tuple(val)) for last, val in fold_days(N_SETUP)]   # BESLISSINGEN #17
TR_TXT, TE_TXT = f"dag 1-{N_SETUP}", f"dag {N_SETUP + 1}-{N_DAYS}"
print(f"Run {RUN_NAME} | training {TR_TXT}, test {TE_TXT}")


def prepare(train_days):
    """Target, heldere hemel en daggemiddelden; capaciteit en schaal ALLEEN uit de trainingsdagen.
    Dips en uitval -> FRAC = NaN (ook niet in de lags of daggemiddelden)."""
    dd = add_target(base, capacity(base[base.DAY.isin(train_days)]))
    dd.loc[dd.DIP, "FRAC"] = np.nan
    dd = add_daily(add_clear(dd, dd[dd.DAY.isin(train_days)]))
    dd["OK"] = (dd.COSZ > MIN_COSZ) & dd.FRAC.notna()        # beoordeeld: overdag, opwek bekend (geen dip)
    return dd, float(dd[dd.DAY.isin(train_days)].KD.mean())


def run(train_days, test_days, hz):
    """Train op train_days, voorspel test_days voor horizon hz. Geeft (testtabel, {model: FRAC-voorspelling})."""
    train_days, test_days = list(train_days), list(test_days)
    dd, k_fill = prepare(train_days)
    f = dayahead(dd) if hz is None else intraday(dd, hz, k_fill)
    tr, te = f[f.DAY.isin(train_days)], f[f.DAY.isin(test_days) & f.OK]
    prof = tr[tr.OK].groupby(["PLANT", "SLOT"]).FRAC.mean()
    k_now = te.KD_1 if hz is None else te.K_L0
    pr = {"Baseline: gemiddelde dag": np.nan_to_num(prof.reindex(list(zip(te.PLANT, te.SLOT))).values),
          "Baseline: persistence": te.FRAC_L0.fillna(0).values,
          "Smart persistence (K blijft gelijk)": (k_now.fillna(k_fill) * te.FRAC_CS).values}
    for name, (cols, make, _) in ladder_fase2(hz is None).items():
        t = tr[tr.K.notna()].dropna(subset=cols)
        k_hat = make().fit(t[cols], t.K).predict(te[cols].fillna(k_fill))
        pr[name] = np.clip(k_hat, 0, K_CLIP) * te.FRAC_CS.values
    return te, {k: np.clip(v, 0, None) for k, v in pr.items()}


# %% 2. Deel A - keuze per horizon op de validatievouwen (trainingsdagen)
rows, choices = [], {}
for hname, hz in HORIZONS.items():
    for tr_days, va_days in VAL:
        te, pr = run(tr_days, va_days, hz)
        fold = f"train 1-{max(tr_days)} -> val {va_days[0]}-{va_days[-1]}"
        rows += [{"horizon": hname, "model": n, "vouw": fold, **metrics(te.FRAC, y, te)} for n, y in pr.items()]
val = pd.DataFrame(rows)
tabs = {}
for hname, hz in HORIZONS.items():
    v = val[val.horizon == hname]
    t = v.groupby("model", sort=False)[["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]].mean()
    t = t.join(v.pivot(index="model", columns="vouw", values="nRMSE_%").add_prefix("nRMSE_% "))
    eenvoud = {**BASELINES_FASE2, **{n: e for n, (_, _, e) in ladder_fase2(hz is None).items()}}
    choices[hname] = choose(t, eenvoud, [c for c in t.columns if c.startswith("nRMSE_% train")])
    tabs[hname] = t
    print(f"\n=== {hname}: validatie binnen {TR_TXT} ===")
    print(t.round(3).to_string())
    print("KEUZE:", choices[hname][0], "|", choices[hname][1])
val.to_csv(OUT / "validatie_per_vouw.csv", index=False)
pd.concat(tabs).round(4).to_csv(OUT / "validatie.csv")

# horizon-grafiek (validatie)
style()
models = list(tabs["+15 min"].index)
COL = {"Baseline: gemiddelde dag": GREY, "Baseline: persistence": "#8a8984",
       "Smart persistence (K blijft gelijk)": ORANGE, "LR op eerdere K": BLUE,
       "Ridge (K + sensor + tijd)": "#184f95", "Random Forest": "#6da7ec", "Gradient Boosting": "#b7d3f6"}
LS = {"Baseline: gemiddelde dag": "--", "Baseline: persistence": ":"}


def horizon_plot(tab_by_h, title, fname, upper=None):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(HORIZONS))
    for m in models:
        ax.plot(x, [tab_by_h[h].loc[m, "nRMSE_%"] for h in HORIZONS], marker="o", ms=7, mec="white", mew=1.5,
                lw=2, color=COL[m], ls=LS.get(m, "-"), label=m)
    if upper is not None:
        ax.axhline(upper, color=INK2, lw=1, ls="-.", label="bovengrens: sensor op hetzelfde moment (fase 1)")
    ax.set_xticks(x, list(HORIZONS)); ax.set_xlabel("hoe ver vooruit"); ax.set_ylabel("nRMSE (% van capaciteit)")
    ax.set_ylim(0, None); ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    save(fig, OUT / fname)


horizon_plot(tabs, f"Fout per horizon - validatie binnen {TR_TXT}", "horizon_validatie.png")
summary = {"fase": 2, "run": RUN_NAME, "vraag": "echt vooruit voorspellen met alleen Kaggle-data", "opzet_dagen": f"1-{N_SETUP}",
           "horizons": {h: (f"{v} kwartier(en)" if v else "morgen, gemaakt om 21:00") for h, v in HORIZONS.items()},
           "keuze_per_horizon": {h: {"model": c[0], "motivatie": c[1]} for h, c in choices.items()},
           "validatie": {h: t.round(4).to_dict("index") for h, t in tabs.items()}}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

if not TOON_TEST:
    print(f"\nSTOP: keuzes per horizon staan vast. {TE_TXT} is nog NIET bekeken. "
          "Na akkoord: fase 2 toevoegen aan config.TEST_VRIJGEGEVEN.")
    sys.exit(0) if __name__ == "__main__" else None

# %% 3. Deel B - testdagen: A. bevroren (trainingsdagen) en B. walk-forward (alle eerdere dagen)
SETUP, TEST = range(1, N_SETUP + 1), range(N_SETUP + 1, N_DAYS + 1)
frozen, walk, per_plant, stab, ex = {}, {}, [], {}, {}
for hname, hz in HORIZONS.items():
    te, pr = run(SETUP, TEST, hz)
    frozen[hname] = pd.DataFrame({n: metrics(te.FRAC, y, te) for n, y in pr.items()}).T
    for p in PLANTS:
        m = (te.PLANT == p).values
        per_plant += [{"horizon": hname, "plant": f"Plant {p}", "model": n, **metrics(te.FRAC[m], y[m], te[m])}
                      for n, y in pr.items()]
    c = choices[hname][0]
    e2 = te.assign(e=(pr[c] - te.FRAC) ** 2).groupby(["PLANT", "DAY"]).e.mean() ** 0.5 * 100
    stab[hname] = {"model": c, "mean": e2.mean(), "std": e2.std(), "CV": e2.std() / e2.mean()}
    ex[hname] = (te, pr)
    wt, wp = [], {n: [] for n in pr}
    for D in TEST:
        t1, p1 = run(range(1, D), [D], hz)
        wt.append(t1)
        for n in p1:
            wp[n].append(p1[n])
    tw = pd.concat(wt)
    walk[hname] = pd.DataFrame({n: metrics(tw.FRAC, np.concatenate(v), tw) for n, v in wp.items()}).T
    print(f"\n=== {hname}: TEST {TE_TXT} (A = bevroren, B = walk-forward) | keuze: {c} ===")
    print(frozen[hname][["nRMSE_%", "nMAE_%", "R2", "dagfout_%"]].join(walk[hname][["nRMSE_%"]], rsuffix=" (B)")
          .round(3).to_string())
pp = pd.DataFrame(per_plant)
stab = pd.DataFrame(stab).T
print("\n=== Stabiliteit van het gekozen model per horizon (nRMSE per plant-dag) ==="); print(stab.round(3).to_string())
pd.concat(frozen).round(4).to_csv(OUT / "test_bevroren.csv"); pd.concat(walk).round(4).to_csv(OUT / "test_walkforward.csv")
pp.round(4).to_csv(OUT / "test_per_plant.csv", index=False); stab.to_csv(OUT / "stabiliteit.csv")

f1 = json.loads((RESULTS / "fase1" / "samenvatting.json").read_text(encoding="utf-8"))
upper = next(r["nRMSE_%"] for r in f1["bevroren"] if r["model"] == f1["keuze"] and r["plant"] == "samen")
horizon_plot(frozen, f"Fout per horizon - test {TE_TXT} (bevroren model)", "horizon_test.png", upper=upper)

# voorbeeld: day-ahead voor 3 dagen van Plant 1 (helder, half, bewolkt volgens de sensor)
te, pr = ex["day-ahead (21:00)"]
kt = te.groupby("DAY").KT_SENSOR.median()
days = {"helder": kt.idxmax(), "half bewolkt": (kt - kt.median()).abs().idxmin(), "bewolkt": kt.idxmin()}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, (lab, D) in zip(axs, days.items()):
    m = ((te.DAY == D) & (te.PLANT == 1)).values
    t = te.DATE_TIME[m].dt.hour + te.DATE_TIME[m].dt.minute / 60
    ax.plot(t, te.FRAC[m] * HOUSE_KWP, color=INK, lw=1.4, label="werkelijk")
    ax.plot(t, pr[choices["day-ahead (21:00)"][0]][m] * HOUSE_KWP, color=BLUE, lw=2,
            label=f"day-ahead: {choices['day-ahead (21:00)'][0]}")
    ax.plot(t, pr["Baseline: gemiddelde dag"][m] * HOUSE_KWP, color=GREY, ls="--", lw=1.2, label="gemiddelde dag")
    ax.set_title(f"Plant 1 - dag {D} ({lab})"); ax.set_xlabel("uur (IST)")
axs[0].set_ylabel("vermogen (kW, huis 5 kWp)"); axs[0].legend(fontsize=8, loc="upper left")
save(fig, OUT / "voorbeeld_dayahead.png")

summary.update({"test_dagen": f"{N_SETUP + 1}-{N_DAYS}", "bovengrens_fase1_nRMSE_%": upper,
                "test_bevroren": {h: t.round(4).to_dict("index") for h, t in frozen.items()},
                "test_walkforward": {h: t.round(4).to_dict("index") for h, t in walk.items()},
                "stabiliteit": stab.to_dict("index")})
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
plt.show()
print("\nKlaar met fase 2.")

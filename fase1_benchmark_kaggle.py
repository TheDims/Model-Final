# -*- coding: utf-8 -*-
"""
FASE 1  -  Regressie-benchmark op de Kaggle-data (30/70)
========================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Vraag: hoe goed verklaart regressie het vermogen uit de data van de plant zelf?
LET OP: dit is GEEN echte voorspelling. De sensor meet op hetzelfde moment als de opwek. Dit is de
BOVENGRENS (een huis heeft geen stralingssensor) en de basis voor fase 2-4.

  Deel A (altijd):  modelladder op de validatievouwen binnen de trainingsdagen -> keuze met de vaste regel.
  Deel B (pas na akkoord, config.TEST_VRIJGEGEVEN): de testdagen
       A. bevroren model (getraind op de trainingsdagen)    B. walk-forward (elke dag opnieuw op alle eerdere dagen)
       + per plant, kruis-plant, stabiliteit (mean/std/CV), residuen, voorbeelddagen, heatmaps.
De 70 % verandert de keuze NIET.

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
from solarcore.config import (RESULTS, CLEAN_FILE, PLANTS, MIN_COSZ, TIE_PP, COMPLEX_PP, GAMMA_PDC,  # noqa: E402
                              PLANT_COLORS, BLUE, ORANGE, GREY, INK, INK2, HOUSE_KWP, RUN_NAME, toon_test)
from solarcore.features import add_time, add_solar, add_clearsky, add_physical, capacity, add_target  # noqa: E402
from solarcore.models import LADDER_FASE1, BASELINES, ALL_SENSOR   # noqa: E402
from solarcore.evaluate import metrics, mean_day, persistence   # noqa: E402
from solarcore.split import day_split, val_folds   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

TOON_TEST = toon_test(1)   # per run vrijgegeven in config.TEST_VRIJGEGEVEN (na akkoord Dimme, AGENTS.md §6.2)
warnings.filterwarnings("ignore")
OUT = RESULTS / "fase1"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
NAMES = BASELINES + list(LADDER_FASE1)


# %% 1. Data + features
d = pd.read_csv(CLEAN_FILE, parse_dates=["DATE_TIME"])
d = add_physical(add_clearsky(add_solar(add_time(d))))
# eerlijk vergelijken: alle modellen op precies dezelfde kwartieren (overdag, target en sensor bekend)
d["OK"] = (d.COSZ > MIN_COSZ) & d.AC_KW.notna() & d[ALL_SENSOR + ["IRR_T"]].notna().all(axis=1)
MET_DIPS = False           # True = de 29 plant-brede dips laten staan (versie van 28 okt vóór BESLISSINGEN #10)
if not MET_DIPS:
    d["OK"] &= ~d.DIP
sp = day_split(d.DATE_TIME)
N_SETUP = sp["n_opzet"]
TR_TXT, TE_TXT = f"dag 1-{N_SETUP}", f"dag {N_SETUP + 1}-{sp['n_dagen']}"
print(f"Run {RUN_NAME} | training dag 1-{N_SETUP} ({sp['opzet_van_tot'][0]} t/m {sp['opzet_van_tot'][1]}), "
      f"test dag {N_SETUP + 1}-{sp['n_dagen']} | kwartieren overdag bruikbaar: {int(d.OK.sum())}")


def run(train_days, test_days, train_plants=PLANTS, test_plants=PLANTS, cap_from=None):
    """Train op (train_days, train_plants), voorspel (test_days, test_plants). Capaciteit per plant alleen
    uit de trainingsdagen (voor een testplant die niet in de training zit: uit zijn eigen dagen in cap_from).
    Geeft (testtabel, {model: voorspelling})."""
    tr_raw = d[d.DAY.isin(train_days)]
    cap = capacity(tr_raw if cap_from is None else d[d.DAY.isin(cap_from)])
    dd = add_target(d, cap)
    tr = dd[dd.DAY.isin(train_days) & dd.PLANT.isin(train_plants) & dd.OK]
    te = dd[dd.DAY.isin(test_days) & dd.PLANT.isin(test_plants) & dd.OK]
    full = dd[dd.PLANT.isin(test_plants) & dd.FRAC.notna()]
    pr = {"Baseline: gemiddelde dag": mean_day(tr if set(test_plants) <= set(train_plants)
                                               else tr.assign(PLANT=test_plants[0]), te),
          "Baseline: persistence (gisteren)": persistence(full, te)}
    for name, (cols, make, _) in LADDER_FASE1.items():
        pr[name] = np.clip(make().fit(tr[cols], tr.FRAC).predict(te[cols]), 0, None)
    return te, pr


def table(rows, by):
    t = pd.DataFrame(rows)
    return t.groupby(by, sort=False)[["nRMSE_%", "nMAE_%", "RMSE_kW", "MAE_kW", "R2", "dagfout_%"]].mean()


# %% 2. Deel A - validatievouwen binnen de trainingsdagen (keuze)
rows = []
for fold, tr_df, va_df in val_folds(d, N_SETUP):
    te, pr = run(sorted(tr_df.DAY.unique()), sorted(va_df.DAY.unique()))
    for name, yhat in pr.items():
        rows.append({"model": name, "vouw": fold, **metrics(te.FRAC, yhat, te)})
val = pd.DataFrame(rows)
val_tab = table(rows, "model")
per_fold = val.pivot(index="model", columns="vouw", values="nRMSE_%").loc[NAMES]
val_tab = val_tab.join(per_fold.add_prefix("nRMSE_% "))
for ref, col in (("Baseline: persistence (gisteren)", "skill_pers"), ("Baseline: gemiddelde dag", "skill_gemdag")):
    val_tab[col] = 1 - val_tab["nRMSE_%"] / val_tab.loc[ref, "nRMSE_%"]
print(f"\n=== Validatie binnen {TR_TXT} (gemiddelde van {val.vouw.nunique()} vouwen) - hier wordt gekozen ===")
print(val_tab.round(3).to_string())
val.to_csv(OUT / "validatie_per_vouw.csv", index=False)
val_tab.round(4).to_csv(OUT / "validatie.csv")

# %% 3. Keuzeregel (vooraf vast, BESLISSINGEN #2)
cand = val_tab.loc[list(LADDER_FASE1)].copy()
cand["eenvoud"] = [LADDER_FASE1[n][2] for n in cand.index]
best = cand["nRMSE_%"].idxmin()
simple = cand[cand.eenvoud.map(lambda e: e[0] == 0)]
best_simple = simple["nRMSE_%"].idxmin()
fold_cols = [c for c in cand.columns if c.startswith("nRMSE_% train")]
if LADDER_FASE1[best][2][0] == 1:                       # het beste model is complex (RF/GB)
    gain = cand.loc[best_simple, "nRMSE_%"] - cand.loc[best, "nRMSE_%"]
    both = all(cand.loc[best, c] < cand.loc[best_simple, c] for c in fold_cols)
    if not (gain >= COMPLEX_PP and both):
        why_complex = (f"'{best}' (complex) is maar {gain:.2f} pp beter dan '{best_simple}'"
                       f"{'' if both else ' en niet op alle vouwen beter'} -> geen complex model")
        best = best_simple
    else:
        why_complex = f"complex model '{best}' is {gain:.2f} pp beter en op alle vouwen beter"
else:
    why_complex = "het beste model is geen complex model"
close = cand[(cand["nRMSE_%"] <= cand.loc[best, "nRMSE_%"] + TIE_PP) & cand.eenvoud.map(lambda e: e[0] == 0)]
CHOICE = min(close.index, key=lambda n: LADDER_FASE1[n][2])
why = (f"laagste nRMSE: '{best}' ({cand.loc[best, 'nRMSE_%']:.2f} %); {why_complex}. Binnen {TIE_PP} pp: "
       f"{list(close.index)} -> eenvoudigste: '{CHOICE}' ({cand.loc[CHOICE, 'nRMSE_%']:.2f} %)")
print("\nKEUZE:", CHOICE); print("Waarom:", why)
summary = {"fase": 1, "run": RUN_NAME, "vraag": "regressie op de plant-sensor (bovengrens, geen echte voorspelling)",
           "opzet_dagen": f"1-{N_SETUP}", "validatievouwen": list(per_fold.columns),
           "keuze": CHOICE, "inputs": LADDER_FASE1[CHOICE][0], "motivatie": why,
           "gamma_aanname": GAMMA_PDC, "validatie": val_tab.round(4).to_dict("index")}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

if not TOON_TEST:
    print(f"\nSTOP: de keuze staat vast. De testdagen ({TE_TXT}) zijn nog NIET bekeken. "
          "Na akkoord: fase 1 toevoegen aan config.TEST_VRIJGEGEVEN.")
    sys.exit(0) if __name__ == "__main__" else None

# %% 4. Deel B - de testdagen: A. bevroren model (getraind op de trainingsdagen)
SETUP = list(range(1, N_SETUP + 1)); TEST = list(range(N_SETUP + 1, sp["n_dagen"] + 1))
te_f, pr_f = run(SETUP, TEST)
rows_f = [{"model": n, "plant": "samen", **metrics(te_f.FRAC, y, te_f)} for n, y in pr_f.items()]
for p in PLANTS:
    m = (te_f.PLANT == p).values
    rows_f += [{"model": n, "plant": f"Plant {p}", **metrics(te_f.FRAC[m], y[m], te_f[m])} for n, y in pr_f.items()]
frozen = pd.DataFrame(rows_f).set_index(["model", "plant"])
for ref, col in (("Baseline: persistence (gisteren)", "skill_pers"), ("Baseline: gemiddelde dag", "skill_gemdag")):
    frozen[col] = 1 - frozen["nRMSE_%"] / frozen.xs(ref)["nRMSE_%"].reindex(frozen.index.get_level_values(1)).values
print(f"\n=== A. BEVROREN model (train {TR_TXT}, test {TE_TXT}) ===")
print(frozen.xs("samen", level=1).round(3).to_string())

# %% 5. B. walk-forward: elke testdag opnieuw trainen op alle eerdere dagen
wf_pred = {n: [] for n in NAMES}; wf_te = []
for D in TEST:
    te, pr = run(list(range(1, D)), [D])
    wf_te.append(te)
    for n in NAMES:
        wf_pred[n].append(pr[n])
te_w = pd.concat(wf_te); pr_w = {n: np.concatenate(v) for n, v in wf_pred.items()}
walk = pd.DataFrame([{"model": n, **metrics(te_w.FRAC, y, te_w)} for n, y in pr_w.items()]).set_index("model")
print("\n=== B. WALK-FORWARD (elke dag opnieuw, alle eerdere dagen) ===")
print(walk.round(3).to_string())

# %% 6. Kruis-plant ('ander dak'): train op de ene plant (trainingsdagen), test op de andere (testdagen)
rows_x = []
for a, b in [(1, 2), (2, 1)]:
    te, pr = run(SETUP, TEST, train_plants=[a], test_plants=[b], cap_from=SETUP)
    rows_x += [{"richting": f"P{a} -> P{b}", "model": n, **metrics(te.FRAC, y, te)} for n, y in pr.items()]
cross = pd.DataFrame(rows_x).pivot(index="model", columns="richting", values="nRMSE_%").loc[NAMES]
print("\n=== Kruis-plant (nRMSE %, bevroren) ==="); print(cross.round(2).to_string())

# %% 7. Stabiliteit: nRMSE per plant-dag en per week (bevroren)
e2 = te_f.assign(**{f"e_{n}": (pr_f[n] - te_f.FRAC) ** 2 for n in NAMES})
per_day = pd.DataFrame({n: e2.groupby(["PLANT", "DAY"])[f"e_{n}"].mean() ** 0.5 * 100 for n in NAMES})
stab = per_day.agg(["mean", "std"]).T
stab["CV"] = stab["std"] / stab["mean"]
week = te_f.DATE_TIME.dt.to_period("W-SUN").astype(str)
per_week = pd.DataFrame({n: e2.groupby(week)[f"e_{n}"].mean() ** 0.5 * 100 for n in NAMES})
print(f"\n=== Stabiliteit (nRMSE per plant-dag, {len(per_day)} plant-dagen) ==="); print(stab.round(3).to_string())

# opslaan
frozen.round(4).to_csv(OUT / "vergelijking_bevroren.csv"); walk.round(4).to_csv(OUT / "vergelijking_walkforward.csv")
cross.round(4).to_csv(OUT / "kruisplant.csv"); stab.round(4).to_csv(OUT / "stabiliteit.csv")
per_day.round(3).to_csv(OUT / "nrmse_per_dag.csv"); per_week.round(3).to_csv(OUT / "nrmse_per_week.csv")
summary.update({"test_dagen": f"{TEST[0]}-{TEST[-1]}",
                "bevroren": frozen.round(4).reset_index().to_dict("records"),
                "walkforward": walk.round(4).to_dict("index"), "kruisplant_nRMSE": cross.round(4).to_dict(),
                "stabiliteit": stab.round(4).to_dict("index")})
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

# %% 8. Grafieken
style()
SHORT = {n: n.replace("Baseline: ", "").replace(" (gisteren)", "") for n in NAMES}

# 8.1 heatmap model x plant (nRMSE, bevroren)
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
BLUES = LinearSegmentedColormap.from_list("blauw", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
H = frozen["nRMSE_%"].unstack("plant").loc[NAMES, ["Plant 1", "Plant 2", "samen"]]


def heat(M, rows, cols, title, fname, fmt="{:.1f}", size=(6.8, 4.6), xlabel=""):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(M, cmap=BLUES, aspect="auto")
    ax.set_xticks(range(len(cols)), cols); ax.set_yticks(range(len(rows)), rows)
    ax.grid(False); ax.tick_params(length=0)
    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if (M[i, j] - lo) / (hi - lo + 1e-9) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("nRMSE (%) - lager is beter")
    cb.outline.set_visible(False)
    ax.set_title(title); ax.set_xlabel(xlabel)
    save(fig, OUT / fname)


heat(H.values, [SHORT[n] for n in NAMES], list(H.columns), f"nRMSE per model en plant (bevroren, {TE_TXT})",
     "heatmap_model_plant.png", xlabel="plant", size=(8.6, 4.6))
heat(per_day.groupby(level="DAY").mean().T.loc[NAMES].values, [SHORT[n] for n in NAMES],
     [str(x) for x in TEST], "nRMSE per model en testdag (gemiddelde van beide plants)", "heatmap_model_dag.png",
     fmt="{:.0f}", size=(12, 4.6), xlabel="testdag")

# 8.2 voorbeelddagen: helder, half bewolkt, bewolkt (op basis van KT_SENSOR per dag, beide plants)
kt_day = te_f.groupby("DAY").KT_SENSOR.median()
ex = {"helder": kt_day.idxmax(), "half bewolkt": (kt_day - kt_day.median()).abs().idxmin(), "bewolkt": kt_day.idxmin()}
fig, axs = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
for j, (lab, D) in enumerate(ex.items()):
    for i, p in enumerate(PLANTS):
        m = ((te_f.DAY == D) & (te_f.PLANT == p)).values
        ax = axs[i, j]
        t = te_f.DATE_TIME[m].dt.hour + te_f.DATE_TIME[m].dt.minute / 60
        ax.plot(t, te_f.FRAC[m] * HOUSE_KWP, color=INK, lw=1.4, label="werkelijk")
        ax.plot(t, pr_f[CHOICE][m] * HOUSE_KWP, color=BLUE, lw=2, label=f"{CHOICE} (bevroren)")
        ax.plot(t, pr_f["Baseline: gemiddelde dag"][m] * HOUSE_KWP, color=GREY, lw=1.2, ls="--", label="gemiddelde dag")
        ax.set_title(f"Plant {p} - dag {D} ({lab})")
        if j == 0:
            ax.set_ylabel("vermogen (kW, huis 5 kWp)")
        if i == 1:
            ax.set_xlabel("uur (IST)")
axs[0, 0].legend(loc="upper left", fontsize=8)
save(fig, OUT / "voorbeelddagen.png")

# 8.3 residuen van het gekozen model (bevroren): tijd van de dag, voorspelde waarde, bewolking, histogram, ACF
from statsmodels.graphics.tsaplots import plot_acf   # noqa: E402
res = (pr_f[CHOICE] - te_f.FRAC.values) * 100           # in % van de capaciteit
hour = te_f.DATE_TIME.dt.hour + te_f.DATE_TIME.dt.minute / 60
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for p in PLANTS:
    m = (te_f.PLANT == p).values
    axs[0, 0].scatter(hour[m], res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0, label=f"Plant {p}")
    axs[0, 1].scatter(pr_f[CHOICE][m] * 100, res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0)
    axs[0, 2].scatter(te_f.KT_SENSOR[m], res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0)
    axs[1, 0].hist(res[m], bins=60, alpha=0.55, color=PLANT_COLORS[p], label=f"Plant {p}")
for ax in (axs[0, 0], axs[0, 1], axs[0, 2]):
    ax.axhline(0, color=INK2, lw=1)
axs[0, 0].set(title="Residu tegen tijd van de dag", xlabel="uur (IST)", ylabel="residu (% van capaciteit)")
axs[0, 0].legend(markerscale=3)
axs[0, 1].set(title="Residu tegen voorspelde waarde", xlabel="voorspeld (% van capaciteit)", ylabel="residu (%)")
axs[0, 2].set(title="Residu tegen helderheid (KT, sensor)", xlabel="KT = gemeten / heldere hemel (-)", ylabel="residu (%)")
axs[1, 0].set(title="Verdeling van de residuen", xlabel="residu (% van capaciteit)", ylabel="aantal kwartieren")
axs[1, 0].legend()
for k, p in enumerate(PLANTS):
    m = (te_f.PLANT == p).values
    plot_acf(res[m], lags=40, ax=axs[1, 1 + k], color=PLANT_COLORS[p], vlines_kwargs={"colors": PLANT_COLORS[p]})
    axs[1, 1 + k].set(title=f"Autocorrelatie residu Plant {p} (overdag, op volgorde)", xlabel="lag (kwartieren)",
                      ylabel="ACF (-)")
fig.suptitle(f"Residuen van '{CHOICE}' (bevroren, {TE_TXT}): voorspeld - werkelijk", x=0.01, ha="left", color=INK)
save(fig, OUT / "residuen.png")
plt.show()
print("\nKlaar met fase 1.")

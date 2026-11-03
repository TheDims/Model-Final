# -*- coding: utf-8 -*-
"""
FASE 4  -  Eindmodel: kiezen, trainen op alle dagen, band, opslaan en documenteren
==================================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

  1. Keuze = de uitkomst van de keuzeregel in fase 3 van DEZE run (op de validatievouwen, niet op de test).
     De testscores blijven die van fase 3. De beste andere ERA5-modellen staan ernaast als vergelijking.
  2. Band P10-P90 per bewolkingsklasse (ERA5-helderheid KT), gekalibreerd OUT-OF-SAMPLE op walk-forward-
     voorspellingen (elke dag voorspeld door een model dat alleen eerdere dagen zag):
       - run met >= 11 trainingsdagen (run 2): kalibratie binnen de training (dag 11 .. laatste trainingsdag),
         dekking gemeten op de testdagen met het bevroren model (BESLISSINGEN #17, AGENTS.md §6.3);
       - run 1 (10 trainingsdagen): kalibratie op de eerste helft van de testdagen, check op de tweede helft.
  3. Eindmodel trainen op ALLE 34 dagen (beide plants, zonder dips).
  4. Naar een huis: kW = FRAC x kWp x factor x verliesfactor (factor uit de natuurkunde = AANNAME).
  5. Opslaan: modellen/model_final_<RUN>.joblib (zelfde sleutels als model_v5.joblib) + resultaten/<RUN>/model_info.json.
     modellen/model_final.joblib (run 1) wordt NIET overschreven.

Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase 0-3 van dezelfde run.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (RESULTS, MODELS, MODEL_FILE, MIN_COSZ, HOUSE_KWP, CAPACITY_QUANTILE, PLANTS,   # noqa: E402
                              BAND_CAL_START, RUN_NAME, BLUE, ORANGE, GREY, INK, INK2)
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.models import LADDER_FASE3   # noqa: E402
from solarcore.physics import PARAMS   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "fase4"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
KT_BINS = [0, 0.45, 0.75, 9]                 # bewolkt | half bewolkt | (bijna) helder (zoals V3-V5)
f3 = json.loads((RESULTS / "fase3" / "samenvatting.json").read_text(encoding="utf-8"))
CHOICE = f3["keuze"]
RAD_SHIFT = int(f3["tijdafspraak_min"])
COLS, MAKE, _ = LADDER_FASE3[CHOICE]
print(f"Run {RUN_NAME} | eindmodel: {CHOICE} | inputs {COLS} | tijdafspraak ERA5 {RAD_SHIFT:+d} min")

# %% 1. Data, walk-forward (out-of-sample) en de band
data = weather_frame(RAD_SHIFT)
N_DAYS = int(data.DAY.max())
N_SETUP = int(f3["opzet_dagen"].split("-")[1])
TRAIN, TEST = list(range(1, N_SETUP + 1)), list(range(N_SETUP + 1, N_DAYS + 1))
TE_TXT = f"dag {TEST[0]}-{TEST[-1]}"


def fit_predict(train_days, test_days, cols=COLS, make=MAKE):
    dd = add_target(data, capacity(data[data.DAY.isin(train_days)]))
    tr, te = dd[dd.DAY.isin(train_days) & dd.OK], dd[dd.DAY.isin(test_days) & dd.OK]
    m = make().fit(tr[cols], tr.FRAC)
    return te, np.clip(m.predict(te[cols]), 0, None) * (te.COSZ.values > MIN_COSZ), m


def walk_forward(days):
    """Elke dag D voorspeld door een model dat alleen dag 1..D-1 zag."""
    out = []
    for D in days:
        te, y, _ = fit_predict(range(1, D), [D])
        out.append(te[["DATE_TIME", "PLANT", "DAY", "FRAC", "KT_W"]].assign(pred=y))
    return pd.concat(out)


def fit_band(cal):
    ok = cal.pred > 0.05
    r = pd.DataFrame({"r": cal.FRAC.values[ok] / cal.pred.values[ok],
                      "k": np.digitize(cal.KT_W.values[ok], KT_BINS) - 1})
    return {str(int(k)): (float(g.r.quantile(0.10)), float(g.r.quantile(0.90))) for k, g in r.groupby("k")}


def apply_band(band, df):
    k = (np.digitize(df.KT_W.values, KT_BINS) - 1).astype(int).astype(str)
    lo = np.array([band.get(x, (0.8, 1.2))[0] for x in k]); hi = np.array([band.get(x, (0.8, 1.2))[1] for x in k])
    return df.pred.values * lo, df.pred.values * hi


def coverage(df, p10, p90):
    return float(np.mean((df.FRAC.values >= p10) & (df.FRAC.values <= p90)) * 100)


if N_SETUP >= BAND_CAL_START:                                # run 2: kalibratie binnen de training
    cal = walk_forward(range(BAND_CAL_START, N_SETUP + 1))
    BAND = fit_band(cal)
    te_f, y_f, _ = fit_predict(TRAIN, TEST)                  # bevroren model op de testdagen
    band_eval = te_f[["DATE_TIME", "PLANT", "DAY", "FRAC", "KT_W"]].assign(pred=y_f)
    cov_check = coverage(band_eval, *apply_band(BAND, band_eval))
    cov_cal = coverage(cal, *apply_band(BAND, cal))
    check_txt = (f"kalibratie op walk-forward dag {BAND_CAL_START}-{N_SETUP} (training), "
                 f"dekking gemeten op {TE_TXT} (bevroren model)")
    pred_label = "voorspeld voor morgen (bevroren model)"
else:                                                        # run 1: te weinig trainingsdagen
    oos = walk_forward(TEST)
    half = TEST[len(TEST) // 2]
    cov_check = coverage(oos[oos.DAY >= half], *apply_band(fit_band(oos[oos.DAY < half]), oos[oos.DAY >= half]))
    BAND = fit_band(oos)
    cov_cal = coverage(oos, *apply_band(BAND, oos))
    band_eval = oos[oos.DAY >= half]
    check_txt = f"kalibratie dag {TEST[0]}-{half - 1}, dekking dag {half}-{TEST[-1]} (walk-forward)"
    pred_label = "voorspeld voor morgen (walk-forward)"
ev = band_eval.assign(p10=apply_band(BAND, band_eval)[0], p90=apply_band(BAND, band_eval)[1])
dsum = ev.groupby(["PLANT", "DAY"])[["FRAC", "p10", "p90"]].sum()
cov_day = float(((dsum.FRAC >= dsum.p10) & (dsum.FRAC <= dsum.p90)).mean() * 100)
print(f"\nBand ({check_txt}): dekking {cov_check:.1f} % (doel 80 %) | op de kalibratiedagen {cov_cal:.1f} % | "
      f"dagband (som P10..som P90): {cov_day:.0f} % van de dagen")
print("Band per klasse:", {k: tuple(round(x, 2) for x in v) for k, v in BAND.items()})

# %% 2. Eindmodel op alle 34 dagen + factor voor een huis
cap_all = capacity(data)
tr_all = add_target(data, cap_all)
tr_all.loc[tr_all.DIP, "FRAC"] = np.nan
tr_all = tr_all[tr_all.OK & tr_all.FRAC.notna()]
final = MAKE().fit(tr_all[COLS], tr_all.FRAC)


def formula(model, cols):
    """Coëfficiënten per eenheid van de input (ook als er een StandardScaler in de Pipeline zit)."""
    est, sc = (model[-1], model[0]) if hasattr(model, "steps") else (model, None)
    coef = np.asarray(est.coef_, float)
    icpt = float(np.atleast_1d(getattr(est, "intercept_", 0.0))[0]) if getattr(est, "fit_intercept", True) else 0.0
    if sc is not None:
        mean = getattr(sc, "mean_", None)
        coef = coef / sc.scale_
        icpt = icpt - float(np.sum(coef * mean)) if mean is not None else icpt
    return icpt, dict(zip(cols, map(float, coef)))


intercept, coefs = formula(final, COLS)
# FACTOR (AANNAME): kW per kWp van de natuurkunde bij FRAC = 1 (99e percentiel van de pvlib-opwek overdag,
# zelfde definitie als de capaciteit), gemiddeld over de plants.
fac = tr_all.groupby("PLANT").PHYS.quantile(CAPACITY_QUANTILE)
FACTOR = float(fac.mean())
print(f"\nEindmodel (getraind op dag 1-{N_DAYS}): FRAC = {intercept:+.5f} "
      + " ".join(f"{c:+.6f} x {k}" for k, c in coefs.items()) + "  (0 als de zon onder is)")
print(f"Factor (kW per kWp bij FRAC = 1): {FACTOR:.3f}  (per plant: {fac.round(3).to_dict()})")
joblib.dump({"versie": f"Model Final ({RUN_NAME})", "run": RUN_NAME, "variant": CHOICE, "features": COLS,
             "model": final, "factor": FACTOR, "band_method": "klasse", "band": BAND, "kt_bins": KT_BINS,
             "kt_kolom": "KT_W", "params": PARAMS, "rad_shift_min": RAD_SHIFT, "min_cosz": MIN_COSZ,
             "capaciteit_kwantiel": CAPACITY_QUANTILE}, MODEL_FILE)

# %% 3. Eindtabel over alle fases (testdagen, bevroren model) + model_info
f0 = json.loads((RESULTS / "fase0" / "samenvatting.json").read_text(encoding="utf-8"))
f1 = json.loads((RESULTS / "fase1" / "samenvatting.json").read_text(encoding="utf-8"))
f2 = json.loads((RESULTS / "fase2" / "samenvatting.json").read_text(encoding="utf-8"))
b3 = {r["model"]: r for r in f3["bevroren"] if r["plant"] == "samen"}
p3 = {(r["model"], r["plant"]): r["nRMSE_%"] for r in f3["bevroren"]}
era_models = [n for n in LADDER_FASE3 if n in b3]
best_test = min(era_models, key=lambda n: b3[n]["nRMSE_%"])
compare = list(dict.fromkeys([n for n in ["LR GHI (ERA5)", "e5: EFF_T, KT_POA, CS_POA, COSZ (LR met intercept)",
                                          "e6: EFF_T x (a + b KT_POA), zonder intercept", best_test]
                              if n in b3 and n != CHOICE]))
rows = [("Baseline: gemiddelde dag", "geen", b3["Baseline: gemiddelde dag"]),
        ("Baseline: persistence (gisteren)", "geen", b3["Baseline: persistence (gisteren)"]),
        ("Fase 2: smart persistence (day-ahead, geen weer)", "eigen opwek", b3["Fase 2: smart persistence (geen weer)"]),
        (f"Fase 3: {CHOICE} – EINDMODEL", "ERA5-weer", b3[CHOICE])]
rows += [(f"Fase 3: {n} (vergelijking)", "ERA5-weer", b3[n]) for n in compare]
rows += [("Fase 1: LR straling (sensor, bovengrens)", "plant-sensor", b3["Fase 1: LR straling (sensor, bovengrens)"])]
final_tab = pd.DataFrame([{"model": n, "input": i, "nRMSE_%": r["nRMSE_%"], "nMAE_%": r["nMAE_%"],
                           "RMSE_kW": r["RMSE_kW"], "MAE_kW": r["MAE_kW"], "R2": r["R2"], "dagfout_%": r["dagfout_%"],
                           "skill_gemdag": r["skill_gemdag"]} for n, i, r in rows]).set_index("model")
print(f"\n=== Eindtabel (test {TE_TXT}, bevroren op dag 1-{N_SETUP}; huis 5 kWp) ===")
print(final_tab.round(3).to_string())
final_tab.round(4).to_csv(OUT / "eindtabel.csv")

n_dips = sum(v.get("dip_kwartieren", 0) for v in f0["per_plant"].values())
na_test = (f"Op de testdagen was '{best_test}' het beste ERA5-model ({b3[best_test]['nRMSE_%']:.2f} % nRMSE; gekozen "
           f"model {b3[CHOICE]['nRMSE_%']:.2f} %). De test verandert de keuze niet (AGENTS.md §6.2)."
           if best_test != CHOICE else "Het gekozen model was ook op de testdagen het beste ERA5-model.")
info = {
    "run": RUN_NAME, "model": CHOICE, "inputs": COLS,
    "type": "Linear Regression" + (" zonder intercept" if not getattr(final[-1] if hasattr(final, "steps") else final,
                                                                      "fit_intercept", True) else "") + ", nacht = 0",
    "formule_FRAC": {"intercept": intercept, **coefs},
    "keuze": {"regel": "laagste gemiddelde nRMSE op de validatievouwen binnen de trainingsdagen; < 0,5 pp -> "
                       "eenvoudigste; complex alleen bij >= 1 pp en op alle vouwen beter (BESLISSINGEN #2/#17)",
              "motivatie": f3["motivatie"], "na_de_test": na_test},
    "data": {"trainingsdagen": f"1-{N_SETUP}", "test_dagen": f"{TEST[0]}-{TEST[-1]}",
             "eindmodel_getraind_op": f"1-{N_DAYS}", "periode": "15 mei - 17 juni 2020",
             "plants": "Kaggle Plant 1 + 2 (India), samen getraind",
             "weggelaten": f"omvormer-uitval en {n_dips} plant-brede dips"},
    "test_bevroren": {**b3[CHOICE], "per_plant_nRMSE_%": {f"Plant {p}": p3[(CHOICE, f'Plant {p}')] for p in PLANTS}},
    "test_walkforward_nRMSE_%": f3["walkforward"][CHOICE]["nRMSE_%"],
    "band": {"methode": "per bewolkingsklasse (ERA5-helderheid KT_W)", "kt_bins": KT_BINS, "per_klasse": BAND,
             "dekking_check_%": cov_check, "check": check_txt, "dekking_kalibratiedagen_%": cov_cal,
             "dagband_dekking_%": cov_day},
    "naar_een_huis": {"formule": "kW = FRAC x kWp x factor x verliesfactor, begrensd op de omvormer; nacht = 0",
                      "factor_kW_per_kWp": FACTOR, "factor_per_plant": {str(k): float(v) for k, v in fac.items()}},
    "tijdafspraak_ERA5_min": RAD_SHIFT,
    "aannames": [
        "factor = 99e percentiel van de pvlib-opwek (kW/kWp, V5-parameters) overdag op de geschatte plantlocaties, "
        "gemiddeld over beide plants (FRAC 1 = 99e percentiel van de opwek).",
        "Plantlocaties geschat (ERA5-correlatie); opstelling plants: vast, hoek = breedtegraad, zuid.",
        "ERA5 (weer achteraf) als stand-in voor een weersvoorspelling: scores zijn een bovengrens."],
    "beperkingen": [f"34 dagen in mei-juni (begin moesson), 2 plants, {N_SETUP} trainingsdagen, "
                    f"{len(TEST)} testdagen ({2 * len(TEST)} plant-dagen)"],
    "eindtabel": final_tab.round(4).to_dict("index"),
    "fase1_bovengrens": f1["keuze"], "fase2_dayahead": f2["keuze_per_horizon"]["day-ahead (21:00)"]["model"],
    "bronnen": ["Kaggle Solar Power Generation Data (anikannal)", "Open-Meteo Historical API / ERA5 (CC BY 4.0)",
                "pvlib (Ineichen, Perez, Faiman, Kimber, PVWatts)", "scikit-learn"]}
(RESULTS / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
(OUT / "samenvatting.json").write_text(json.dumps({"fase": 4, **info}, indent=2, ensure_ascii=False, default=float),
                                       encoding="utf-8")

# %% 4. Grafieken: eindvergelijking en de band op voorbeelddagen
style()
v = final_tab["nRMSE_%"][::-1]
fig, ax = plt.subplots(figsize=(10, 0.55 * len(v) + 1.4))
cols = [BLUE if "EINDMODEL" in n else (ORANGE if "sensor" in n else GREY) for n in v.index]
ax.barh(range(len(v)), v.values, color=cols, height=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.2, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(v)), [n.replace(" (vergelijking)", " (vgl.)") for n in v.index]); ax.grid(axis="y", visible=False)
ax.set_xlabel("nRMSE (% van capaciteit) - lager is beter")
ax.set_title(f"Eindvergelijking {RUN_NAME}: voorspelling voor morgen (test {TE_TXT})")
save(fig, OUT / "eindvergelijking.png")

days = sorted(ev.DAY.unique())[-5:]
fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
for ax, p in zip(axs, PLANTS):
    s = ev[(ev.PLANT == p) & ev.DAY.isin(days)].set_index("DATE_TIME").asfreq("15min")
    ax.fill_between(s.index, s.p10 * HOUSE_KWP, s.p90 * HOUSE_KWP, color=BLUE, alpha=0.18, lw=0, label="P10-P90")
    ax.plot(s.index, s.FRAC * HOUSE_KWP, color=INK, lw=1.2, label="werkelijk")
    ax.plot(s.index, s.pred * HOUSE_KWP, color=BLUE, lw=2, label=pred_label)
    ax.set_title(f"Plant {p} - laatste 5 testdagen, met de band"); ax.set_ylabel("vermogen (kW, huis 5 kWp)")
axs[0].legend(ncol=3, loc="upper right", fontsize=8)
save(fig, OUT / "band_voorbeeld.png")
plt.show()
print(f"\nKlaar met fase 4. Model: {MODEL_FILE.relative_to(ROOT)} | info: {(RESULTS / 'model_info.json').relative_to(ROOT)}")

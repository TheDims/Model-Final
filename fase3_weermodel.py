# -*- coding: utf-8 -*-
"""
FASE 3  -  Weermodel (ERA5) als input in plaats van de plant-sensor
===================================================================
Model Final - SolarHome Forecast (MADS4E, ISEP)

Vraag: hoe goed is het model als we de plant-sensor vervangen door een bestaand weermodel?
Een huis heeft geen stralingssensor, wel een weersvoorspelling. ERA5 is weer ACHTERAF (heranalyse): de scores
zijn een BOVENGRENS voor echt gebruik met een voorspelling.

  0. Tijdafspraak: bij welke verschuiving past de ERA5-straling het best bij de sensor? (alleen trainingsdagen)
  1. Natuurkunde (pvlib, overgenomen uit V5): POA, celtemperatuur, EFF_T, KT_POA.
  2. Feature-ladder: GHI -> POA -> EFF_T -> e5 -> e6 -> Ridge-e6 -> e6 + bewolking/regen/vochtigheid (+ RF/GB).
     Keuze op de validatievouwen binnen de trainingsdagen (regel #2/#17). Coëfficiënten a en b per vouw (collineariteit).
  Deel B (pas na akkoord, config.TEST_VRIJGEGEVEN): testdagen bevroren + walk-forward, per plant, kruis-plant,
     stabiliteit, residuen, gevoeligheid voor weerfouten (10/20/30 %).
Nooit de plant-sensor als input (alleen als bovengrens/vergelijking en voor de tijdafspraak).

Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase0, fase1 en fase2.
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
from solarcore.config import (RESULTS, CLEAN_FILE, PLANTS, PLANT_LOCATIONS, MIN_COSZ, HOUSE_KWP,   # noqa: E402
                              PLANT_COLORS, BLUE, ORANGE, GREY, INK, INK2, RANDOM_STATE, RUN_NAME, toon_test)
from solarcore.features import add_time, add_solar, add_clearsky, capacity, add_target   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.weather import load_era5   # noqa: E402
from solarcore.physics import hourly_to_15min, soiling_loss, panel_inputs   # noqa: E402
from solarcore.models import LADDER_FASE3, LADDER_FASE1   # noqa: E402
from solarcore.evaluate import metrics, choose, bootstrap_days   # noqa: E402
from solarcore.split import day_split, fold_days   # noqa: E402
from solarcore.plots import style, save   # noqa: E402

TOON_TEST = toon_test(3)   # per run vrijgegeven in config.TEST_VRIJGEGEVEN (na akkoord Dimme, AGENTS.md §6.2)
warnings.filterwarnings("ignore")
OUT = RESULTS / "fase3"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
SHIFTS = [-60, -45, -30, -15, 0, 15, 30, 45, 60]                       # minuten
WEATHER_COLS = ["GHI_W", "CS_GHI_W", "KT_W", "T", "RH", "WS10", "CLOUD", "PRECIP", "POA", "EFF_T", "KT_POA",
                "EFF_T_KT", "T_CELL", "PHYS", "SOILING", "CS_POA"]
REF_SENSOR = "Fase 1: LR straling (sensor, bovengrens)"
REF_DA = "Fase 2: smart persistence (geen weer)"
style()

# %% 0. Data + tijdafspraak ERA5 (alleen trainingsdagen)
clean = pd.read_csv(CLEAN_FILE, parse_dates=["DATE_TIME"])
clean = add_clearsky(add_solar(add_time(clean)))
sp = day_split(clean.DATE_TIME)
N_SETUP, N_DAYS = sp["n_opzet"], sp["n_dagen"]
era = {p: load_era5(*PLANT_LOCATIONS[p]) for p in PLANTS}
VAL = [(range(1, last + 1), tuple(val)) for last, val in fold_days(N_SETUP)]   # BESLISSINGEN #17
TR_TXT, TE_TXT = f"dag 1-{N_SETUP}", f"dag {N_SETUP + 1}-{N_DAYS}"
print(f"Run {RUN_NAME} | training {TR_TXT}, test {TE_TXT}")

rows = []
for p in PLANTS:
    h, elev = era[p]
    s = clean[(clean.PLANT == p) & (clean.DAY <= N_SETUP)][["DATE_TIME", "IRRADIATION"]]
    for sh in SHIFTS:
        w = hourly_to_15min(h, *PLANT_LOCATIONS[p], elev, rad_shift_min=sh)
        m = s.merge(w[["DATE_TIME", "GHI_W", "ZENITH"]], on="DATE_TIME").dropna()
        m = m[np.cos(np.deg2rad(m.ZENITH)) > 0.1]
        rows.append({"plant": p, "verschuiving_min": sh, "r": m.IRRADIATION.corr(m.GHI_W),
                     "RMSE_W_m2": float(np.sqrt(((m.GHI_W - m.IRRADIATION * 1000) ** 2).mean()))})
shift = pd.DataFrame(rows)
piv = shift.pivot(index="verschuiving_min", columns="plant", values="r")
RAD_SHIFT = int(piv.mean(axis=1).idxmax())
print(f"Tijdafspraak (correlatie ERA5-GHI met sensor, kwartieren, {TR_TXT}):")
print(shift.pivot(index="verschuiving_min", columns="plant", values=["r", "RMSE_W_m2"]).round(3).to_string())
print(f"-> gekozen verschuiving: {RAD_SHIFT} min (straling van uur h hoort bij het interval met midden h {RAD_SHIFT:+d} min)")
shift.to_csv(OUT / "tijdafspraak.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for p in PLANTS:
    ax.plot(piv.index, piv[p], marker="o", ms=6, mec="white", lw=2, color=PLANT_COLORS[p], label=f"Plant {p}")
ax.axvline(RAD_SHIFT, color=INK2, ls="--", lw=1)
ax.set_xlabel("verschuiving ERA5-straling (minuten)"); ax.set_ylabel("correlatie met de sensor (r)")
ax.set_title(f"Tijdafspraak ERA5: welke verschuiving past het best? ({TR_TXT})"); ax.legend()
save(fig, OUT / "tijdafspraak.png")

# %% 1. Natuurkunde per plant (opstelling: hoek = breedtegraad, zuid - AANNAME uit V3-V5)
parts = []
for p in PLANTS:
    h, elev = era[p]
    w = hourly_to_15min(h, *PLANT_LOCATIONS[p], elev, rad_shift_min=RAD_SHIFT)
    d = panel_inputs(w, soiling_loss(h, pd.DatetimeIndex(w.DATE_TIME)), *PLANT_LOCATIONS[p], elev)
    parts.append(d[["DATE_TIME"] + WEATHER_COLS].assign(PLANT=p))
data = clean.merge(pd.concat(parts), on=["DATE_TIME", "PLANT"], how="left")
data["OK"] = ((data.COSZ > MIN_COSZ) & data.AC_KW.notna() & ~data.DIP & data.IRRADIATION.notna()
              & data[WEATHER_COLS].notna().all(axis=1))
print(f"\nKwartieren overdag bruikbaar (zelfde voor alle modellen): {int(data.OK.sum())}")
NAMES_REF = ["Baseline: gemiddelde dag", "Baseline: persistence (gisteren)", REF_DA, REF_SENSOR]
NAMES = NAMES_REF + list(LADDER_FASE3)


def raw_coef(model):
    """Coëfficiënten per eenheid (voor de stabiliteit van a en b)."""
    if hasattr(model, "steps"):
        sc, est = model[0], model[-1]
        return est.coef_ / sc.scale_
    return model.coef_


def run(train_days, test_days, train_plants=PLANTS, test_plants=PLANTS, cap_from=None, frame=None):
    """Train op (dagen, plants), voorspel (dagen, plants). Capaciteit/schaal alleen uit de training.
    frame: optioneel een tabel met verstoord weer (weerfout-test). Geeft (testtabel, voorspellingen, coëf.)."""
    train_days, test_days = list(train_days), list(test_days)
    src = data if frame is None else frame
    cap = capacity(data[data.DAY.isin(train_days if cap_from is None else cap_from)])
    dd = add_target(src, cap)
    dd.loc[dd.DIP, "FRAC"] = np.nan
    tr = dd[dd.DAY.isin(train_days) & dd.PLANT.isin(train_plants) & dd.OK]
    te = dd[dd.DAY.isin(test_days) & dd.PLANT.isin(test_plants) & dd.OK]
    # referenties: gemiddelde dag, persistence, fase 2 (smart persistence), fase 1 (sensor)
    prof = tr.groupby("SLOT").FRAC.mean() if not set(test_plants) <= set(train_plants) else \
        tr.groupby(["PLANT", "SLOT"]).FRAC.mean()
    keys = list(te.SLOT) if not set(test_plants) <= set(train_plants) else list(zip(te.PLANT, te.SLOT))
    s = dd.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    s = s[~s.index.duplicated()]
    # heldere-hemelschaal per plant uit de trainingsdagen (ook van de testplant: zoals de capaciteit, een huis
    # kent zijn eigen verleden; nodig voor de kruis-plant-test)
    k = add_daily(add_clear(dd, dd[dd.DAY.isin(train_days if cap_from is None else cap_from)]))
    k = dayahead(k).loc[te.index]
    k_fill = float(k.KD.mean()) if k.KD.notna().any() else 0.8
    pr = {"Baseline: gemiddelde dag": np.nan_to_num(prof.reindex(keys).values),
          "Baseline: persistence (gisteren)": s.reindex(list(zip(te.PLANT, te.DAY - 1, te.SLOT))).fillna(0).values,
          REF_DA: (k.KD_1.fillna(k_fill) * k.FRAC_CS).values}
    cols, make, _ = LADDER_FASE1["LR straling"]
    pr[REF_SENSOR] = make().fit(tr[cols], tr.FRAC).predict(te[cols])
    coefs = {}
    for name, (cols, make, _) in LADDER_FASE3.items():
        m = make().fit(tr[cols], tr.FRAC)
        pr[name] = m.predict(te[cols])
        if name.startswith(("e6", "Ridge e6 (")):
            coefs[name] = dict(zip(["a", "b"], raw_coef(m)))
    pr = {n: np.clip(v, 0, None) * (te.COSZ.values > MIN_COSZ) for n, v in pr.items()}
    return te, pr, coefs


# %% 2. Deel A - keuze op de validatievouwen (trainingsdagen)
rows, coef_rows = [], []
for tr_days, va_days in VAL:
    te, pr, coefs = run(tr_days, va_days)
    fold = f"train 1-{max(tr_days)} -> val {va_days[0]}-{va_days[-1]}"
    rows += [{"model": n, "vouw": fold, **metrics(te.FRAC, y, te)} for n, y in pr.items()]
    coef_rows += [{"model": n, "vouw": fold, **c} for n, c in coefs.items()]
val = pd.DataFrame(rows)
tab = val.groupby("model", sort=False)[["nRMSE_%", "nMAE_%", "RMSE_kW", "MAE_kW", "R2", "dagfout_%"]].mean()
tab = tab.join(val.pivot(index="model", columns="vouw", values="nRMSE_%").add_prefix("nRMSE_% ")).loc[NAMES]
for ref, col in (("Baseline: persistence (gisteren)", "skill_pers"), ("Baseline: gemiddelde dag", "skill_gemdag")):
    tab[col] = 1 - tab["nRMSE_%"] / tab.loc[ref, "nRMSE_%"]
CHOICE, WHY = choose(tab, {n: e for n, (_, _, e) in LADDER_FASE3.items()},
                     [c for c in tab.columns if c.startswith("nRMSE_% train")])
coef_tab = pd.DataFrame(coef_rows)
print(f"\n=== Validatie binnen {TR_TXT} (gemiddelde van {val.vouw.nunique()} vouwen) - hier wordt gekozen ===")
print(tab.round(3).to_string())
print("\nCoëfficiënten per vouw (FRAC = a x EFF_T + b x EFF_T x KT_POA):")
print(coef_tab.round(6).to_string(index=False))
print("\nKEUZE:", CHOICE); print("Waarom:", WHY)
val.to_csv(OUT / "validatie_per_vouw.csv", index=False); tab.round(4).to_csv(OUT / "validatie.csv")
coef_tab.to_csv(OUT / "coefficienten_per_vouw.csv", index=False)
summary = {"fase": 3, "run": RUN_NAME, "vraag": "ERA5-weermodel als input (bovengrens voor een echte voorspelling)",
           "opzet_dagen": f"1-{N_SETUP}", "tijdafspraak_min": RAD_SHIFT,
           "tijdafspraak_r": piv.round(4).to_dict(), "keuze": CHOICE, "inputs": LADDER_FASE3[CHOICE][0],
           "motivatie": WHY, "opstelling_plants": "vast, hoek = breedtegraad, zuid (AANNAME uit V3-V5)",
           "validatie": tab.round(4).to_dict("index"), "coefficienten": coef_rows}
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

if not TOON_TEST:
    print(f"\nSTOP: de keuze staat vast. {TE_TXT} is nog NIET bekeken. Na akkoord: fase 3 toevoegen aan "
          "config.TEST_VRIJGEGEVEN.")
    sys.exit(0) if __name__ == "__main__" else None

# %% 3. Deel B - testdagen: A. bevroren, B. walk-forward, per plant
SETUP, TEST = list(range(1, N_SETUP + 1)), list(range(N_SETUP + 1, N_DAYS + 1))
te_f, pr_f, coef_f = run(SETUP, TEST)
rows = []
for plant, m in [("samen", np.ones(len(te_f), bool))] + [(f"Plant {p}", (te_f.PLANT == p).values) for p in PLANTS]:
    rows += [{"model": n, "plant": plant, **metrics(te_f.FRAC[m], y[m], te_f[m])} for n, y in pr_f.items()]
frozen = pd.DataFrame(rows).set_index(["model", "plant"])
for ref, col in (("Baseline: persistence (gisteren)", "skill_pers"), ("Baseline: gemiddelde dag", "skill_gemdag")):
    frozen[col] = 1 - frozen["nRMSE_%"] / frozen.xs(ref)["nRMSE_%"].reindex(frozen.index.get_level_values(1)).values
wt, wp = [], {n: [] for n in NAMES}
for D in TEST:
    t1, p1, _ = run(range(1, D), [D])
    wt.append(t1)
    for n in NAMES:
        wp[n].append(p1[n])
te_w = pd.concat(wt)
walk = pd.DataFrame({n: metrics(te_w.FRAC, np.concatenate(v), te_w) for n, v in wp.items()}).T
print(f"\n=== TEST {TE_TXT}: A. bevroren (samen) + B. walk-forward ===")
print(frozen.xs("samen", level=1)[["nRMSE_%", "nMAE_%", "RMSE_kW", "R2", "dagfout_%", "skill_gemdag"]]
      .join(walk[["nRMSE_%"]], rsuffix=" (B)").loc[NAMES].round(3).to_string())
boot_m, boot_d = bootstrap_days(te_f, pr_f, pairs=[(CHOICE, n) for n in ["LR GHI (ERA5)", REF_DA,
                                                                        "Baseline: gemiddelde dag"] if n != CHOICE])
print(f"\n=== Bootstrap over de {te_f.DAY.nunique()} testdagen (90 %-interval, bevroren) ===")
print(boot_m.loc[NAMES].round(2).to_string()); print(boot_d.round(2).to_string())
boot_m.round(4).to_csv(OUT / "bootstrap_modellen.csv"); boot_d.round(4).to_csv(OUT / "bootstrap_verschillen.csv")
print("\nPer plant (nRMSE %, bevroren):")
print(frozen["nRMSE_%"].unstack("plant").loc[NAMES, ["Plant 1", "Plant 2", "samen"]].round(2).to_string())

# %% 4. Kruis-plant en stabiliteit
rows_x = []
for a, b in [(1, 2), (2, 1)]:
    te, pr, _ = run(SETUP, TEST, train_plants=[a], test_plants=[b], cap_from=SETUP)
    rows_x += [{"richting": f"P{a} -> P{b}", "model": n, **metrics(te.FRAC, y, te)} for n, y in pr.items()]
cross = pd.DataFrame(rows_x).pivot(index="model", columns="richting", values="nRMSE_%").loc[NAMES]
e2 = te_f.assign(**{f"e_{i}": (pr_f[n] - te_f.FRAC) ** 2 for i, n in enumerate(NAMES)})
per_day = pd.DataFrame({n: e2.groupby(["PLANT", "DAY"])[f"e_{i}"].mean() ** 0.5 * 100 for i, n in enumerate(NAMES)})
stab = per_day.agg(["mean", "std"]).T
stab["CV"] = stab["std"] / stab["mean"]
print("\n=== Kruis-plant (nRMSE %) ==="); print(cross.round(2).to_string())
print("\n=== Stabiliteit (nRMSE per plant-dag) ==="); print(stab.round(3).to_string())

# %% 5. Gevoeligheid voor weerfouten (10/20/30 % op de straling, alleen testdagen, 5 herhalingen)
rng = np.random.default_rng(RANDOM_STATE)
test_dates = sorted(data[data.DAY.isin(TEST)].DATE_TIME.dt.normalize().unique())
WMODELS = list(dict.fromkeys([CHOICE, "LR GHI (ERA5)", "e6: EFF_T x (a + b KT_POA), zonder intercept",
                              "Ridge e6 + bewolking, regen, vochtigheid"]))
rows_w = []
for sigma in [0.0, 0.1, 0.2, 0.3]:
    for rep in range(5 if sigma > 0 else 1):
        parts = []
        for p in PLANTS:
            h, elev = era[p]
            hh = h.copy()
            day_idx = hh.DATE_TIME.dt.normalize()
            on = day_idx.isin(test_dates).values
            day_f = pd.Series(1 + rng.normal(0, sigma, len(test_dates)), index=test_dates)
            f = np.ones(len(hh))
            f[on] = np.clip(day_f.reindex(day_idx[on]).values * (1 + rng.normal(0, sigma, on.sum())), 0, None)
            for c in ("shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"):
                hh[c] = hh[c] * f
            w = hourly_to_15min(hh, *PLANT_LOCATIONS[p], elev, rad_shift_min=RAD_SHIFT)
            d = panel_inputs(w, soiling_loss(h, pd.DatetimeIndex(w.DATE_TIME)), *PLANT_LOCATIONS[p], elev)
            parts.append(d[["DATE_TIME"] + WEATHER_COLS].assign(PLANT=p))
        noisy = clean.merge(pd.concat(parts), on=["DATE_TIME", "PLANT"], how="left").assign(OK=data.OK.values)
        noisy.loc[noisy.DAY.isin(SETUP), WEATHER_COLS] = data.loc[data.DAY.isin(SETUP), WEATHER_COLS].values
        te, pr, _ = run(SETUP, TEST, frame=noisy)
        rows_w += [{"fout_%": int(sigma * 100), "herhaling": rep, "model": n, **metrics(te.FRAC, pr[n], te)}
                   for n in WMODELS]
werr = pd.DataFrame(rows_w).groupby(["model", "fout_%"])[["nRMSE_%", "dagfout_%"]].agg(["mean", "std"]).round(3)
print(f"\n=== Gevoeligheid voor weerfouten (bevroren, {TE_TXT}) ==="); print(werr.to_string())

# opslaan
frozen.round(4).to_csv(OUT / "test_bevroren.csv"); walk.round(4).to_csv(OUT / "test_walkforward.csv")
cross.round(4).to_csv(OUT / "kruisplant.csv"); stab.round(4).to_csv(OUT / "stabiliteit.csv")
werr.to_csv(OUT / "weerfout.csv")
summary.update({"test_dagen": f"{TEST[0]}-{TEST[-1]}", "bevroren": frozen.round(4).reset_index().to_dict("records"),
                "walkforward": walk.round(4).to_dict("index"), "kruisplant": cross.round(4).to_dict(),
                "stabiliteit": stab.round(4).to_dict("index"), "coef_bevroren": coef_f,
                "weerfout": {f"{m} | {f}%": v for (m, f), v in
                             pd.DataFrame(rows_w).groupby(["model", "fout_%"])["nRMSE_%"].mean().round(3).items()}})
(OUT / "samenvatting.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

# %% 6. Grafieken: vergelijking, voorbeelddagen, residuen, weerfout
SHORT = {n: n.split(" (")[0] if not n.startswith(("Fase", "Baseline")) else n for n in NAMES}
SHORT.update({"e6: EFF_T x (a + b KT_POA), zonder intercept": "e6: EFF_T x (a + b KT_POA)",
              "Ridge e6 (zonder intercept)": "Ridge e6", "Random Forest (e6 + extra)": "Random Forest",
              "Gradient Boosting (e6 + extra)": "Gradient Boosting"})
v = frozen.xs("samen", level=1).loc[NAMES, "nRMSE_%"][::-1]
fig, ax = plt.subplots(figsize=(9.5, 5))
cols = [BLUE if n == CHOICE else (ORANGE if n.startswith("Fase") else GREY) for n in v.index]
ax.barh(range(len(v)), v.values, color=cols, height=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.2, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(v)), [SHORT[n] for n in v.index]); ax.grid(axis="y", visible=False)
ax.set_xlabel("nRMSE (% van capaciteit) - lager is beter")
ax.set_title(f"Fase 3: ERA5-modellen tegen baselines en fase 1/2 ({TE_TXT})")
save(fig, OUT / "vergelijking.png")

kt = te_f.groupby("DAY").KT_SENSOR.median()
ex = {"helder": kt.idxmax(), "half bewolkt": (kt - kt.median()).abs().idxmin(), "bewolkt": kt.idxmin()}
fig, axs = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
for j, (lab, D) in enumerate(ex.items()):
    for i, p in enumerate(PLANTS):
        m = ((te_f.DAY == D) & (te_f.PLANT == p)).values
        t = te_f.DATE_TIME[m].dt.hour + te_f.DATE_TIME[m].dt.minute / 60
        ax = axs[i, j]
        ax.plot(t, te_f.FRAC[m] * HOUSE_KWP, color=INK, lw=1.4, label="werkelijk")
        ax.plot(t, pr_f[CHOICE][m] * HOUSE_KWP, color=BLUE, lw=2, label=f"{SHORT[CHOICE]} (ERA5)")
        ax.plot(t, pr_f[REF_SENSOR][m] * HOUSE_KWP, color=ORANGE, lw=1.2, ls=":", label="sensor (fase 1)")
        ax.set_title(f"Plant {p} - dag {D} ({lab})")
        if j == 0:
            ax.set_ylabel("vermogen (kW, huis 5 kWp)")
        if i == 1:
            ax.set_xlabel("uur (IST)")
axs[0, 0].legend(fontsize=8, loc="upper left")
save(fig, OUT / "voorbeelddagen.png")

from statsmodels.graphics.tsaplots import plot_acf   # noqa: E402
res = (pr_f[CHOICE] - te_f.FRAC.values) * 100
hour = te_f.DATE_TIME.dt.hour + te_f.DATE_TIME.dt.minute / 60
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for p in PLANTS:
    m = (te_f.PLANT == p).values
    axs[0, 0].scatter(hour[m], res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0, label=f"Plant {p}")
    axs[0, 1].scatter(pr_f[CHOICE][m] * 100, res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0)
    axs[0, 2].scatter(te_f.CLOUD[m], res[m], s=4, alpha=0.3, color=PLANT_COLORS[p], lw=0)
    axs[1, 0].hist(res[m], bins=60, alpha=0.55, color=PLANT_COLORS[p], label=f"Plant {p}")
for ax in (axs[0, 0], axs[0, 1], axs[0, 2]):
    ax.axhline(0, color=INK2, lw=1)
axs[0, 0].set(title="Residu tegen tijd van de dag", xlabel="uur (IST)", ylabel="residu (% van capaciteit)")
axs[0, 0].legend(markerscale=3)
axs[0, 1].set(title="Residu tegen voorspelde waarde", xlabel="voorspeld (% van capaciteit)", ylabel="residu (%)")
axs[0, 2].set(title="Residu tegen bewolking (ERA5)", xlabel="bewolking (%)", ylabel="residu (%)")
axs[1, 0].set(title="Verdeling van de residuen", xlabel="residu (% van capaciteit)", ylabel="aantal kwartieren")
axs[1, 0].legend()
for k, p in enumerate(PLANTS):
    m = (te_f.PLANT == p).values
    plot_acf(res[m], lags=40, ax=axs[1, 1 + k], color=PLANT_COLORS[p], vlines_kwargs={"colors": PLANT_COLORS[p]})
    axs[1, 1 + k].set(title=f"Autocorrelatie residu Plant {p}", xlabel="lag (kwartieren)", ylabel="ACF (-)")
fig.suptitle(f"Residuen van '{SHORT[CHOICE]}' (ERA5, bevroren, {TE_TXT}): voorspeld - werkelijk", x=0.01, ha="left", color=INK)
save(fig, OUT / "residuen.png")

wm = pd.DataFrame(rows_w).groupby(["model", "fout_%"])["nRMSE_%"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(6.8, 3.8))
for n, c in zip(WMODELS, [BLUE, GREY, ORANGE]):
    s = wm[wm.model == n]
    ax.errorbar(s["fout_%"], s["mean"], yerr=s["std"], marker="o", ms=7, lw=2, capsize=3, color=c, label=SHORT[n])
base = frozen.loc[("Baseline: gemiddelde dag", "samen"), "nRMSE_%"]
ax.axhline(base, color=INK2, ls="--", lw=1, label="gemiddelde dag (geen weer)")
ax.set_xticks([0, 10, 20, 30], ["0 %", "10 %", "20 %", "30 %"])
ax.set_xlabel("gesimuleerde fout op de voorspelde straling (sigma, per dag x per uur)")
ax.set_ylabel("nRMSE (%)"); ax.set_title(f"Hoe gevoelig voor fouten in de weersvoorspelling? ({TE_TXT})")
ax.legend(fontsize=8)
save(fig, OUT / "weerfout.png")
plt.show()
print("\nKlaar met fase 3.")

# -*- coding: utf-8 -*-
"""
TAAK 8 - Feedback pitch week 9: helpen deze tips eindmodel v2? (BESLISSINGEN #67-#70, herzien in #75)
=====================================================================================================
Twee datasets, zelfde varianten, zelfde regel:
  UK PV   40 huizen, dag-vooruit (ECMWF dag 1). Training 6 mrt - 30 jun 2024, test 1 jul 2024 - 31 mrt 2025.
  HKUST   5 stations Hong Kong, ERA5 (weer achteraf). Training 2021-2022, test 2023 (alle seizoenen in de training).
Referentie = eindmodel v2 (pvlib x f uit 4 weken, gemengd met de gemiddelde dag). Elke variant verandert ÉÉN ding:
  T1  decision tree i.p.v. lineaire regressie (heeft training nodig, geen cold start)
  T2  decision tree als keuzeschakelaar: per tijdstap v2 (pvlib x f) of de gemiddelde dag van het huis
  T3  v2 + decision tree op de restfout
  F1  kalibratievenster f: 1, 2, 4 (= v2), 8 weken of alle eerdere data (compleet verleden)
  F2  extra features op v2: gemiddelde/std van de dagelijkse helderheid (laatste 7 dagen of alles)
  S1  stream: f elke DAG bijwerken (laatste 28 dagen) i.p.v. elke week
  S2  stream met exponentieel vergeten (halfwaardetijd 7/14/28 dagen)
  B1  Bayesiaans: f als 'local level' (random walk) met Kalman-filter, elke dag bijgewerkt. Start bij een nieuw
      huis met f = 1 en een brede onzekerheid (cold start). Dit is de kern van Orbit's ETS/DLT-niveau, zonder Stan.
Regel (#67): overnemen als test >= 0,3 pp lager dan v2 en het 90 %-bootstrapinterval van het verschil < 0.

Herzien na de code-review (#75):
  - twee scenario's voor wanneer de dagopbrengst bekend is (config TAAK8_SCENARIOS): 'avond' (voorspelling om
    21:00, vandaag bekend) en 'strikt' (24 uur vooruit, vandaag nog niet af). Alle varianten, ook de wekelijkse
    v2-referentie, gebruiken alleen dagen tot en met (doeldag - lag);
  - keuzes binnen een familie (F1-venster, S2-halfwaardetijd, B1-q, boomdiepte) alleen op de validatie;
  - menggewicht en v2-feature in de validatie alleen uit data vóór de validatieperiode;
  - opvulwaarden alleen uit de training; random_state uit config; perioden in samenvatting.json;
  - cache zonder kalibratie (die wordt elke run opnieuw berekend).
Runnen: F5 in Spyder of VS Code (Anaconda). Eerst de UK- en HKUST-scripts (cache). Argument 'hkust' = HKUST.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import (UKPV_RESULTS, UKPV_FC_START, UKPV_TRAIN_END, UKPV_CLEAN, UKPV_FOLDS,   # noqa: E402
                              HKUST_RESULTS, HKUST_CLEAN, HKUST_YEARS, CAPACITY_QUANTILE, MIN_COSZ,
                              RANDOM_STATE, TAAK8_RULE_PP, TAAK8_SCENARIOS)
from solarcore import model_v2 as v2   # noqa: E402
from solarcore.evaluate import bootstrap_days   # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
OUT = ROOT / "resultaten" / "taak8"
X_T1 = ["POA_F1", "KT_W_F1", "COSZ", "CLOUD_F1", "T_F1"]
X_SEL = ["KT_W_F1", "CLOUD_F1", "COSZ", "SLOT"]
X_RES = ["V2F", "KT_W_F1", "CLOUD_F1", "COSZ", "SLOT"]
COLS = ["PLANT", "DATE_TIME", "SLOT", "P_KW", "KWP", "CAP", "FRAC", "PV_KW", "CS_GHI", "COSZ",
        "POA_F1", "KT_W_F1", "CLOUD_F1", "T_F1", "f", "c", "K", "n"]


def end_of(day):
    return pd.Timestamp(day) + pd.Timedelta("23:59:59")


def add_v2_stats(data, lag):
    """Wekelijkse v2-kalibratie (f, c, K uit de 4 weken ervoor). De week waarmee gekalibreerd wordt is die van
    (doeldag - (lag - 1)): bij lag 1 telt de dag van gisteren mee, bij lag 2 niet (zie TAAK8_SCENARIOS)."""
    data = data.copy()
    data["WEEK"] = data.DATE_TIME.dt.to_period("W-SUN")
    data["CALWEEK"] = (data.DATE_TIME.dt.normalize() - pd.Timedelta(days=lag - 1)).dt.to_period("W-SUN")
    stats = v2.rolling_stats(data[["PLANT", "WEEK", "P_KW", "PV_KW", "CS_GHI", "COSZ"]])
    return (data.merge(stats.rename(columns={"WEEK": "CALWEEK"}), on=["PLANT", "CALWEEK"], how="left")[COLS]
            .reset_index(drop=True))


def kalman_f(y, ok, q, r, p0=0.3 ** 2):
    """Local level op log f: x_D = x_(D-1) + ruis(q), y_D = x_D + ruis(r). Geeft f voor dag D uit dagen < D."""
    x, p, out = 0.0, p0, np.empty(len(y))
    for i in range(len(y)):
        out[i] = x
        p += q
        if ok[i]:
            k = p / (p + r)
            x += k * (y[i] - x)
            p *= 1 - k
    return np.exp(out)


def daily_f(d, trm):
    """Per huis en dag: f-varianten uit dagtotalen van de dagen VÓÓR die dag (venster, exponentieel, Kalman).
    Opzoeken met een latere datum geeft de lag (zie evaluate). r van het Kalman-filter alleen uit de training."""
    day = d.groupby(["PLANT", "DATE"])[["P_KW", "PV_KW"]].sum()
    day["N"] = d.groupby(["PLANT", "DATE"]).size()
    cal = []
    for sid, g in day.groupby(level=0):
        g = g.droplevel(0)
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"), fill_value=0.0)
        g.index.name = "DATE"
        for c in ("P_KW", "PV_KW", "N"):
            for nw in (1, 2, 4, 8):                                 # som over [D - 7N, D - 1]
                g[f"{c}_{nw}w"] = g[c].shift(1).rolling(7 * nw, min_periods=1).sum()
            g[f"{c}_all"] = g[c].shift(1).cumsum()
            for hl in (7, 14, 28):                                  # exponentieel vergeten, dagen vóór D
                a, s, o = 0.5 ** (1 / hl), 0.0, []
                for v in g[c].values:
                    o.append(s)
                    s = a * (s + v)
                g[f"{c}_exp{hl}"] = o
        okd = (g.PV_KW > 0) & (g.P_KW > 0) & (g.N >= 8)
        g["LOGY"] = np.log(np.where(okd, g.P_KW / g.PV_KW.where(g.PV_KW > 0, 1), 1.0))
        g["OKD"] = okd
        g["PLANT"] = sid
        cal.append(g.reset_index())
    cal = pd.concat(cal, ignore_index=True)
    fk = cal[["PLANT", "DATE"]].copy()
    for key in ("1w", "2w", "4w", "8w", "all", "exp7", "exp14", "exp28"):
        pv, p, n = cal[f"PV_KW_{key}"], cal[f"P_KW_{key}"], cal[f"N_{key}"]
        fk[key] = np.where((pv > 0) & (n > 0), p / pv.where(pv > 0, 1), 1.0)
    # B1: ruis r uit de training (dag-op-dag sprong van log f ~ 2r), q als fractie van r gekozen op de validatie
    tr_days = cal.DATE.between(d.DATE[trm].min(), d.DATE[trm].max()) & cal.OKD
    r = float(cal[tr_days].groupby("PLANT").LOGY.apply(lambda s: s.diff().var()).median() / 2)
    for ratio in (0.01, 0.03, 0.1, 0.3):
        fk[f"b{ratio}"] = np.concatenate([kalman_f(g.LOGY.values, g.OKD.values, ratio * r, r)
                                          for _, g in cal.groupby("PLANT", sort=False)])
    return fk, r


def evaluate(d, start, trm, te, folds, valm, out, lag, meta):
    """Alle varianten op één dataset en één scenario (lag). trm/te/valm: maskers (training, test, validatie)."""
    out.mkdir(parents=True, exist_ok=True)
    d = d.copy()
    d["DATE"] = d.DATE_TIME.dt.normalize()
    d["CALDATE"] = d.DATE - pd.Timedelta(days=lag - 1)          # laatste bekende dag = CALDATE - 1 = doel - lag
    d["WSTART"] = d.CALDATE - pd.to_timedelta(d.CALDATE.dt.dayofweek, unit="D")
    d["MEANDAY"] = d.K * d.c * d.CS_GHI
    val_fit = trm.values & (d.DATE_TIME < d.DATE_TIME[valm].min()).values   # vóór de validatie: voor gewichten

    def frac(pred_kw):
        return np.clip(np.nan_to_num(np.asarray(pred_kw, float)), 0, None) / d.CAP.values

    def blend(f, fitm):
        """v2-mengen met gewicht w, geschat op de rijen van fitm."""
        raw, md = f * d.PV_KW.values, d.MEANDAY.values
        ok = np.isfinite(md)
        m = np.asarray(fitm) & (d.n.values > 0) & ok
        w = v2.mix_weight(raw[m], md[m], d.P_KW.values[m])
        return frac(np.where(ok, w * raw + (1 - w) * md, raw)), w

    def nrmse(p, m):
        m = np.asarray(m)
        return float(np.sqrt(np.mean((p[m] - d.FRAC.values[m]) ** 2)) * 100)

    f_ref = d.f.fillna(1.0).values
    PRED = {}
    PRED["v2 (referentie)"], w0 = blend(f_ref, trm)

    # kalibratie-varianten (geen fit): keuze binnen de familie op de validatie
    fk, r = daily_f(d, trm)
    wk = d[["PLANT", "WSTART"]].merge(fk.rename(columns={"DATE": "WSTART"}), on=["PLANT", "WSTART"], how="left")
    dy = d[["PLANT", "CALDATE"]].merge(fk.rename(columns={"DATE": "CALDATE"}), on=["PLANT", "CALDATE"], how="left")
    fams = {"F1": {f"F1: f uit {lab}": wk[key].fillna(1.0).values for key, lab in
                   (("1w", "1 week"), ("2w", "2 weken"), ("4w", "4 weken (= v2)"), ("8w", "8 weken"),
                    ("all", "alle eerdere data (compleet verleden)"))},
            "S2": {f"S2: elke dag, exponentieel vergeten (halfwaardetijd {hl} d)": dy[f"exp{hl}"].fillna(1.0).values
                   for hl in (7, 14, 28)},
            "B1": {f"B1: Bayesiaans (Kalman, q = {q} r), elke dag": dy[f"b{q}"].fillna(1.0).values
                   for q in (0.01, 0.03, 0.1, 0.3)}}
    val_cal, chosen = {}, {}
    for fam, opts in fams.items():
        sc = {lab: nrmse(blend(f, val_fit)[0], valm) for lab, f in opts.items()}
        val_cal.update(sc)
        chosen[fam] = min(sc, key=sc.get)
        PRED[chosen[fam]] = blend(opts[chosen[fam]], trm)[0]
    PRED["F1: f uit 4 weken (= v2, controle)"] = blend(fams["F1"]["F1: f uit 4 weken (= v2)"], trm)[0]
    PRED["S1: f elke dag bijwerken (28 dagen)"] = blend(dy["4w"].fillna(1.0).values, trm)[0]
    val_cal["S1: f elke dag bijwerken (28 dagen)"] = nrmse(blend(dy["4w"].fillna(1.0).values, val_fit)[0], valm)
    val_cal["v2 (referentie)"] = nrmse(blend(f_ref, val_fit)[0], valm)

    # varianten met training: features
    d["RAWF"] = frac(f_ref * d.PV_KW)
    d["MDF"] = frac(np.where(np.isfinite(d.MEANDAY), d.MEANDAY, f_ref * d.PV_KW))
    kd = d.groupby(["PLANT", "DATE"]).apply(lambda g: g.P_KW.sum() / max((g.CS_GHI * g.KWP / 1000).sum(), 1e-6)).rename("KD")
    kd = kd.reset_index().sort_values(["PLANT", "DATE"])
    gk = kd.groupby("PLANT").KD
    kd["KM7"] = gk.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())   # dagen vóór DATE
    kd["KS7"] = gk.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    kd["KMALL"] = gk.transform(lambda s: s.shift(1).expanding(3).mean())
    kd["KSALL"] = gk.transform(lambda s: s.shift(1).expanding(3).std())
    d = d.merge(kd.drop(columns="KD").rename(columns={"DATE": "CALDATE"}), on=["PLANT", "CALDATE"], how="left")
    for c in ("KM7", "KS7", "KMALL", "KSALL"):
        d[c] = d[c].fillna(d.loc[trm.values, c].median())      # opvulwaarde alleen uit de training

    def set_v2f(fitm):
        """v2-feature (en de afgeleide kolommen) met het menggewicht uit fitm."""
        d["V2F"] = blend(f_ref, fitm)[0]
        for c in ("KM7", "KS7", "KMALL", "KSALL"):
            d[f"V2x{c}"] = d.V2F * d[c]
        d["V2xKT"] = d.V2F * d.KT_W_F1

    cand = {
        **{f"T1: decision tree i.p.v. LR (diepte {k})": (lambda k=k: DecisionTreeRegressor(max_depth=k, min_samples_leaf=50,
                                                             random_state=RANDOM_STATE), X_T1, "reg") for k in (3, 5, 8, 12)},
        **{f"T2: tree kiest v2 of gemiddelde dag (diepte {k})": (lambda k=k: DecisionTreeClassifier(max_depth=k,
                                                             min_samples_leaf=200, random_state=RANDOM_STATE), X_SEL, "sel")
           for k in (2, 3, 4, 6)},
        **{f"T3: v2 + tree op de restfout (diepte {k})": (lambda k=k: DecisionTreeRegressor(max_depth=k, min_samples_leaf=200,
                                                             random_state=RANDOM_STATE), X_RES, "res") for k in (2, 3, 5, 8)},
        "F2: LR op alleen v2 (controle)": (LinearRegression, ["V2F"], "reg"),
        "F2: LR op v2 + KT + gem./std helderheid 7 d (recent)": (LinearRegression, ["V2F", "V2xKT", "V2xKM7", "V2xKS7"], "reg"),
        "F2: LR op v2 + KT + gem./std helderheid alle dagen (compleet)": (LinearRegression, ["V2F", "V2xKT", "V2xKMALL", "V2xKSALL"], "reg"),
    }

    def fit_predict(make, cols, kind, tm, am):
        t = d[tm & (d.n > 0)] if kind != "reg" or "V2F" in cols else d[tm]
        m = make()
        if kind == "reg":
            m.fit(t[cols], t.FRAC)
            return np.clip(m.predict(d.loc[am, cols]), 0, None)
        if kind == "sel":
            m.fit(t[cols], (np.abs(t.RAWF - t.FRAC) < np.abs(t.MDF - t.FRAC)).astype(int))
            return np.where(m.predict(d.loc[am, cols]) == 1, d.loc[am, "RAWF"], d.loc[am, "MDF"])
        m.fit(t[cols], t.FRAC - t.V2F)
        return np.clip(d.loc[am, "V2F"].values + m.predict(d.loc[am, cols]), 0, None)

    val = {}
    for tr0, tr_end, v0, v1 in folds:
        tm = d.DATE_TIME.between(pd.Timestamp(tr0), end_of(tr_end))
        vm = d.DATE_TIME.between(pd.Timestamp(v0), end_of(v1))
        set_v2f(tm)                                             # menggewicht alleen uit de vouw-training
        val.setdefault("v2 (referentie)", []).append(nrmse(d.V2F.values, vm))
        for name, (make, cols, kind) in cand.items():
            val.setdefault(name, []).append(nrmse(_fill(fit_predict(make, cols, kind, tm, vm), vm), vm))
    vtab = pd.DataFrame(val, index=[f"vouw {i + 1}" for i in range(len(folds))]).T
    vtab["gemiddeld"] = vtab.mean(axis=1)
    print("\n=== Validatie bomen/F2, nRMSE % ==="); print(vtab.round(2).to_string())
    print("\n=== Validatie kalibratie-varianten, nRMSE % ===")
    print(pd.Series(val_cal).round(2).to_string())
    for fam in ("T1", "T2", "T3"):
        chosen[fam] = vtab.loc[[n for n in vtab.index if n.startswith(fam)], "gemiddeld"].idxmin()
    set_v2f(trm)
    for n in [chosen[f] for f in ("T1", "T2", "T3")] + [n for n in vtab.index if n.startswith("F2")]:
        make, cols, kind = cand[n]
        PRED[n] = _fill(fit_predict(make, cols, kind, trm, te), te)

    tdf = d[te.values].copy()
    rows = {}
    for n, p in PRED.items():
        e2 = (p[te.values] - tdf.FRAC.values) ** 2
        rows[n] = {"nRMSE_%": float(np.sqrt(e2.mean()) * 100),
                   "std_over_dagen_pp": float((np.sqrt(pd.Series(e2).groupby(tdf.DATE.values).mean()) * 100).std()),
                   "std_over_huizen_pp": float((np.sqrt(pd.Series(e2).groupby(tdf.PLANT.values).mean()) * 100).std()),
                   "training_nodig": n.startswith(("T", "F2"))}
    tab = pd.DataFrame(rows).T
    tdf["DAY"] = (tdf.DATE - tdf.DATE.min()).dt.days                     # bootstrap over dagen
    _, diffs = bootstrap_days(tdf, {n: p[te.values] for n, p in PRED.items()},
                              pairs=[(n, "v2 (referentie)") for n in PRED if n != "v2 (referentie)"])
    diffs.index = [i.replace(" - v2 (referentie)", "") for i in diffs.index]
    tab = tab.join(diffs)
    tab["overnemen"] = (tab.verschil_pp <= -TAAK8_RULE_PP) & (tab.hoog_pp < 0)
    print(f"\n=== Test {meta['dataset']} / {meta['scenario']}, nRMSE % ==="); print(tab.round(2).to_string())
    vtab.round(3).to_csv(out / "validatie.csv")
    pd.Series(val_cal, name="nRMSE_%").round(3).to_csv(out / "validatie_kalibratie.csv")
    tab.round(3).to_csv(out / "test.csv")
    (out / "samenvatting.json").write_text(json.dumps({
        **meta, "lag_dagen": lag, "gekozen_op_validatie": chosen, "menggewicht_v2": w0, "bayes_r": r,
        "regel_pp": TAAK8_RULE_PP, "training": [str(d.DATE_TIME[trm].min()), str(d.DATE_TIME[trm].max())],
        "validatie": [str(d.DATE_TIME[valm].min()), str(d.DATE_TIME[valm].max())],
        "vouwen": [list(f) for f in folds], "test": [str(tdf.DATE_TIME.min()), str(tdf.DATE_TIME.max())]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return tab


def _fill(p, mask):
    """Voorspelling op de rijen van mask -> array over alle rijen (rest NaN)."""
    a = np.full(len(mask), np.nan)
    a[np.asarray(mask)] = p
    return a


def run_all(base, dataset, start, trm, te, folds, valm):
    """Beide scenario's (TAAK8_SCENARIOS) op één dataset."""
    for scen, lag in TAAK8_SCENARIOS.items():
        d = add_v2_stats(base, lag)
        evaluate(d, start, trm, te, folds, valm, OUT / dataset / scen, lag, {"dataset": dataset, "scenario": scen})


# %% 1. UK PV (dag-vooruit, echte voorspellingen)
def uk_frame():
    """Basistabel UK (zonder v2-kalibratie). Cache per tijdverschuiving en einde training (bepaalt CAP)."""
    from solarcore.ukpv import chosen_systems, model_frame
    shift = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))["tijdverschuiving_min"]
    cache = UKPV_CLEAN / f"taak8_base_shift{shift}_tr{UKPV_TRAIN_END}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    data = model_frame(shift, chosen_systems())
    tr = data.DATE_TIME.between(pd.Timestamp(UKPV_FC_START), end_of(UKPV_TRAIN_END))
    data["CAP"] = data.PLANT.map(data[tr & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE))
    data["FRAC"] = data.P_KW / data.CAP
    data["PV_KW"] = data.PHYS_F1 * data.KWP
    base = data[data.OK][[c for c in COLS if c not in ("f", "c", "K", "n")]].reset_index(drop=True)
    base.to_parquet(cache, index=False)
    return base


if __name__ == "__main__" and "hkust" not in sys.argv:
    base = uk_frame()
    start = pd.Timestamp(UKPV_FC_START)
    trm = base.DATE_TIME.between(start, end_of(UKPV_TRAIN_END))
    te = base.DATE_TIME > end_of(UKPV_TRAIN_END)
    valm = base.DATE_TIME.between(pd.Timestamp(UKPV_FOLDS[0][2]), end_of(UKPV_FOLDS[-1][3]))
    print(f"UK: {len(base):,} halfuren overdag, {base.PLANT.nunique()} huizen")
    run_all(base, "uk", start, trm, te, UKPV_FOLDS, valm)


# %% 2. HKUST (Hong Kong, 3 jaar, ERA5 = weer achteraf)
def hk_frame():
    """Basistabel HKUST (zonder v2-kalibratie). Cache per tijdverschuiving en testjaar (bepaalt CAP)."""
    from solarcore.hkust import load_clean, load_era5_hk, add_era5, capacity, pvlib_ac_kw
    from solarcore.physics import PARAMS
    shift = json.loads((HKUST_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))["tijdverschuiving_min"]
    test_start = f"{HKUST_YEARS[2]}-01-01"
    cache = HKUST_CLEAN / f"taak8_base_shift{shift}_test{HKUST_YEARS[2]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    h, elev = load_era5_hk()
    data = add_era5(load_clean(), h, elev, shift)
    data = data[data.OK & (data.COSZ > MIN_COSZ)].copy()
    meta = data.groupby("PLANT")[["KWP_AC", "KWP_DC"]].first()
    data["CAP"] = data.PLANT.map(capacity(data[data.DATE_TIME < test_start]))
    data["FRAC"] = data.P_KW / data.CAP
    data["KWP"] = data.PLANT.map(meta.KWP_DC)
    data["PV_KW"] = pvlib_ac_kw(data, data.KWP.values, data.PLANT.map(meta.KWP_AC).values, PARAMS)
    data = data.rename(columns={"POA": "POA_F1", "KT_W": "KT_W_F1", "CLOUD": "CLOUD_F1", "T": "T_F1"})
    base = data[[c for c in COLS if c not in ("f", "c", "K", "n")]].reset_index(drop=True)
    base.to_parquet(cache, index=False)
    return base


if __name__ == "__main__" and "hkust" in sys.argv:
    base = hk_frame()
    y0, y1, y2 = HKUST_YEARS                                    # train 2021-2022 (vouw: 2021 -> 2022), test 2023
    start = base.DATE_TIME.min()
    trm = base.DATE_TIME < f"{y2}-01-01"
    te = base.DATE_TIME >= f"{y2}-01-01"
    folds = ((str(start.date()), f"{y0}-12-31", f"{y1}-01-01", f"{y1}-12-31"),)
    valm = base.DATE_TIME.between(f"{y1}-01-01", end_of(f"{y1}-12-31"))
    print(f"HKUST: {len(base):,} kwartieren overdag, {base.PLANT.nunique()} stations")
    run_all(base, "hkust", start, trm, te, folds, valm)

# -*- coding: utf-8 -*-
"""
TAAK 8 - Feedback pitch week 9: helpen deze tips eindmodel v2? (BESLISSINGEN #67-#70)
=====================================================================================
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
Runnen: F5 in Spyder of VS Code (Anaconda). Eerst de UK- en HKUST-scripts (cache) en eindmodel_v2.py.
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
from solarcore.config import (UKPV_RESULTS, UKPV_FC_START, UKPV_TRAIN_END, UKPV_CLEAN,   # noqa: E402
                              HKUST_RESULTS, CAPACITY_QUANTILE, MIN_COSZ)
from solarcore import model_v2 as v2   # noqa: E402
from solarcore.evaluate import bootstrap_days   # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
RULE_PP = 0.3
X_T1 = ["POA_F1", "KT_W_F1", "COSZ", "CLOUD_F1", "T_F1"]
X_SEL = ["KT_W_F1", "CLOUD_F1", "COSZ", "SLOT"]
X_RES = ["V2F", "KT_W_F1", "CLOUD_F1", "COSZ", "SLOT"]
COLS = ["PLANT", "DATE_TIME", "SLOT", "P_KW", "KWP", "CAP", "FRAC", "PV_KW", "CS_GHI", "COSZ",
        "POA_F1", "KT_W_F1", "CLOUD_F1", "T_F1", "f", "c", "K", "n"]


def add_v2_stats(data):
    """Wekelijkse v2-kalibratie (f, c, K uit de 4 weken ervoor) op rijen overdag met meting."""
    data = data.copy()
    data["WEEK"] = data.DATE_TIME.dt.to_period("W-SUN")
    stats = v2.rolling_stats(data[["PLANT", "WEEK", "P_KW", "PV_KW", "CS_GHI", "COSZ"]])
    return data.merge(stats, on=["PLANT", "WEEK"], how="left")[COLS].reset_index(drop=True)


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


def evaluate(d, start, trm, te, folds, valm, out):
    """Alle varianten op één dataset. trm/te/valm: maskers (training, test, validatie voor varianten zonder fit)."""
    out.mkdir(parents=True, exist_ok=True)
    d = d.copy()
    d["DATE"] = d.DATE_TIME.dt.normalize()
    d["WSTART"] = d.DATE - pd.to_timedelta(d.DATE.dt.dayofweek, unit="D")
    d["MEANDAY"] = d.K * d.c * d.CS_GHI

    def frac(pred_kw):
        return np.clip(np.nan_to_num(np.asarray(pred_kw, float)), 0, None) / d.CAP.values

    def blend(f):
        raw, md = f * d.PV_KW.values, d.MEANDAY.values
        ok = np.isfinite(md)
        m = trm.values & (d.n.values > 0) & ok
        w = v2.mix_weight(raw[m], md[m], d.P_KW.values[m])
        return frac(np.where(ok, w * raw + (1 - w) * md, raw)), w

    def nrmse(p, m):
        m = np.asarray(m)
        return float(np.sqrt(np.mean((p[m] - d.FRAC.values[m]) ** 2)) * 100)

    PRED = {}
    PRED["v2 (referentie)"], w0 = blend(d.f.fillna(1.0).values)

    # dagtotalen per huis -> f-varianten
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
    wk = d[["PLANT", "WSTART"]].merge(fk.rename(columns={"DATE": "WSTART"}), on=["PLANT", "WSTART"], how="left")
    dy = d[["PLANT", "DATE"]].merge(fk, on=["PLANT", "DATE"], how="left")
    for key, lab in (("1w", "F1: f uit 1 week"), ("2w", "F1: f uit 2 weken"), ("4w", "F1: f uit 4 weken (= v2, controle)"),
                     ("8w", "F1: f uit 8 weken"), ("all", "F1: f uit alle eerdere data (compleet verleden)")):
        PRED[lab], _ = blend(wk[key].fillna(1.0).values)
    PRED["S1: f elke dag bijwerken (28 dagen)"], _ = blend(dy["4w"].fillna(1.0).values)
    for hl in (7, 14, 28):
        PRED[f"S2: elke dag, exponentieel vergeten (halfwaardetijd {hl} d)"], _ = blend(dy[f"exp{hl}"].fillna(1.0).values)
    bval = {}
    for ratio in (0.01, 0.03, 0.1, 0.3):
        p, _ = blend(dy[f"b{ratio}"].fillna(1.0).values)
        bval[ratio] = (nrmse(p, valm), p)
    best_ratio = min(bval, key=lambda k: bval[k][0])
    PRED[f"B1: Bayesiaans (Kalman, q = {best_ratio} r), elke dag"] = bval[best_ratio][1]

    # varianten met training
    d["V2F"] = PRED["v2 (referentie)"]
    d["RAWF"] = frac(d.f.fillna(1.0) * d.PV_KW)
    d["MDF"] = frac(np.where(np.isfinite(d.MEANDAY), d.MEANDAY, d.f.fillna(1.0) * d.PV_KW))
    kd = d.groupby(["PLANT", "DATE"]).apply(lambda g: g.P_KW.sum() / max((g.CS_GHI * g.KWP / 1000).sum(), 1e-6)).rename("KD")
    kd = kd.reset_index().sort_values(["PLANT", "DATE"])
    gk = kd.groupby("PLANT").KD
    kd["KM7"] = gk.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    kd["KS7"] = gk.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    kd["KMALL"] = gk.transform(lambda s: s.shift(1).expanding(3).mean())
    kd["KSALL"] = gk.transform(lambda s: s.shift(1).expanding(3).std())
    d = d.merge(kd.drop(columns="KD"), on=["PLANT", "DATE"], how="left")
    for c in ("KM7", "KS7", "KMALL", "KSALL"):
        d[c] = d[c].fillna(d[c].median())
        d[f"V2x{c}"] = d.V2F * d[c]
    d["V2xKT"] = d.V2F * d.KT_W_F1
    cand = {
        **{f"T1: decision tree i.p.v. LR (diepte {k})": (lambda k=k: DecisionTreeRegressor(max_depth=k, min_samples_leaf=50,
                                                             random_state=0), X_T1, "reg") for k in (3, 5, 8, 12)},
        **{f"T2: tree kiest v2 of gemiddelde dag (diepte {k})": (lambda k=k: DecisionTreeClassifier(max_depth=k,
                                                             min_samples_leaf=200, random_state=0), X_SEL, "sel") for k in (2, 3, 4, 6)},
        **{f"T3: v2 + tree op de restfout (diepte {k})": (lambda k=k: DecisionTreeRegressor(max_depth=k, min_samples_leaf=200,
                                                             random_state=0), X_RES, "res") for k in (2, 3, 5, 8)},
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
    for name, (make, cols, kind) in cand.items():
        sc = []
        for tr_end, v0, v1 in folds:
            tm = d.DATE_TIME.between(start, pd.Timestamp(tr_end) + pd.Timedelta("23:59:59"))
            vm = d.DATE_TIME.between(pd.Timestamp(v0), pd.Timestamp(v1) + pd.Timedelta("23:59:59"))
            sc.append(nrmse(_fill(fit_predict(make, cols, kind, tm, vm), vm), vm))
        val[name] = sc
    for tr_end, v0, v1 in folds:
        vm = d.DATE_TIME.between(pd.Timestamp(v0), pd.Timestamp(v1) + pd.Timedelta("23:59:59"))
        val.setdefault("v2 (referentie)", []).append(nrmse(d.V2F.values, vm))
    vtab = pd.DataFrame(val, index=[f"vouw {i + 1}" for i in range(len(folds))]).T
    vtab["gemiddeld"] = vtab.mean(axis=1)
    print("\n=== Validatie, nRMSE % ==="); print(vtab.round(2).to_string())
    chosen = {fam: vtab.loc[[n for n in vtab.index if n.startswith(fam)], "gemiddeld"].idxmin() for fam in ("T1", "T2", "T3")}
    for n in list(chosen.values()) + [n for n in vtab.index if n.startswith("F2")]:
        make, cols, kind = cand[n]
        PRED[n] = _fill(fit_predict(make, cols, kind, trm, te), te)

    tdf = d[te].copy()
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
    tab["overnemen"] = (tab.verschil_pp <= -RULE_PP) & (tab.hoog_pp < 0)
    print("\n=== Test, nRMSE % ==="); print(tab.round(2).to_string())
    vtab.round(3).to_csv(out / "validatie.csv")
    tab.round(3).to_csv(out / "test.csv")
    (out / "samenvatting.json").write_text(json.dumps({"gekozen_op_vouwen": chosen, "menggewicht_v2": w0,
                                                       "bayes_r": r, "bayes_q_ratio": best_ratio,
                                                       "bayes_validatie": {str(k): v[0] for k, v in bval.items()},
                                                       "regel_pp": RULE_PP}, indent=2, ensure_ascii=False), encoding="utf-8")
    return tab


def _fill(p, mask):
    """Voorspelling op de rijen van mask -> array over alle rijen (rest NaN)."""
    a = np.full(len(mask), np.nan)
    a[np.asarray(mask)] = p
    return a


# %% 1. UK PV (dag-vooruit, echte voorspellingen)
def uk_frame():
    from solarcore.ukpv import chosen_systems, model_frame
    shift = json.loads((UKPV_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))["tijdverschuiving_min"]
    cache = UKPV_CLEAN / f"taak8_base_shift{shift}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    data = model_frame(shift, chosen_systems())
    tr = data.DATE_TIME.between(pd.Timestamp(UKPV_FC_START), pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59"))
    data["CAP"] = data.PLANT.map(data[tr & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE))
    data["FRAC"] = data.P_KW / data.CAP
    data["PV_KW"] = data.PHYS_F1 * data.KWP
    d = add_v2_stats(data[data.OK])
    d.to_parquet(cache, index=False)
    return d


if __name__ == "__main__" and "hkust" not in sys.argv:
    d = uk_frame()
    start = pd.Timestamp(UKPV_FC_START)
    trm = d.DATE_TIME.between(start, pd.Timestamp(UKPV_TRAIN_END) + pd.Timedelta("23:59:59"))
    print(f"UK: {len(d):,} halfuren overdag, {d.PLANT.nunique()} huizen")
    evaluate(d, start, trm, d.DATE_TIME >= "2024-07-01",
             [("2024-04-30", "2024-05-01", "2024-05-31"), ("2024-05-31", "2024-06-01", "2024-06-30")],
             d.DATE_TIME.between("2024-05-01", "2024-06-30 23:59"), ROOT / "resultaten" / "taak8" / "uk")


# %% 2. HKUST (Hong Kong, 3 jaar, ERA5 = weer achteraf)
def hk_frame():
    from solarcore.hkust import load_clean, load_era5_hk, add_era5, capacity, pvlib_ac_kw
    from solarcore.physics import PARAMS
    cache = ROOT / "data" / "extern" / "hkust" / "clean" / "taak8_base.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    shift = json.loads((HKUST_RESULTS / "stap3_tests" / "samenvatting.json").read_text(encoding="utf-8"))["tijdverschuiving_min"]
    h, elev = load_era5_hk()
    data = add_era5(load_clean(), h, elev, shift)
    data = data[data.OK & (data.COSZ > MIN_COSZ)].copy()
    meta = data.groupby("PLANT")[["KWP_AC", "KWP_DC"]].first()
    data["CAP"] = data.PLANT.map(capacity(data[data.DATE_TIME < "2023-01-01"]))
    data["FRAC"] = data.P_KW / data.CAP
    data["KWP"] = data.PLANT.map(meta.KWP_DC)
    data["PV_KW"] = pvlib_ac_kw(data, data.KWP.values, data.PLANT.map(meta.KWP_AC).values, PARAMS)
    data = data.rename(columns={"POA": "POA_F1", "KT_W": "KT_W_F1", "CLOUD": "CLOUD_F1", "T": "T_F1"})
    d = add_v2_stats(data)
    d.to_parquet(cache, index=False)
    return d


if __name__ == "__main__" and "hkust" in sys.argv:
    d = hk_frame()
    start = d.DATE_TIME.min()
    print(f"HKUST: {len(d):,} kwartieren overdag, {d.PLANT.nunique()} stations")
    evaluate(d, start, d.DATE_TIME < "2023-01-01", d.DATE_TIME >= "2023-01-01",
             [("2021-12-31", "2022-01-01", "2022-12-31")],
             d.DATE_TIME.between("2022-01-01", "2022-12-31 23:59"), ROOT / "resultaten" / "taak8" / "hkust")

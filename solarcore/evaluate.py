# -*- coding: utf-8 -*-
"""
Metrics (zelfde definities als V5, AGENTS.md §7) en baselines
=============================================================
Alles alleen OVERDAG (cos zenit > 0,02). FRAC 1 = capaciteit.
  nRMSE, nMAE (%)       RMSE/MAE van FRAC x 100
  RMSE_kW, MAE_kW       voor een voorbeeldhuis van 5 kWp
  R2
  dagfout_%             gemiddelde |voorspelde - echte dagopbrengst| / gemiddelde echte dagopbrengst
  skill                 1 - nRMSE_model / nRMSE_referentie (in de tabel berekend)
Geen MAPE (ontploft bij opwek ~ 0).
"""
import numpy as np
import pandas as pd

from .config import HOUSE_KWP, MIN_COSZ


def daytime(df):
    return df[(df.COSZ > MIN_COSZ) & df.FRAC.notna()]


def metrics(y, yhat, meta):
    """y, yhat: FRAC overdag; meta: tabel met PLANT en DAY (zelfde rijen)."""
    y, yhat = np.asarray(y, float), np.clip(np.asarray(yhat, float), 0, None)
    e = yhat - y
    daily = pd.DataFrame({"p": meta.PLANT.values, "d": meta.DAY.values, "t": y, "f": yhat}) \
        .groupby(["p", "d"])[["t", "f"]].sum() * 0.25
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"nRMSE_%": float(np.sqrt((e ** 2).mean()) * 100), "nMAE_%": float(np.abs(e).mean() * 100),
            "RMSE_kW": float(np.sqrt((e ** 2).mean()) * HOUSE_KWP), "MAE_kW": float(np.abs(e).mean() * HOUSE_KWP),
            "R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "dagfout_%": float((daily.f - daily.t).abs().mean() / daily.t.mean() * 100)}


def choose(tab, eenvoud, fold_cols):
    """Keuzeregel (BESLISSINGEN #2). tab: per model 'nRMSE_%' (gemiddeld) + kolommen per vouw;
    eenvoud: model -> (complex 0/1, ruwe inputs, features). Geeft (keuze, motivatie)."""
    from .config import TIE_PP, COMPLEX_PP
    cand = tab.loc[list(eenvoud)]
    best = cand["nRMSE_%"].idxmin()
    simple = [n for n in cand.index if eenvoud[n][0] == 0]
    best_simple = cand.loc[simple, "nRMSE_%"].idxmin()
    note = "het beste model is geen complex model"
    if eenvoud[best][0] == 1:
        gain = cand.loc[best_simple, "nRMSE_%"] - cand.loc[best, "nRMSE_%"]
        both = all(cand.loc[best, c] < cand.loc[best_simple, c] for c in fold_cols)        # op ALLE vouwen beter
        if gain >= COMPLEX_PP and both:
            return best, f"complex model '{best}' is {gain:.2f} pp beter en op alle vouwen beter"
        note = (f"'{best}' (complex) is maar {gain:.2f} pp beter dan '{best_simple}'"
                f"{'' if both else ' en niet op alle vouwen beter'} -> geen complex model")
        best = best_simple
    close = [n for n in simple if cand.loc[n, "nRMSE_%"] <= cand.loc[best, "nRMSE_%"] + TIE_PP]
    pick = min(close, key=lambda n: (eenvoud[n], cand.loc[n, "nRMSE_%"]))
    return pick, (f"laagste nRMSE: '{cand['nRMSE_%'].idxmin()}' ({cand['nRMSE_%'].min():.2f} %); {note}. "
                  f"Binnen {TIE_PP} pp: {close} -> eenvoudigste: '{pick}' ({cand.loc[pick, 'nRMSE_%']:.2f} %)")


def bootstrap_days(te, preds, pairs=(), n=None, seed=None, level=0.90):
    """Bootstrap over testDAGEN (AGENTS.md §6.5): trek n keer evenveel dagen met teruglegging (beide plants van
    een dag blijven samen) en bereken de nRMSE. Geeft per model (nRMSE, ondergrens, bovengrens) en per paar (a, b)
    het interval van nRMSE_a - nRMSE_b. Verschil met een interval rond 0 = niet aantoonbaar."""
    from .config import N_BOOTSTRAP, RANDOM_STATE
    n = n or N_BOOTSTRAP
    rng = np.random.default_rng(RANDOM_STATE if seed is None else seed)
    days = np.sort(te.DAY.unique())
    idx = {d: i for i, d in enumerate(days)}
    di = np.array([idx[d] for d in te.DAY.values])
    se = {m: np.bincount(di, weights=(np.clip(np.asarray(p, float), 0, None) - te.FRAC.values) ** 2,
                         minlength=len(days))                    # zelfde als metrics(): negatief -> 0
          for m, p in preds.items()}
    cnt = np.bincount(di, minlength=len(days)).astype(float)
    draws = rng.integers(0, len(days), size=(n, len(days)))
    w = np.stack([np.bincount(r, minlength=len(days)) for r in draws])          # hoe vaak elke dag getrokken is
    boot = {m: np.sqrt((w @ s) / (w @ cnt)) * 100 for m, s in se.items()}
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    point = {m: float(np.sqrt(s.sum() / cnt.sum()) * 100) for m, s in se.items()}
    models = pd.DataFrame({m: {"nRMSE_%": point[m], "laag_%": float(np.percentile(boot[m], q[0])),
                               "hoog_%": float(np.percentile(boot[m], q[1]))} for m in preds}).T
    diffs = pd.DataFrame({f"{a} - {b}": {"verschil_pp": point[a] - point[b],
                                         "laag_pp": float(np.percentile(boot[a] - boot[b], q[0])),
                                         "hoog_pp": float(np.percentile(boot[a] - boot[b], q[1]))}
                          for a, b in pairs}).T
    if len(diffs):
        diffs["aantoonbaar"] = (diffs.laag_pp > 0) | (diffs.hoog_pp < 0)
    return models, diffs


KT_BINS = [0, 0.45, 0.75, 9]                 # bewolkt | half bewolkt | (bijna) helder (zoals V3-V5)


def fit_band(cal, kt_col="KT_W", bins=KT_BINS):
    """Band P10-P90 per bewolkingsklasse (fase 4): kwantielen van gemeten / voorspeld op out-of-sample-rijen.
    cal heeft FRAC, pred en de helderheidskolom kt_col."""
    ok = cal.pred > 0.05
    r = pd.DataFrame({"r": cal.FRAC.values[ok] / cal.pred.values[ok],
                      "k": np.digitize(cal[kt_col].values[ok], bins) - 1})
    return {str(int(k)): (float(g.r.quantile(0.10)), float(g.r.quantile(0.90))) for k, g in r.groupby("k")}


def apply_band(band, df, kt_col="KT_W", bins=KT_BINS):
    """(P10, P90) = voorspelling x (ondergrens, bovengrens) van de klasse; onbekende klasse -> 0,8-1,2."""
    k = (np.digitize(df[kt_col].values, bins) - 1).astype(int).astype(str)
    lo = np.array([band.get(x, (0.8, 1.2))[0] for x in k]); hi = np.array([band.get(x, (0.8, 1.2))[1] for x in k])
    return df.pred.values * lo, df.pred.values * hi


def coverage(df, p10, p90):
    """Dekking (%) = deel van de kwartieren met de gemeten waarde binnen de band."""
    return float(np.mean((df.FRAC.values >= p10) & (df.FRAC.values <= p90)) * 100)


def mean_day(train, test):
    """Baseline: gemiddelde FRAC per kwartier en plant over de trainingsdagen (klimatologie)."""
    prof = train.groupby(["PLANT", "SLOT"]).FRAC.mean()
    return np.nan_to_num(prof.reindex(list(zip(test.PLANT, test.SLOT))).values)


def persistence(full, test):
    """Baseline: zelfde kwartier van gisteren (gisteren is op het voorspelmoment bekend). full = alle dagen
    met FRAC in dezelfde schaal. Ontbrekend -> 0."""
    s = full.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    s = s[~s.index.duplicated()]
    return s.reindex(list(zip(test.PLANT, test.DAY - 1, test.SLOT))).fillna(0).values

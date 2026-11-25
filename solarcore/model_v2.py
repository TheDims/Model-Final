# -*- coding: utf-8 -*-
"""
Eindmodel v2 - "nieuw huis": werkt zonder trainingsperiode en kalibreert zichzelf (BESLISSINGEN #53, #54)
========================================================================================================
Bundelt de winsten uit TAAK 5:
  1. start: pvlib met kWp, dakhoek en richting (zero-shot);
  2. niveaufactor f = som gemeten / som pvlib over de laatste 4 weken, elke week bijgewerkt (#42);
  3. gemiddelde dag = gemiddelde helderheid K x heldere-hemelcurve, ook uit de laatste 4 weken;
  4. komende dagen: w_d x voorspelling + (1 - w_d) x gemiddelde dag (#41);
  5. band P10-P90: voorspelling + kwantielen van (gemeten - voorspeld) per klasse van de voorspelling,
     elke week op de laatste 4 weken (#54, variant B).
Alle functies werken op één huis (of per systeem in een groupby) en gebruiken alleen data van VOOR de week.
"""
import numpy as np
import pandas as pd

WEEKS = 4                                   # kalibratievenster (weken)
PRED_BINS = [-1, 0.05, 0.2, 0.5, 9]         # klassen van de voorspelling (fractie van de capaciteit) voor de band
CS_MIN, COSZ_SCALE = 200.0, 0.3             # heldere hemel: alleen kwartieren met genoeg zon


def week_stats(hist):
    """Kalibratie uit de laatste weken van één huis. hist: P_KW (gemeten), PV_KW (pvlib-voorspelling, kW),
    CS_GHI (W/m²), COSZ. Geeft f, c (kW per W/m² heldere hemel) en K (gemiddelde helderheid)."""
    h = hist[hist.P_KW.notna() & hist.PV_KW.notna()]
    f = float(h.P_KW.sum() / h.PV_KW.sum()) if h.PV_KW.sum() > 0 else 1.0
    s = h[(h.COSZ > COSZ_SCALE) & (h.CS_GHI > 0)]
    c = float((s.P_KW / s.CS_GHI).quantile(0.9)) if len(s) > 20 else np.nan
    k = h[h.CS_GHI > CS_MIN]
    kmean = float((k.P_KW / (c * k.CS_GHI)).clip(0, 1.5).mean()) if len(k) > 20 and c > 0 else np.nan
    return {"f": f, "c": c, "K": kmean, "n": int(len(h))}


def rolling_stats(df, week_col="WEEK"):
    """Per huis en per week: week_stats over de WEEKS weken ervoor (nooit de week zelf). df heeft PLANT + week_col."""
    rows = []
    for sid, g in df.groupby("PLANT"):
        weeks = sorted(g[week_col].unique())
        for w in weeks:
            hist = g[(g[week_col] < w) & (g[week_col] >= w - WEEKS)]
            rows.append({"PLANT": sid, week_col: w, **(week_stats(hist) if len(hist) else
                                                      {"f": 1.0, "c": np.nan, "K": np.nan, "n": 0})})
    return pd.DataFrame(rows)


def predict(pv_kw, cs_ghi, f, c, kmean, w=1.0):
    """Voorspelling (kW) = w x f x pvlib + (1 - w) x gemiddelde dag; zonder historie (c of K onbekend) alleen f x pvlib."""
    pred = f * np.asarray(pv_kw, float)
    mean_day = np.asarray(kmean, float) * np.asarray(c, float) * np.asarray(cs_ghi, float)
    return np.where(np.isfinite(mean_day), w * pred + (1 - w) * mean_day, pred), np.where(np.isfinite(mean_day), mean_day, pred)


def mix_weight(pred, mean_day, y):
    """Gewicht w in [0, 1] met kleinste kwadraten: y ~ w x pred + (1 - w) x mean_day."""
    d = np.asarray(pred) - np.asarray(mean_day)
    if not (d ** 2).sum() > 0:                  # voorspelling = gemiddelde dag (of geen data): mengen maakt niets uit
        return 1.0
    return float(np.clip(((np.asarray(y) - np.asarray(mean_day)) * d).sum() / (d ** 2).sum(), 0, 1))


def band_quantiles(hist_pred, hist_y):
    """Band variant B: per klasse van de voorspelling de 10e en 90e percentiel van (gemeten - voorspeld)."""
    b = np.digitize(hist_pred, PRED_BINS) - 1
    r = pd.Series(np.asarray(hist_y) - np.asarray(hist_pred)).groupby(b).quantile([0.1, 0.9]).unstack()
    return {int(k): (float(v[0.1]), float(v[0.9])) for k, v in r.iterrows()}


def apply_band(pred, bq, default=(-0.1, 0.1)):
    """(P10, P90) = voorspelling + kwantielen van de klasse; P10 nooit onder 0."""
    b = np.digitize(pred, PRED_BINS) - 1
    lo = np.array([bq.get(int(x), default)[0] for x in b]); hi = np.array([bq.get(int(x), default)[1] for x in b])
    return np.clip(pred + lo, 0, None), pred + hi

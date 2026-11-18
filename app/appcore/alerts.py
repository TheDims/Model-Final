# -*- coding: utf-8 -*-
"""
Melding bij te lage opwek: panelen schoonmaken of onderhoud nodig?
==================================================================
Idee
  Het huis meet zelf hoeveel stroom de panelen per dag maakten (omvormer of meter).
  Het model rekent uit hoeveel het had MOETEN zijn met het weer van die dag en met
  SCHONE panelen (forecast.expected_history). r = gemeten / verwacht.

  - Eerst leert de app de 'normale' verhouding van dit huis (mediaan van de eerste 10
    zonnige dagen). Zo tellen een vaste afwijking van het model of van de kWp niet mee.
  - Alleen dagen met genoeg zon tellen (heldere-hemel-index >= 0,5): op bewolkte dagen
    is de weersvoorspelling te onzeker.

  Meldingen
    SERVICE (onderhoud):  één zonnige dag met r < t_service   (bv. omvormer of string uit)
                          OF nog steeds te laag NA regen of schoonmaken.
    CLEAN (schoonmaken):  mediaan van de laatste 5 zonnige dagen < t_clean.
    CLEAN (trend):        (optioneel, t_trend) de opwek daalt gestaag sinds de laatste regen of
                          schoonmaakbeurt: robuuste trendlijn (Theil-Sen) over de laatste
                          trend_days zonnige dagen geeft een daling van >= t_trend. Bedoeld voor
                          LANGZAAM stof (0,5-1 %/dag), dat de 5-dagenregel pas laat ziet.

De drempels worden gekozen en getest in stap2_meldingen_kalibreren.py
(-> modellen/meldingsregels.json).
"""
import json

import numpy as np
import pandas as pd

from .config import RULES_FILE

DEFAULT_RULES = {"min_kt": 0.5, "baseline_days": 10, "baseline_q": 0.5, "window": 5,
                 "t_service": 0.7, "t_clean": 0.8, "rain_mm": 6.0,
                 "t_trend": None, "trend_days": 14, "trend_min_points": 8}


def trend_drop(r):
    """Daling over de reeks volgens een robuuste trendlijn (Theil-Sen: mediaan van alle
    hellingen tussen twee punten). Uitschieters (bv. één storingsdag) tellen nauwelijks mee.
    Geeft de relatieve daling van begin tot eind (0,10 = 10 % lager)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    i, j = np.triu_indices(n, k=1)
    slope = np.median((r[j] - r[i]) / (j - i))
    return float(-slope * (n - 1) / max(np.median(r), 1e-6))


def load_rules():
    if RULES_FILE.exists():
        return {**DEFAULT_RULES, **json.loads(RULES_FILE.read_text())["rules"]}
    return dict(DEFAULT_RULES)


def ratios(actual, expected, rules, baseline=None):
    """actual: Series (datum -> gemeten kWh). expected: DataFrame (datum -> exp_kWh, kt, rain_mm).
    Geeft per dag: gemeten, verwacht, zonnig genoeg?, r (t.o.v. de normale verhouding)."""
    df = expected.join(actual.rename("act"), how="inner").dropna(subset=["act"]).sort_index()
    df["valid"] = (df.kt >= rules["min_kt"]) & (df.exp_kWh > 0)
    df["raw"] = df.act / df.exp_kWh.where(df.exp_kWh > 0)
    if baseline is None:
        first = df[df.valid].raw.head(rules["baseline_days"])
        baseline = float(first.quantile(rules["baseline_q"])) if len(first) >= 3 else 1.0
    df["r"] = df.raw / baseline
    return df, baseline


def status(df, rules, cleaned=()):
    """Status op de laatste dag van df. cleaned = datums waarop de gebruiker schoonmaakte."""
    v = df[df.valid]
    if len(v) == 0:
        return {"level": "nodata"}
    washed = set(df.index[df.rain_mm >= rules["rain_mm"]]) | {pd.Timestamp(c).date() for c in cleaned}
    last_wash = max((w for w in washed if w <= df.index[-1]), default=None)
    recent = v.tail(rules["window"])
    out = {"last_date": v.index[-1], "r_last": float(v.r.iloc[-1]), "r_window": float(recent.r.median()),
           "n_window": len(recent), "last_wash": last_wash,
           "dry_days": (df.index[-1] - last_wash).days if last_wash else None}
    after = v[v.index > last_wash] if last_wash else v.iloc[0:0]
    if out["r_last"] < rules["t_service"]:
        out.update(level="service", reason="low_day")
    elif len(recent) >= 3 and out["r_window"] < rules["t_clean"]:
        if len(after) >= 2 and after.r.tail(rules["window"]).median() < rules["t_clean"]:
            out.update(level="service", reason="low_after_wash")
        else:
            out.update(level="clean", reason="dust")
    else:
        out.update(level="ok", reason="ok")
    if out["level"] == "ok" and rules.get("t_trend"):                # langzaam stof?
        pts = (after if len(after) else v).r.tail(rules["trend_days"])
        if len(pts) >= rules["trend_min_points"]:
            out["trend_drop"] = trend_drop(pts.values)
            if out["trend_drop"] >= rules["t_trend"]:
                out.update(level="clean", reason="trend")
    return out


def dust_reminder(soiling_pct, dry_days):
    """Herinnering zonder metingen: volgens het stofmodel (Kimber) al flink vervuild."""
    return soiling_pct >= 3.0 or (dry_days is not None and dry_days >= 30)


def demo_log(expected, scenario="healthy", seed=1):
    """NAGEBOOTSTE meetdata voor de demo (geen echte metingen):
    healthy = normaal (95 % van verwacht, +-5 % ruis per dag)
    dusty   = stof hoopt op met 0,9 % per dag in de laatste 25 dagen. Voor de demo doen we
              alsof die periode DROOG was (de app zet dan ook de regen op 0)
    fault   = de laatste 3 dagen werkt een deel van het systeem niet (40 % opwek)."""
    rng = np.random.default_rng(seed)
    e = expected.sort_index()
    act = e.exp_kWh.values * 0.95 * (1 + rng.normal(0, 0.05, len(e)))
    if scenario == "dusty":
        k = np.clip(np.arange(len(e)) - (len(e) - 25), 0, None)
        act = act * (1 - 0.009 * k)
    if scenario == "fault":
        act[-3:] = act[-3:] * 0.4
    return pd.Series(np.clip(act, 0, None).round(2), index=e.index)

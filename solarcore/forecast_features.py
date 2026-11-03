# -*- coding: utf-8 -*-
"""
Features om ECHT vooruit te voorspellen (fase 2): alleen informatie van vóór het voorspelmoment t0
================================================================================================
  add_clear(df, train)   FRAC_CS = opwek bij heldere hemel (Ineichen x schaal per plant, schaal ALLEEN uit
                         de training) en K = FRAC / FRAC_CS (heldere-hemelindex: 1 = helder, 0 = donker).
  add_daily(df)          KD = gemiddelde K per plant-dag, KSD = gemiddelde helderheid van de sensor per plant-dag.
  intraday(df, h)        voorspelling h kwartieren vooruit: lags van K en de sensor op t0 = t - h (en eerder).
  dayahead(df)           voorspelling voor morgen, gemaakt om 21:00: daggemiddelden van vandaag en gisteren.
Alle lags met shift(+n) binnen één plant op een volledig kwartier-rooster: nooit toekomst (geen shift(-1),
geen center=True). Ontbrekende lags (nacht, gaten) -> gemiddelde K van de vorige dag (op t0 bekend).
"""
import numpy as np
import pandas as pd

K_MIN_CS = 50.0          # W/m²: onder deze heldere-hemelstraling is K niet zinvol (schemer)
K_CLIP = 1.5
CS_SCALE_QUANTILE = 0.9  # schaal heldere hemel = 90e percentiel van FRAC / CS_GHI in de training (zon hoog)
LAGS = (0, 1, 2, 4)      # t0, t0 - 15 min, t0 - 30 min, t0 - 1 uur


def add_clear(df, train):
    """FRAC_CS en K. De schaal per plant komt alleen uit de trainingsrijen (zon hoog: COSZ > 0,3)."""
    sub = train[(train.COSZ > 0.3) & train.FRAC.notna()]
    c = (sub.FRAC / sub.CS_GHI).groupby(sub.PLANT).quantile(CS_SCALE_QUANTILE)
    fcs = df.CS_GHI * df.PLANT.map(c)
    k = (df.FRAC / fcs).where(df.CS_GHI > K_MIN_CS).clip(0, K_CLIP)
    return df.assign(FRAC_CS=fcs, K=k)


def add_daily(df):
    """Daggemiddelden per plant-dag (alleen kwartieren met genoeg zon). Pas om 21:00 van die dag bekend."""
    core = df[df.CS_GHI > 200]
    kd = core.groupby(["PLANT", "DAY"]).K.mean().rename("KD")
    ksd = core.groupby(["PLANT", "DAY"]).KT_SENSOR.mean().rename("KSD")
    return df.join(kd, on=["PLANT", "DAY"]).join(ksd, on=["PLANT", "DAY"])


def _prev_day(df, col, n=1):
    """Waarde van col van n dagen eerder (zelfde plant)."""
    s = df.groupby(["PLANT", "DAY"])[col].first()
    return pd.Series(s.reindex(list(zip(df.PLANT, df.DAY - n))).values, index=df.index)


def intraday(df, h, k_fill):
    """Features voor h kwartieren vooruit (t0 = t - h). df: volledig rooster, gesorteerd per plant en tijd.
    k_fill = gemiddelde K uit de TRAINING (als ook de vorige dag onbekend is; nooit uit de testdagen)."""
    df = df.sort_values(["PLANT", "DATE_TIME"])
    g = df.groupby("PLANT")
    kd1 = _prev_day(df, "KD", 1)
    fill = kd1.fillna(k_fill)
    out = pd.DataFrame(index=df.index)
    for lag in LAGS:
        out[f"K_L{lag}"] = g.K.shift(h + lag).fillna(fill)
    out["KS_L0"] = g.KT_SENSOR.shift(h).fillna(_prev_day(df, "KSD", 1)).fillna(fill)
    out["FRAC_L0"] = g.FRAC.shift(h)                       # voor gewone persistence
    out["KD_1"] = fill
    return df.join(out)


def dayahead(df):
    """Features voor morgen (gemaakt om 21:00 de dag ervoor): daggemiddelden van vandaag (D-1) en gisteren (D-2)."""
    out = pd.DataFrame(index=df.index)
    out["KD_1"] = _prev_day(df, "KD", 1)
    out["KD_2"] = _prev_day(df, "KD", 2).fillna(out.KD_1)
    out["KSD_1"] = _prev_day(df, "KSD", 1).fillna(out.KD_1)
    s = df.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    s = s[~s.index.duplicated()]
    out["FRAC_L0"] = s.reindex(list(zip(df.PLANT, df.DAY - 1, df.SLOT))).values   # zelfde kwartier gisteren
    return df.join(out)

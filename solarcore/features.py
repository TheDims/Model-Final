# -*- coding: utf-8 -*-
"""
Features: tijd van de dag, zonnestand en de fysische input (fase 1+)
====================================================================
  add_time(df)    SLOT (kwartier 0-95), HOUR_SIN/HOUR_COS (tijd van de dag als cirkel)
  add_solar(df)   COSZ = cos(zenit) op de GESCHATTE plantlocatie (pvlib, midden van het kwartier)
  add_physical(df) IRR_T = IRRADIATION x (1 + gamma x (T_module - 25)): straling met temperatuurcorrectie
  add_target(df, cap) FRAC = AC_KW / capaciteit (capaciteit alleen uit de training, zie capacity())
"""
import numpy as np
import pandas as pd
import pvlib

from .config import PLANT_LOCATIONS, TZ, GAMMA_PDC, CAPACITY_QUANTILE, MIN_COSZ


def add_time(df):
    h = df.DATE_TIME.dt.hour + df.DATE_TIME.dt.minute / 60 + 7.5 / 60      # midden van het kwartier
    return df.assign(SLOT=(df.DATE_TIME.dt.hour * 4 + df.DATE_TIME.dt.minute // 15),
                     HOUR_SIN=np.sin(2 * np.pi * h / 24), HOUR_COS=np.cos(2 * np.pi * h / 24))


def add_solar(df):
    """cos(zenit) per kwartier op de geschatte locatie van elke plant (overgenomen idee uit V5 physics.py)."""
    out = []
    for p, sub in df.groupby("PLANT"):
        lat, lon = PLANT_LOCATIONS[p]
        t = pd.DatetimeIndex(sub.DATE_TIME + pd.Timedelta(minutes=7.5)).tz_localize(TZ)
        sp = pvlib.solarposition.get_solarposition(t, lat, lon)
        out.append(sub.assign(COSZ=np.cos(np.deg2rad(sp.zenith.values)).clip(0)))
    return pd.concat(out).sort_index()


def add_clearsky(df):
    """CS_GHI (W/m², Ineichen op de geschatte locatie) en KT_SENSOR = gemeten straling / heldere hemel:
    hoe helder het was (1 = onbewolkt). Alleen voor analyse van de residuen, geen input."""
    out = []
    for p, sub in df.groupby("PLANT"):
        lat, lon = PLANT_LOCATIONS[p]
        t = pd.DatetimeIndex(sub.DATE_TIME + pd.Timedelta(minutes=7.5)).tz_localize(TZ)
        cs = pvlib.location.Location(lat, lon, TZ).get_clearsky(t, model="ineichen").ghi.values
        kt = np.where(cs > 50, sub.IRRADIATION.values * 1000 / np.maximum(cs, 1), np.nan)
        out.append(sub.assign(CS_GHI=cs, KT_SENSOR=np.clip(kt, 0, 1.5)))
    return pd.concat(out).sort_index()


def add_physical(df, gamma=GAMMA_PDC):
    """Straling x temperatuurcorrectie (PVWatts-vorm). gamma is een AANNAME (-0,4 %/°C)."""
    return df.assign(IRR_T=df.IRRADIATION * (1 + gamma * (df.MODULE_TEMPERATURE - 25)))


def capacity(train):
    """Capaciteit per plant = 99e percentiel van AC_KW overdag, ALLEEN uit de trainingsdagen."""
    day = train[train.COSZ > MIN_COSZ]
    return day.groupby("PLANT").AC_KW.quantile(CAPACITY_QUANTILE)


def add_target(df, cap):
    """FRAC = AC_KW / capaciteit van die plant."""
    return df.assign(FRAC=df.AC_KW / df.PLANT.map(cap))

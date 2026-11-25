# -*- coding: utf-8 -*-
"""
Live data voor de app (gecachet, zodat de app snel blijft)
==========================================================
  - weer:      Open-Meteo, 16 dagen vooruit + 45 dagen terug, + het weer van NU.
               Wordt elke 15 minuten opnieuw opgehaald (realtime).
  - opwek:     natuurkundig model per kwartier, voor het verleden (weer zoals het was)
               en de toekomst (weersvoorspelling).
  - jaar:      ERA5 vanaf 1 januari (1x per dag opgehaald).
  - verbruik:  geschat met appcore/usage.py, of gemeten als de gebruiker het invult.
  - meldingen: gemeten opwek vs verwacht (appcore/alerts.py).
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from appcore.config import PANELS_FILE, RULES_FILE, MODEL_INFO, TZ
from solarcore.weather import get_forecast, get_current
from appcore.forecast import (simulate_location, daily_table, site_meta, year_estimate,
                                expected_history)
from appcore.alerts import ratios, status, load_rules
from appcore import usage
from .state import setup

REFRESH_S = 900                       # 15 minuten: zo vaak wordt het weer opnieuw opgehaald


def now():
    return pd.Timestamp.now(tz=TZ).tz_localize(None)


# ---------- weer en opwek ----------
@st.cache_data(ttl=REFRESH_S, show_spinner=False)
def weather(lat, lon):
    hourly, elev = get_forecast(lat, lon, days=16, past_days=45)
    try:
        cur = get_current(lat, lon)
    except Exception:                                  # het huidige weer is 'nice to have'
        cur = None
    return hourly, elev, cur, now()


@st.cache_data(ttl=REFRESH_S, show_spinner=False)
def solar(lat, lon, kwp, tilt, az, ov_json, measured_json="{}"):
    """v2-voorspelling; measured_json = ingevulde dagopbrengst {datum: kWh} voor de kalibratie (f)."""
    hourly, elev, _, _ = weather(lat, lon)
    measured = {pd.Timestamp(k).date(): v for k, v in json.loads(measured_json).items()}
    d, p, cal = simulate_location(lat, lon, kwp, tilt, az, hourly, elev, json.loads(ov_json), measured=measured)
    meta = site_meta(d[d.FUTURE], hourly, p, lat, lon, elev, kwp, tilt, az)
    return d, daily_table(d, hourly), {**meta, "calib": cal}


@st.cache_data(ttl=86400, show_spinner=False)
def year(lat, lon, kwp, tilt, az, ov_json):
    hourly, elev, _, _ = weather(lat, lon)
    return year_estimate(lat, lon, kwp, tilt, az, json.loads(ov_json), recent=hourly, recent_elevation=elev)


@st.cache_data(ttl=REFRESH_S, show_spinner=False)
def expected(lat, lon, kwp, tilt, az, ov_json):
    hourly, elev, _, _ = weather(lat, lon)
    return expected_history(lat, lon, kwp, tilt, az, hourly, elev, json.loads(ov_json))


@st.cache_data(show_spinner=False)
def model_info():
    return json.loads(MODEL_INFO.read_text()) if MODEL_INFO.exists() else {}


@st.cache_data(show_spinner=False)
def rules_info():
    return json.loads(RULES_FILE.read_text()) if RULES_FILE.exists() else {}


def bundle():
    """Alles wat een pagina nodig heeft, voor de huidige instellingen."""
    from appcore.i18n import t                      # laadmelding in de taal van de gebruiker
    lat, lon, place, kwp, tilt, az, ov = setup()
    readings = load_readings()
    ov = {**ov, "wash_dates": readings["cleaned"]}  # knop "I cleaned my panels today" -> stofmodel
    key = (round(lat, 2), round(lon, 2), kwp, tilt, az, json.dumps(ov, sort_keys=True))
    sol = json.dumps({str(k): v for k, v in measured(readings, "solar").items()}, sort_keys=True)
    with st.spinner(t("loading", st.session_state.get("lang", "en"))):   # eerste keer ~5-10 s
        hourly, elev, cur, fetched = weather(*key[:2])
        d, daily, meta = solar(*key, sol)
    n = now()
    return dict(lat=lat, lon=lon, place=place, kwp=kwp, tilt=tilt, az=az, key=key, cur=cur,
                fetched=fetched, d=d, daily=daily, meta=meta, now=n, today=n.date(),
                readings=readings)


# ---------- gemeten waarden (ingevuld door de gebruiker) ----------
def load_readings():
    """{'log': {datum: {'solar': kWh, 'use': kWh}}, 'cleaned': [...], 'demo': ...}"""
    data = json.loads(PANELS_FILE.read_text(encoding="utf-8")) if PANELS_FILE.exists() else {}
    log = {k: (v if isinstance(v, dict) else {"solar": v}) for k, v in data.get("log", {}).items()}
    return {"log": log, "cleaned": data.get("cleaned", []), "demo": data.get("demo", False)}


def save_readings(data):
    PANELS_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")


def measured(readings, what):
    """Series datum -> gemeten kWh ('solar' of 'use')."""
    return pd.Series({pd.Timestamp(k).date(): v[what] for k, v in readings["log"].items()
                      if v.get(what) is not None}, dtype=float)


# ---------- verbruik en opwek samen ----------
def flows(d15, readings):
    """Per kwartier: opwek, verbruik, uit zon, overschot, van net (kW) + verbruik per groep per dag.
    Gemeten dagtotalen (als ingevuld) gaan voor: de dagvorm blijft, het totaal wordt gemeten."""
    cfg = st.session_state
    d15 = d15.assign(date=d15.DATE_TIME.dt.date)
    t_mean = d15.groupby("date")["T"].mean()
    cats = usage.daily_use(t_mean, cfg.monthly_units, cfg.ac_setpoint, cfg.has_ac)
    m_use, m_sol = measured(readings, "use"), measured(readings, "solar")
    for day, v in m_use.items():                       # gemeten verbruik: groepen schalen
        if day in cats.index and cats.loc[day].sum() > 0:
            cats.loc[day] *= v / cats.loc[day].sum()
    solar_kw = d15.pred_kW.values.copy()
    if len(m_sol):
        tot = d15.groupby("date").pred_kW.sum() * 0.25
        fac = (m_sol.reindex(tot.index) / tot.replace(0, np.nan)).fillna(1.0)
        solar_kw = solar_kw * d15.date.map(fac).values
    f = usage.split(solar_kw, usage.load_kw(cats, d15.DATE_TIME))
    f["measured_use"] = pd.Index(f.index.date).isin(m_use.index)
    return f, cats


def health():
    """(tabel per dag, status, regels, gemeten data, verwacht) voor de gemeten opwek."""
    lat, lon, _, kwp, tilt, az, ov = setup()
    readings, rules = load_readings(), load_rules()
    exp = expected(round(lat, 2), round(lon, 2), kwp, tilt, az, json.dumps(ov, sort_keys=True))
    if readings.get("demo") == "dusty":                # demo: nagebootste droge periode
        exp = exp.assign(rain_mm=0.0)
    act = measured(readings, "solar")
    if act.empty:
        return None, {"level": "nodata"}, rules, readings, exp
    df, _ = ratios(act, exp, rules)
    if df.valid.sum() < 5:
        return df, {"level": "learn", "n": int(df.valid.sum())}, rules, readings, exp
    return df, status(df, rules, cleaned=readings["cleaned"]), rules, readings, exp


def best_window(x15, hours=4):
    """Zonnigste blok van 'hours' uur in een kwartierreeks: (start, eind)."""
    end = x15.rolling(hours * 4).sum().idxmax()
    return end - pd.Timedelta(minutes=15 * (hours * 4 - 1)), end + pd.Timedelta(minutes=15)

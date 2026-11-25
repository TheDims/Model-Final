# -*- coding: utf-8 -*-
"""
Live voorspelling voor elke locatie in India met eindmodel v2 (BESLISSINGEN #53, #60)
====================================================================================
Overgenomen uit V5 solarhome/forecast.py (zelfde functies voor de app); het model is nu eindmodel v2:
  1. weersvoorspelling van Open-Meteo (+ de afgelopen 45 dagen: regen -> stof, en kalibratie);
  2. natuurkunde (solarcore/physics.py, dezelfde code als bij het trainen): pvlib met kWp, dakhoek, richting
     en de eigen setup van het huis -> PHYS_KW;
  3. v2 (solarcore/model_v2.py): voorspelling = w_d x f x pvlib + (1 - w_d) x gemiddelde dag, met f en de
     gemiddelde dag uit de laatste 4 weken ingevoerde opwek (elke week bijgewerkt; geen data -> f = 1, niet mengen);
  4. band P10-P90 variant B (voorspelling + kwantielen per voorspelklasse, fractie van kWp).

expected_history(...)  verwachte opwek van de AFGELOPEN dagen met schone panelen (pvlib, zonder f): voor de
                       meldingen (alerts.py); die vergelijken met de eigen normale verhouding van het huis.

overrides = eigen setup van het huis (uit de app), bv. {'gamma_pdc': -0.003,
  'dc_ac_ratio': 1.0, 'u0': 20, 'u1': 5, 'bifacial_gain': 0.05, 'wash_every_days': 14,
  'shading': 3, 'age': 2.5}   (shading/age = extra verlies in %).
"""
import joblib
import numpy as np
import pandas as pd

from solarcore import model_v2 as v2
from solarcore.config import APP_ROUGH_FROM_DAY
from solarcore.weather import get_forecast, get_history, rain_per_day, ATTRIBUTION, VARS
from solarcore.physics import hourly_to_15min, soiling_loss, panel_inputs, PARAMS
from .config import INDIA, MODEL_FILE, TZ

_MODEL = None
# solarcore gebruikt _W voor weermodel-straling; de app (V5) gebruikt de namen zonder _W
RENAME = {"GHI_W": "GHI", "CS_GHI_W": "CS_GHI", "KT_W": "KT", "DHI_W": "DHI", "DNI_W": "DNI"}
# Zonder modelbestand: pvlib met de standaardwaarden, niet mengen, standaardband (melding in de app)
FALLBACK = {"versie": "fallback", "params": PARAMS, "rad_shift_min": -30, "min_cosz": 0.02, "band": {},
            "v2": {"menggewichten_per_dag": {}}, "fallback": True}


def load_model():
    """Eindmodel v2 (solarcore/config.py: APP_MODEL_FILE); ontbreekt het bestand -> FALLBACK."""
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else FALLBACK
    return _MODEL


def in_india(lat, lon):
    return INDIA["lat"][0] <= lat <= INDIA["lat"][1] and INDIA["lon"][0] <= lon <= INDIA["lon"][1]


def model_params(m, overrides=None, clean=False):
    """Parameters van de natuurkunde: standaard + eigen setup.
    clean=True: schone panelen (geen stof), voor de verwachte opwek bij de meldingen."""
    ov = dict(overrides or {})
    p = {**PARAMS, **m["params"], **{k: v for k, v in ov.items() if k not in ("shading", "age")}}
    p["losses"] = {**p["losses"], "shading": ov.get("shading", 0), "age": ov.get("age", 0)}
    if clean:
        p["max_soiling"] = 0.0
    return p


def run_model(hourly, lat, lon, elevation, kwp, tilt=None, azimuth=180.0, p=None, m=None):
    """Weer (uurtabel) -> pvlib-vermogen van dit huis per kwartier (PHYS_KW = pred_kW, nog zonder kalibratie)."""
    m = m or load_model()
    p = p or model_params(m)
    w = hourly_to_15min(hourly, lat, lon, elevation, rad_shift_min=m["rad_shift_min"], tz=TZ)
    soil = soiling_loss(hourly, pd.DatetimeIndex(w.DATE_TIME), p, tz=TZ)
    # zelfde natuurkunde als bij het trainen (solarcore/physics.py); PHYS = kW per kWp
    d = panel_inputs(w, soil, lat, lon, elevation, tilt=tilt, azimuth=azimuth, p=p, tz=TZ).rename(columns=RENAME)
    d["PHYS_KW"] = d.PHYS * kwp
    d["pred_kW"] = d.PHYS_KW
    return d


def daily_expected(d, hourly):
    """Kwartiertabel -> per dag: verwacht kWh, heldere-hemel-index, regen (mm)."""
    g = d.assign(date=d.DATE_TIME.dt.date).groupby("date")
    days = list(g.groups)
    return pd.DataFrame({"exp_kWh": g.pred_kW.sum().values * 0.25,
                         "kt": (g.GHI.sum() / g.CS_GHI.sum().replace(0, np.nan)).values,
                         "rain_mm": rain_per_day(hourly).reindex(days).fillna(0).values}, index=days)


def _today():
    return pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()


def calibration(d, measured, today):
    """v2-kalibratie uit de ingevulde dagopbrengst (kWh) van de 4 weken vóór deze week (maandag):
    f = som gemeten / som pvlib; c = som gemeten / som heldere hemel (gemiddelde dag = c x CS_GHI).
    shortcut: dagtotalen i.p.v. kwartieren (de app vraagt alleen kWh per dag); bij kwartierdata -> v2.week_stats."""
    week = today - pd.Timedelta(days=today.weekday())
    days = pd.date_range(week - pd.Timedelta(weeks=v2.WEEKS), week - pd.Timedelta(days=1)).date
    meas = pd.Series(measured, dtype=float).reindex(days).dropna()
    g = d.assign(date=d.DATE_TIME.dt.date).groupby("date")
    pv, cs = (g.PHYS_KW.sum() * 0.25).reindex(meas.index), (g.CS_GHI.sum() * 0.25).reindex(meas.index)
    ok = pv.notna() & (pv > 0)
    if ok.sum() == 0:
        return {"f": 1.0, "c": np.nan, "n_days": 0, "since": str(week.date())}
    return {"f": float(meas[ok].sum() / pv[ok].sum()), "c": float(meas[ok].sum() / cs[ok].sum()),
            "n_days": int(ok.sum()), "since": str(week.date())}


def band(pred_kw, kwp, m):
    """Band P10-P90 variant B (#54): voorspelling + kwantielen van (gemeten - voorspeld) per voorspelklasse,
    in fractie van kWp. shortcut: vaste band uit de UK-test (laatste week); per huis bijwerken kan pas met kwartierdata."""
    lo, hi = v2.apply_band(np.asarray(pred_kw, float) / kwp, m["band"])
    return lo * kwp, hi * kwp


def simulate_location(lat, lon, kwp=3.0, tilt=None, azimuth=180.0, hourly=None, elevation=None,
                      overrides=None, days=16, measured=None):
    """Hele periode per kwartier: de afgelopen 45 dagen (weer zoals het WAS) en de komende dagen (voorspelling),
    met band p10/p90, FUTURE, DAYS_AHEAD (vandaag = 1) en ROUGH. measured: {datum: kWh} ingevulde opwek.
    Geeft (tabel, parameters, kalibratie)."""
    if not in_india(lat, lon):
        raise ValueError("Deze locatie ligt niet in India.")
    m = load_model()
    p = model_params(m, overrides)
    if hourly is None:
        hourly, elevation = get_forecast(lat, lon, days=days, past_days=45)
    d = run_model(hourly, lat, lon, elevation, kwp, tilt, azimuth, p, m)
    today = _today()
    cal = calibration(d, measured or {}, today)
    d["FUTURE"] = d.DATE_TIME >= today
    d["DAYS_AHEAD"] = np.where(d.FUTURE, (d.DATE_TIME.dt.normalize() - today).dt.days + 1, 0)
    weights = m["v2"]["menggewichten_per_dag"]
    w = d.DAYS_AHEAD.map(lambda k: weights.get(k, weights.get(max(weights), 1.0)) if weights and k > 0 else 1.0)
    pred, _ = v2.predict(d.PHYS_KW, d.CS_GHI, cal["f"], cal["c"], 1.0, w.values)   # K zit al in c (dagtotalen)
    d["pred_kW"] = np.clip(np.where(d.FUTURE, pred, cal["f"] * d.PHYS_KW), 0, kwp / p["dc_ac_ratio"])
    d.loc[d.COSZ <= m["min_cosz"], "pred_kW"] = 0.0
    d["p10_kW"], p90 = band(d.pred_kW, kwp, m)
    clear = np.maximum(d.PHYS_KW, d.pred_kW) / np.maximum(d.KT, 0.2) * 1.05     # bovenkant nooit boven heldere hemel (uit V5)
    d["p90_kW"] = np.minimum.reduce([p90, np.maximum(clear, d.pred_kW), np.full(len(d), kwp / p["dc_ac_ratio"])])
    d["ROUGH"] = d.DAYS_AHEAD >= APP_ROUGH_FROM_DAY
    return d, p, {**cal, "fallback": bool(m.get("fallback", False))}


def year_estimate(lat, lon, kwp=3.0, tilt=None, azimuth=180.0, overrides=None, recent=None,
                  recent_elevation=None):
    """Geschatte opwek per kwartier van 1 januari tot gisteren, met het weer zoals het WAS:
    ERA5 (archief, ~1 week vertraging) + de laatste dagen uit de forecast-API (recent)."""
    today = _today()
    end_archive = today - pd.Timedelta(days=7)
    start = pd.Timestamp(today.year, 1, 1)
    cols = ["DATE_TIME"] + VARS
    parts, elev, cut = [], recent_elevation, start
    if end_archive >= start:
        h_old, elev = get_history(lat, lon, str(start.date()), str(end_archive.date()))
        parts.append(h_old[cols])
        cut = end_archive + pd.Timedelta(days=1)
    if recent is not None:
        parts.append(recent.loc[(recent.DATE_TIME >= cut) & (recent.DATE_TIME < today), cols])
    hourly = pd.concat(parts, ignore_index=True).sort_values("DATE_TIME")
    m = load_model()
    d = run_model(hourly, lat, lon, elev, kwp, tilt, azimuth, model_params(m, overrides), m)
    return d[(d.DATE_TIME >= start) & (d.DATE_TIME < today)].reset_index(drop=True)


def forecast_location(lat, lon, kwp=3.0, tilt=None, azimuth=180.0, days=7, hourly=None,
                      elevation=None, overrides=None, measured=None):
    """Voorspelling per kwartier + per dag, vanaf vandaag. tilt=None -> hoek = breedtegraad.
    azimuth 180 = panelen naar het zuiden (beste richting in India).
    hourly/elevation: optioneel een eerder opgehaald Open-Meteo-antwoord (voor de app-cache)."""
    if hourly is None:
        hourly, elevation = get_forecast(lat, lon, days=days, past_days=45)
    d, p, cal = simulate_location(lat, lon, kwp, tilt, azimuth, hourly, elevation, overrides, measured=measured)
    d = d[d.FUTURE].reset_index(drop=True)
    meta = site_meta(d, hourly, p, lat, lon, elevation, kwp, tilt, azimuth)
    return d, daily_table(d, hourly), {**meta, "calib": cal}


def daily_table(d, hourly):
    """Kwartiertabel -> tabel per dag (opwek met band, temperatuur, bewolking, regen, stof, ruwe schatting)."""
    d = d.assign(CLOUD_SUN=d.CLOUD.where(d.COSZ > 0.05), date=d.DATE_TIME.dt.date)
    day = d.groupby("date")
    daily = pd.DataFrame({
        "kWh": day.pred_kW.sum() * 0.25, "p10_kWh": day.p10_kW.sum() * 0.25,
        "p90_kWh": day.p90_kW.sum() * 0.25,
        "t_mean": day["T"].mean(), "t_max": day["T"].max(), "t_min": day["T"].min(),
        "t_cell_max": day.T_CELL.max(), "cloud": day.CLOUD_SUN.mean(),
        "rain_prob": day.RAIN_PROB.max() if "RAIN_PROB" in d else np.nan,
        "soiling_%": day.SOILING.mean() * 100,
        "future": day.FUTURE.first() if "FUTURE" in d else True,
        "rough": day.ROUGH.first() if "ROUGH" in d else False,
    })
    daily["rain_mm"] = rain_per_day(hourly).reindex(daily.index).fillna(0).values
    return daily


def site_meta(d_future, hourly, p, lat, lon, elevation, kwp, tilt, azimuth):
    """Gegevens over de locatie en de omstandigheden (voor 'wat beïnvloedt je panelen')."""
    today = _today()
    sun = d_future[d_future.COSZ > 0.05]
    rain = rain_per_day(hourly)
    wet = rain[(rain >= p["clean_mm"]) & (rain.index < today.date())]
    return {"lat": lat, "lon": lon, "elevation_m": elevation, "kwp": kwp,
            "dry_days": (today.date() - wet.index[-1]).days if len(wet) else None,   # sinds flinke regen
            "tilt": abs(lat) if tilt is None else tilt, "azimuth": azimuth,
            "pressure_hpa": float(sun.P_HPA.mean()), "air_density": float(sun.RHO.mean()),
            "airmass_noon": float(sun.AM_ABS.min()), "water_vapour_cm": float(sun.PW_CM.mean()),
            "spectral_%": float((sun.SPECTRAL.mean() - 1) * 100),
            "soiling_now_%": float(d_future.SOILING.iloc[0] * 100),
            "source": ATTRIBUTION}


def expected_history(lat, lon, kwp, tilt=None, azimuth=180.0, hourly=None, elevation=None,
                     overrides=None):
    """Verwachte opwek van de AFGELOPEN dagen met SCHONE panelen en het weer van die dagen (pvlib, zonder f:
    de meldingen vergelijken met de eigen normale verhouding van het huis). Per datum: exp_kWh, kt, rain_mm."""
    m = load_model()
    if hourly is None:
        hourly, elevation = get_forecast(lat, lon, days=1, past_days=45)
    d = run_model(hourly, lat, lon, elevation, kwp, tilt, azimuth, model_params(m, overrides, clean=True), m)
    return daily_expected(d[d.DATE_TIME < _today()], hourly)

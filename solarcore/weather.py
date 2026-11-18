# -*- coding: utf-8 -*-
"""
Weer van Open-Meteo (CC BY 4.0, geen API-key): ERA5 (weer achteraf) en de live weersvoorspelling
================================================================================================
Overgenomen uit V5 solarhome/weather.py. Tijden = lokale tijd (standaard India, IST). Straling in W/m², wind in m/s.
  load_era5(lat, lon, start, end)    ERA5 met cache in data/ (onderzoek, TAAK 5)
  get_history(lat, lon, start, end)  ERA5 zonder cache (app: opwek van dit jaar)
  get_forecast(lat, lon, days)       weersvoorspelling + de afgelopen dagen (app)
  get_current(lat, lon)              het weer van NU (app)
  rain_per_day(hourly)               neerslag per dag (mm)
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .config import DATA, TZ

VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation", "cloud_cover"]
PERIOD = ("2020-04-01", "2020-06-18")        # april = aanloop voor het stofmodel
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


def _call(url, params, timeout=60, tries=3):
    """API-aanroep; bij een tijdelijke fout (bv. 503, time-out) nog twee keer proberen (uit V5)."""
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500
            if client_error or i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo fout: {data.get('reason')}")
    return data


def _frame(data, extra=()):
    """API-antwoord -> uurtabel met DATE_TIME en de variabelen."""
    h = data["hourly"]
    df = pd.DataFrame({v: h[v] for v in list(VARS) + list(extra) if v in h})
    df.insert(0, "DATE_TIME", pd.to_datetime(h["time"]))
    return df


def get_history(lat, lon, start, end, tz=TZ):
    """Uurlijks ERA5 (weer achteraf) zonder cache -> (tabel, hoogte in m)."""
    data = _call(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                               "hourly": ",".join(VARS), "timezone": tz, "wind_speed_unit": "ms"})
    return _frame(data), float(data["elevation"])


def load_era5(lat, lon, start=PERIOD[0], end=PERIOD[1], tz=TZ, folder=DATA):
    """Uurlijks ERA5 voor (lat, lon) -> (tabel, hoogte in m). Cache: {folder}/era5_{lat}_{lon}_{start}_{end}.csv.

    tz en folder alleen anders voor externe datasets (TAAK 5, bv. Asia/Hong_Kong); standaard India.
    """
    cache = folder / f"era5_{lat}_{lon}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["DATE_TIME"])
        return df.drop(columns="elevation"), float(df.elevation.iloc[0])
    df, elev = get_history(lat, lon, start, end, tz)
    df.assign(elevation=elev).to_csv(cache, index=False)
    return df, elev


def get_forecast(lat, lon, days=7, past_days=45, tz=TZ):
    """Weersvoorspelling voor vandaag + komende dagen (max 16), plus de afgelopen past_days dagen
    (nodig voor de regen -> vervuiling, en voor de kalibratie van het huis)."""
    data = _call(FORECAST_URL, {"latitude": lat, "longitude": lon, "timezone": tz,
                                "hourly": ",".join(VARS + ["precipitation_probability"]),
                                "forecast_days": days, "past_days": past_days, "wind_speed_unit": "ms"})
    return _frame(data, extra=["precipitation_probability"]), float(data["elevation"])


def _sky(code):
    """WMO-weercode -> eenvoudige klasse (voor tekst en icoon in de app)."""
    if code is None:
        return "partly"
    code = int(code)
    if code == 0:
        return "clear"
    if code == 1:
        return "mostly_clear"
    if code == 2:
        return "partly"
    if code in (3, 45, 48):
        return "overcast"
    if 51 <= code <= 57:
        return "drizzle"
    if 95 <= code <= 99:
        return "storm"
    return "rain" if (61 <= code <= 67 or 80 <= code <= 82) else "overcast"


def get_current(lat, lon, tz=TZ):
    """Het weer van NU (Open-Meteo 'current', elke 15 min bijgewerkt)."""
    data = _call(FORECAST_URL, {"latitude": lat, "longitude": lon, "timezone": tz,
                                "current": "temperature_2m,weather_code,cloud_cover,is_day,"
                                           "shortwave_radiation,precipitation"})
    c = data["current"]
    return {"time": pd.Timestamp(c["time"]), "temp": c.get("temperature_2m"), "code": c.get("weather_code"),
            "sky": _sky(c.get("weather_code")), "cloud": c.get("cloud_cover"), "is_day": bool(c.get("is_day", 1)),
            "ghi": c.get("shortwave_radiation"), "rain": c.get("precipitation")}


def rain_per_day(hourly):
    """Neerslag per dag (mm) uit een uurtabel."""
    return hourly.assign(date=hourly.DATE_TIME.dt.date).groupby("date").precipitation.sum()

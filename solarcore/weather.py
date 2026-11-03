# -*- coding: utf-8 -*-
"""
ERA5-weer (Open-Meteo Historical API, CC BY 4.0) laden uit de cache
===================================================================
Overgenomen uit V5 solarhome/weather.py (get_history), ingekort: eerst de cache in data/, alleen als die
ontbreekt de API (internet). Tijden = lokale tijd India (IST). Straling in W/m², wind in m/s.
"""
import json
import urllib.parse
import urllib.request

import pandas as pd

from .config import DATA, TZ

VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "temperature_2m",
        "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation", "cloud_cover"]
PERIOD = ("2020-04-01", "2020-06-18")        # april = aanloop voor het stofmodel
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_era5(lat, lon, start=PERIOD[0], end=PERIOD[1]):
    """Uurlijks ERA5 voor (lat, lon) -> (tabel, hoogte in m). Cache: data/era5_{lat}_{lon}_{start}_{end}.csv."""
    cache = DATA / f"era5_{lat}_{lon}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["DATE_TIME"])
        return df.drop(columns="elevation"), float(df.elevation.iloc[0])
    q = ARCHIVE_URL + "?" + urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "hourly": ",".join(VARS),
        "timezone": TZ, "wind_speed_unit": "ms"})
    with urllib.request.urlopen(q, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    h = data["hourly"]
    df = pd.DataFrame({v: h[v] for v in VARS})
    df.insert(0, "DATE_TIME", pd.to_datetime(h["time"]))
    df.assign(elevation=data["elevation"]).to_csv(cache, index=False)
    return df, float(data["elevation"])

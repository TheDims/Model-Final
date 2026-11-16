# -*- coding: utf-8 -*-
"""
Archief-weersvoorspellingen van Open-Meteo (CC BY 4.0) ophalen en cachen (TAAK 5b, BESLISSINGEN #37)
====================================================================================================
  previous_runs(lat, lon, ...)   Previous Runs API: per uur de waarde zoals 0, 1, 2 dagen eerder voorspeld
  single_run(lats, lons, run)    Single Runs API: één modelrun (bv. ECMWF IFS 00 UTC), 15 dagen vooruit, meerdere locaties
Straling = gemiddelde over het VOORGAANDE uur (documentatie Open-Meteo). Alle tijden in UTC (timezone=GMT).
Elke aanvraag wordt als csv bewaard: opnieuw draaien haalt alleen wat nog ontbreekt.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

SATELLITE_URL = "https://satellite-api.open-meteo.com/v1/archive"
PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
FC_VARS = ["shortwave_radiation", "diffuse_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"]


def _get(url, params, tries=5, wait=60):
    """GET met nette herhaling bij een limiet (429) of tijdelijke fout."""
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                print(f"    API {e.code}, wacht {wait * (i + 1)} s ...")
                time.sleep(wait * (i + 1))
                continue
            raise


def satellite(lat, lon, start, end, cache, tz, model="satellite_radiation_seamless"):
    """Satellietstraling (Open-Meteo Satellite Radiation API; boven Azie = JMA Himawari), eigen resolutie
    (10 min), GHI en DHI in W/m². Per jaar opgehaald en samen gecachet."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["DATE_TIME"])
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a, b = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        r = _get(SATELLITE_URL, {"latitude": lat, "longitude": lon, "start_date": f"{a:%Y-%m-%d}", "end_date": f"{b:%Y-%m-%d}",
                                 "models": model, "hourly": "shortwave_radiation,diffuse_radiation",
                                 "temporal_resolution": "native", "timezone": tz})
        h = r["hourly"]
        parts.append(pd.DataFrame({"DATE_TIME": pd.to_datetime(h["time"]), "GHI_SAT": h["shortwave_radiation"],
                                   "DHI_SAT": h["diffuse_radiation"]}))
        time.sleep(2)
    df = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def satellite_15min(sat, shift):
    """10-min-satelliet -> 15 min: elke waarde dekt (t + shift - 10, t + shift]; twee stukjes van 5 min, dan
    gemiddelde per kwartier (label = begin), alleen als minstens 2 van de 3 stukjes bekend zijn."""
    s = sat.set_index("DATE_TIME").GHI_SAT
    five = pd.concat([pd.Series(s.values, index=s.index + pd.Timedelta(minutes=shift - 10)),
                      pd.Series(s.values, index=s.index + pd.Timedelta(minutes=shift - 5))]).sort_index()
    r = five.resample("15min", label="left", closed="left")
    return r.mean().where(r.count() >= 2)


def previous_runs(lat, lon, start, end, cache, model="ecmwf_ifs025", days=(1, 2), vars_=FC_VARS):
    """Uurtabel met per variabele de actuele run (_d0) en de voorspelling van N dagen eerder (_d1, _d2)."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["DATE_TIME"])
    names = [v + (f"_previous_day{d}" if d else "") for v in vars_ for d in (0,) + tuple(days)]
    r = _get(PREVIOUS_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(names), "start_date": start,
                            "end_date": end, "models": model, "timezone": "GMT", "wind_speed_unit": "ms"})
    h = r["hourly"]
    df = pd.DataFrame({n.replace("_previous_day", "_d") if "previous" in n else n + "_d0": h[n] for n in names})
    df.insert(0, "DATE_TIME", pd.to_datetime(h["time"]))
    df["elevation"] = r.get("elevation")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def single_run(lats, lons, ids, run, cache, model="ecmwf_ifs", days=15, vars_=FC_VARS[:3]):
    """Eén modelrun voor meerdere locaties -> lange tabel (ID, DATE_TIME, LEAD_H, variabelen)."""
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["DATE_TIME", "RUN"])
    r = _get(SINGLE_URL, {"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
                          "hourly": ",".join(vars_), "run": run, "models": model, "forecast_days": days,
                          "timezone": "GMT", "wind_speed_unit": "ms"})
    r = r if isinstance(r, list) else [r]
    parts = []
    for i, loc in zip(ids, r):
        d = pd.DataFrame(loc["hourly"])
        d.insert(0, "ID", i)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).rename(columns={"time": "DATE_TIME"})
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df["RUN"] = pd.Timestamp(run)
    df["LEAD_H"] = (df["DATE_TIME"] - df["RUN"]).dt.total_seconds() / 3600
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df

# -*- coding: utf-8 -*-
"""HKUST rooftop PV (Hong Kong, 2021-2023) inlezen: Brick-metadata (.ttl), omvormerdata (5 min) en weerstation (1 min).

Bron: Lin et al. (2025), Scientific Data; Dryad doi:10.5061/dryad.m37pvmd99 (versie 8 sep 2024, CC0).
Tijden staan zonder tijdzone-label in de bestanden; we nemen aan dat het lokale tijd Hong Kong is (HKT, UTC+8, geen
zomertijd). De README zegt het niet: dit controleren we met de zonnestand (zie taak5a_hkust_stations.py).
"""
import re

import numpy as np
import pandas as pd
import pvlib
import rdflib

from .config import (HKUST_TTL, HKUST_INV, HKUST_MET, HKUST_DATE_FORMATS, HKUST_NAME_ALIAS, HKUST_MIN_SUB, FREQ,
                     HKUST_CLEAN, HKUST_TZ, HKUST_LOCATION, HKUST_WET_MONTHS, CAPACITY_QUANTILE, MIN_COSZ, DATA)

# ERA5 + pvlib-kolommen die de modellen gebruiken (TAAK 5a stap 3)
WEATHER_COLS = ["GHI_W", "CS_GHI_W", "KT_W", "T", "RH", "WS10", "CLOUD", "PRECIP", "POA", "EFF_T", "KT_POA",
                "EFF_T_KT", "T_CELL", "SOILING", "CS_POA", "EFF_IRR"]

# weerstation: (map/bestandsnaam, kolom in de csv, nieuwe naam)
WEATHER_VARS = [("Irradiance", "Irradiance (W/m2)", "GHI_SENSOR"),
                ("Temperature", "Temp (Degree Celsius)", "T_AIR"),
                ("Wind", "Wind Speed (m/s)", "WIND"),
                ("Relative Humidity", "RH (%)", "RH")]

# SPARQL op de Brick-graaf: per systeem -> omvormer -> modulegroep (een omvormer kan 2 modulegroepen hebben)
_QUERY = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ext: <https://www.HKUST_PV_System.com/schema/BrickExtension#>
SELECT ?sys ?kw ?tilt ?az ?conn ?inv ?mod ?nmod ?modw ?modtype WHERE {
  ?sys a brick:PV_Generation_System .
  OPTIONAL { ?sys ext:ratedPowerOutput/brick:value ?kw }
  OPTIONAL { ?sys ext:tiltAngle/brick:value ?tilt }
  OPTIONAL { ?sys ext:azimuth/brick:value ?az }
  OPTIONAL { ?sys ext:connectionDate/brick:value ?conn }
  OPTIONAL { ?sys brick:hasPart ?inv .
             OPTIONAL { ?inv brick:hasPart ?mod .
                        OPTIONAL { ?mod ext:number/brick:value ?nmod }
                        OPTIONAL { ?mod ext:ratedPowerOutput/brick:value ?modw }
                        OPTIONAL { ?mod ext:specificType/brick:value ?modtype } } }
}"""


def norm_name(name):
    """Naam vergelijkbaar maken tussen csv en .ttl: kleine letters, alleen letters en cijfers."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return HKUST_NAME_ALIAS.get(key, key)


def read_brick(path=HKUST_TTL):
    """Brick-model inlezen met rdflib + SPARQL -> tabel per modulegroep (systeem, omvormer, kWp, tilt, azimuth, ...).

    Het bronbestand heeft 1 tikfout ('] .' waar '] ;' hoort, bij Zone_D): die repareren we in het geheugen,
    het ruwe bestand blijft onveranderd.
    """
    txt = path.read_text(encoding="utf-8")
    txt, n_fix = re.subn(r"\]\s*\.(\s*\n\s*ext:)", r"] ;\1", txt)
    g = rdflib.Graph()
    g.parse(data=txt, format="turtle")
    rows = [{k: (r[k].toPython() if r[k] is not None else None) for k in r.labels} for r in g.query(_QUERY)]
    df = pd.DataFrame(rows)
    for col in ("sys", "inv", "mod"):
        df[col] = df[col].astype(str).str.split("#").str[-1]
    for col in ("kw", "nmod", "modw"):
        df[col] = pd.to_numeric(df[col])
    df["conn"] = pd.to_datetime(df["conn"].astype(str).str.replace("T", " "), format="%Y-%m-%d %H:%M:%S")
    df.attrs["ttl_reparaties"] = n_fix
    return df


def systems_table(brick):
    """Eén rij per systeem: kWp (opgegeven en uit de modules), tilt, azimuth, aansluitdatum, aantal omvormers."""
    mods = brick.drop_duplicates(["sys", "inv", "mod", "nmod", "modw"])
    kwp_mod = (mods["nmod"] * mods["modw"]).groupby(mods["sys"]).sum() / 1000
    out = brick.groupby("sys").agg(kWp=("kw", "first"), tilt=("tilt", "first"), azimuth=("az", "first"),
                                   aangesloten=("conn", "first"), n_omvormers=("inv", "nunique"),
                                   moduletype=("modtype", lambda s: "/".join(sorted(set(s.dropna())))))
    out["kWp_modules"] = kwp_mod.round(2)
    out["key"] = [norm_name(s) for s in out.index]
    return out


def parse_time(s):
    """Tijdkolom met twee mogelijke formaten (per bestand consequent) -> datetime; altijd expliciet format=."""
    t = pd.to_datetime(s, format=HKUST_DATE_FORMATS[0], errors="coerce")
    for fmt in HKUST_DATE_FORMATS[1:]:
        t = t.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    if t.isna().any():
        raise ValueError(f"{t.isna().sum()} tijden onleesbaar, bv. {s[t.isna()].iloc[0]!r}")
    return t


def inverter_files():
    """csv-bestanden per station (met optimizer): {station-key: [paden]} en de leesbare naam."""
    files, names = {}, {}
    for p in sorted(HKUST_INV.glob("*.csv")):
        station = re.sub(r"_Inverter(_\d+)?$", "", p.stem)
        files.setdefault(norm_name(station), []).append(p)
        names[norm_name(station)] = station
    return files, names


def read_station_power(paths):
    """Omvormers van één station optellen -> AC-vermogen (kW) per 5 min. NaN als een omvormer ontbreekt.

    Kolom totalActivePower(W) = AC-werkvermogen van de omvormer. Dubbele tijdstempels -> gemiddelde.
    """
    parts = []
    for p in paths:
        d = pd.read_csv(p, usecols=["Time", "totalActivePower(W)"])
        s = d.set_index(parse_time(d["Time"]))["totalActivePower(W)"]
        parts.append(s.groupby(level=0).mean().rename(p.stem))
    wide = pd.concat(parts, axis=1)
    return (wide.sum(axis=1, min_count=len(paths)) / 1000).rename("P_KW")


def read_weather(var, col, years):
    """Eén grootheid van het weerstation (1 min) voor de gegeven jaren -> Series met tijdindex."""
    parts = []
    for y in years:
        p = HKUST_MET / var / f"{var}_{y}.csv"
        d = pd.read_csv(p, usecols=["Time", col])
        parts.append(d.set_index(parse_time(d["Time"]))[col])
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index()


def to_15min(s, min_count):
    """Gemiddelde per kwartier (label = begin van het kwartier, zoals Kaggle); NaN bij te weinig waarden."""
    r = s.resample(FREQ, label="left", closed="left")
    return r.mean().where(r.count() >= min_count)


def weather_15min(years):
    """Weerstation per kwartier: GHI_SENSOR (W/m²), T_AIR (°C), WIND (m/s, 10 m mast), RH (%)."""
    return pd.concat({new: to_15min(read_weather(var, col, years), HKUST_MIN_SUB[1])
                      for var, col, new in WEATHER_VARS}, axis=1)


def station_15min(paths):
    """Station per kwartier: P_KW (som van de omvormers); alleen als alle 3 vijf-minutenwaarden er zijn."""
    return to_15min(read_station_power(paths), HKUST_MIN_SUB[0])


# ---------- modeltabel (stap 3 en de kalibratietest) ----------
def load_clean():
    """Opgeschoonde 15-min-tabel (stap 2) + tijd, dagnummer, maand, seizoen, heldere hemel en KT van de sensor.
    STATION heet hier PLANT (zelfde kolomnamen als de Kaggle-functies); RH van de sensor -> RH_SENSOR."""
    from .features import add_time
    c = pd.read_csv(HKUST_CLEAN / "hkust_15min.csv", parse_dates=["DATE_TIME"])
    c = add_time(c.rename(columns={"STATION": "PLANT", "RH": "RH_SENSOR"}))
    c["DAY"] = (c.DATE_TIME.dt.normalize() - pd.Timestamp("2021-01-01")).dt.days + 1
    c["MONTH"] = c.DATE_TIME.dt.month
    c["SEASON"] = np.where(c.MONTH.isin(HKUST_WET_MONTHS), "moesson (mei-sep)", "droog (okt-apr)")
    loc = pvlib.location.Location(*HKUST_LOCATION, tz=HKUST_TZ)
    mid = pd.DatetimeIndex(c.DATE_TIME + pd.Timedelta(minutes=7.5)).tz_localize(HKUST_TZ)
    c["CS_GHI"] = loc.get_clearsky(mid, model="ineichen").ghi.values
    c["KT_SENSOR"] = (c.GHI_SENSOR / c.CS_GHI).where((c.CS_GHI > 50) & c.SENSOR_OK).clip(0, 1.5)
    return c


def load_era5_hk():
    """ERA5 (uur) voor het weerstation van HKUST, 2021-2023, uit de cache -> (tabel, hoogte)."""
    from .weather import load_era5
    return load_era5(*HKUST_LOCATION, "2021-01-01", "2023-12-31", tz=HKUST_TZ, folder=DATA / "extern" / "hkust")


def add_era5(clean, hourly, elev, rad_shift_min):
    """ERA5 -> 15 min -> pvlib-inputs (alle gekozen stations zijn plat: tilt 0°) + vlag OK."""
    from .physics import hourly_to_15min, soiling_loss, panel_inputs
    lat, lon = HKUST_LOCATION
    w = hourly_to_15min(hourly, lat, lon, elev, rad_shift_min=rad_shift_min, tz=HKUST_TZ)
    phys = panel_inputs(w, soiling_loss(hourly, pd.DatetimeIndex(w.DATE_TIME), tz=HKUST_TZ), lat, lon, elev,
                        tilt=0.0, azimuth=180.0, tz=HKUST_TZ)
    d = clean.merge(phys[["DATE_TIME"] + WEATHER_COLS], on="DATE_TIME", how="left")
    d["OK"] = d.DAG & d.P_KW.notna() & ~d.STORING & ~d.UITVAL & d[WEATHER_COLS].notna().all(axis=1)
    return d


def capacity(train):
    """Capaciteit per station = 99e percentiel van de opwek overdag, alleen uit de training (BESLISSINGEN #27)."""
    return train[train.COSZ > MIN_COSZ].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)


def mean_day(tr, te):
    """Gemiddelde dag per station, maand en kwartier uit de training (seizoen!); maand ontbreekt -> hele training."""
    prof = tr.groupby(["PLANT", "MONTH", "SLOT"]).FRAC.mean()
    fb = tr.groupby(["PLANT", "SLOT"]).FRAC.mean()
    v = prof.reindex(list(zip(te.PLANT, te.MONTH, te.SLOT))).values
    return np.nan_to_num(np.where(np.isnan(v), fb.reindex(list(zip(te.PLANT, te.SLOT))).values, v))


def pvlib_ac_kw(df, kwp_dc, kwp_ac, p):
    """Zero-shot 2b: effectieve straling + celtemperatuur (Faiman, ERA5-wind) -> DC uit de modules -> knippen op AC.
    kwp_dc, kwp_ac: arrays per rij; p = physics.PARAMS (aannames). Niets gefit op Hong Kong."""
    pdc = pvlib.pvsystem.pvwatts_dc(df.EFF_IRR, df.T_CELL, pdc0=kwp_dc * 1000, gamma_pdc=p["gamma_pdc"])
    pdc = pdc * (1 - pvlib.pvsystem.pvwatts_losses(**p["losses"]) / 100)
    pac = pvlib.inverter.pvwatts(pdc, pdc0=kwp_ac * 1000 / p["eta_inv"], eta_inv_nom=p["eta_inv"]) / 1000
    return np.clip(np.nan_to_num(np.asarray(pac, float)), 0, None)

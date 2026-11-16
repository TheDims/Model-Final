# -*- coding: utf-8 -*-
"""Open Climate Fix UK PV (Sheffield Solar) lezen: metadata, bekende foute periodes en 30-min-opwek via polars.

Bron: huggingface.co/datasets/openclimatefix/uk_pv (CC BY 4.0, doi:10.57967/hf/0878). Gated dataset: polars leest
met de omgevingsvariabele HF_TOKEN. Tijden in UTC; een rij met tijd t = energie van (t - 30 min, t]. Vermogen (kW)
= generation_Wh x 2 / 1000.
"""
import os

import numpy as np
import pandas as pd
import polars as pl
import pvlib

from .config import UKPV_HF, UKPV_RAW

# weerfeatures per bron (achtervoegsel _E = ERA5, _F1/_F2 = voorspeld 1/2 dagen eerder)
FEAT = ["GHI_W", "KT_W", "CS_GHI_W", "POA", "EFF_T", "EFF_T_KT", "CLOUD", "T", "WS10", "PHYS"]
FIXED = {"relative_humidity_2m": 75.0, "surface_pressure": 1013.0, "precipitation": 0.0}   # BESLISSINGEN #38


def ensure_token():
    """HF_TOKEN uit de gebruikersinstellingen van Windows halen als het proces hem (nog) niet kent."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            os.environ["HF_TOKEN"] = winreg.QueryValueEx(k, "HF_TOKEN")[0]
    except OSError:
        raise RuntimeError("HF_TOKEN ontbreekt: zie README (TAAK 5b) om toegang te krijgen.")


def read_metadata():
    """metadata.csv: ss_id, start/eind (UTC), lat/lon (afgerond ~1 km), orientation (°, 180 = zuid), tilt (°), kWp."""
    m = pd.read_csv(UKPV_RAW / "metadata.csv", parse_dates=["start_datetime_GMT", "end_datetime_GMT"])
    return m.set_index("ss_id")


def read_bad():
    """bad_data.csv: periodes die de makers als fout zien (eind leeg = tot het einde van de dataset)."""
    b = pd.read_csv(UKPV_RAW / "bad_data.csv", parse_dates=["start_datetime_GMT", "end_datetime_GMT"])
    b["end_datetime_GMT"] = b["end_datetime_GMT"].fillna(pd.Timestamp("2100-01-01", tz="UTC"))
    return b


def months(period):
    """Lijst (jaar, maand) van de periode."""
    return [(d.year, d.month) for d in pd.date_range(period[0], period[1], freq="MS")]


def _month(y, m):
    return pl.scan_parquet(f"{UKPV_HF}/year={y}/month={m:02d}/data.parquet")


def month_counts(period):
    """Aantal halve uren per systeem per maand (alleen de kolom ss_id lezen: snel)."""
    ensure_token()
    parts = []
    for y, m in months(period):
        c = _month(y, m).group_by("ss_id").agg(pl.len().alias("n")).collect().to_pandas()
        parts.append(c.assign(maand=f"{y}-{m:02d}"))
        print(f"  telling {y}-{m:02d}: {len(c)} systemen")
    return pd.concat(parts).pivot(index="ss_id", columns="maand", values="n")


def stream_systems(ids, period):
    """30-min-opwek van alleen deze systemen (polars filtert per bestand op ss_id) -> pandas, UTC zonder label."""
    ensure_token()
    ids = [int(i) for i in ids]
    parts = []
    for y, m in months(period):
        parts.append(_month(y, m).filter(pl.col("ss_id").is_in(ids)).collect().to_pandas())
        print(f"  opgehaald {y}-{m:02d}")
    df = pd.concat(parts, ignore_index=True)
    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df["P_KW"] = df["generation_Wh"].astype(float) * 2 / 1000
    return df.sort_values(["ss_id", "datetime_GMT"]).reset_index(drop=True)


def add_sun(df, meta):
    """Zonnehoogte (°) in het midden van elk halfuur (t - 15 min), per systeem op zijn eigen locatie."""
    out = []
    for sid, g in df.groupby("ss_id"):
        t = pd.DatetimeIndex(g["datetime_GMT"] - pd.Timedelta("15min")).tz_localize("UTC")
        sp = pvlib.solarposition.get_solarposition(t, meta.loc[sid, "latitude_rounded"], meta.loc[sid, "longitude_rounded"])
        out.append(pd.Series(sp["apparent_elevation"].to_numpy(), index=g.index))
    return df.assign(ELEV=pd.concat(out))


def quality(df, meta, kwp_limit=750):
    """Kwaliteit per systeem (aanbevolen in de README): negatief, > 750 x kWp, dagen met nachtopwek,
    dagen zonder opwek bij hoge zon. Geeft (tabel per systeem, vlag SLECHTE_DAG per rij)."""
    d = df.assign(DATE=df.datetime_GMT.dt.normalize(), KWP=df.ss_id.map(meta.kWp))
    sunny = d.ELEV > 20
    d = d.assign(NACHT=(d.ELEV < -5) & (d.P_KW > 0.01 * d.KWP), ZON_P=d.P_KW.where(sunny, 0.0), ZON_N=sunny,
                 NEG=d.generation_Wh < 0, HOOG=d.generation_Wh > kwp_limit * d.KWP)
    day = d.groupby(["ss_id", "DATE"]).agg(nacht=("NACHT", "any"), zon_som=("ZON_P", "sum"), zon_n=("ZON_N", "sum"))
    day["nul_bij_zon"] = (day.zon_n > 0) & (day.zon_som <= 0)
    day["slecht"] = day.nacht | day.nul_bij_zon
    q = pd.DataFrame({"negatief": d.groupby("ss_id").NEG.sum(), "te_hoog": d.groupby("ss_id").HOOG.sum(),
                      "dagen": day.groupby("ss_id").size(),
                      "dagen_nachtopwek": day.groupby("ss_id").nacht.sum(),
                      "dagen_nul_bij_zon": day.groupby("ss_id").nul_bij_zon.sum()})
    q["slechte_dagen_frac"] = (day.groupby("ss_id").slecht.sum() / q.dagen).round(4)
    q["P_max_per_kWp"] = (d.groupby("ss_id").P_KW.max() / meta.kWp.reindex(q.index)).round(3)
    q["jaaropbrengst_kWh_per_kWp"] = (d.groupby("ss_id").P_KW.sum() * 0.5 / meta.kWp.reindex(q.index)
                                      / (d.datetime_GMT.max() - d.datetime_GMT.min()).days * 365).round(0)
    bad_rows = d.set_index(["ss_id", "DATE"]).index.isin(day.index[day.slecht])
    return q, pd.Series(np.asarray(bad_rows), index=df.index)


# ---------- weer -> features per halfuur (TAAK 5b stap 3) ----------
def forecast_hourly(df, lead):
    """Previous-Runs-tabel -> uurtabel in ERA5-vorm voor 'lead' dagen eerder (_d1, _d2) of een Single Run.
    Vochtigheid, druk en regen staan niet in het archief -> vaste waarden (kleine invloed, geen echt weer)."""
    s = f"_d{lead}" if lead is not None else ""
    h = pd.DataFrame({"DATE_TIME": df["DATE_TIME"]})
    for v in ("shortwave_radiation", "diffuse_radiation", "temperature_2m", "wind_speed_10m", "cloud_cover"):
        h[v] = df[v + s] if v + s in df else np.nan
    h["wind_speed_10m"] = h["wind_speed_10m"].fillna(3.0)         # Single Runs: alleen als het model wind nodig heeft
    h["cloud_cover"] = h["cloud_cover"].fillna(50.0)
    return h.assign(**FIXED)


def features_30min(h, lat, lon, elev, shift, tilt, azimuth):
    """Uurweer -> 15 min (heldere-hemelindex) -> pvlib op het echte dak -> gemiddelde per halfuur (label = begin).
    Halve uren buiten de geldige uren van de bron worden NaN (geen verzonnen voorspellingen)."""
    from .physics import hourly_to_15min, panel_inputs
    h = h[h.shortwave_radiation.notna()]
    w = hourly_to_15min(h, lat, lon, elev, rad_shift_min=shift, tz="UTC")
    d = panel_inputs(w, np.zeros(len(w)), lat, lon, elev, tilt=tilt, azimuth=azimuth, tz="UTC")
    x = d.set_index("DATE_TIME")[FEAT].resample("30min").mean()
    hour = (x.index + pd.Timedelta("1h")).floor("h")                # uur waarvan het gemiddelde dit halfuur dekt
    x.loc[~hour.isin(pd.DatetimeIndex(h.DATE_TIME))] = np.nan
    return x


def chosen_systems():
    """De 40 gekozen systemen (stap 1)."""
    from .config import UKPV_RESULTS
    return pd.read_csv(UKPV_RESULTS / "stap1_systemen" / "gekozen_systemen.csv", index_col="ss_id")


def era5_system(chosen, sid):
    """ERA5 (uur, UTC) voor één systeem uit de cache -> (tabel, hoogte)."""
    from .config import UKPV_PERIOD, UKPV_WEATHER
    from .weather import load_era5
    return load_era5(chosen.latitude_rounded[sid], chosen.longitude_rounded[sid], *UKPV_PERIOD, tz="GMT",
                     folder=UKPV_WEATHER / "era5")


def model_frame(rad_shift, chosen):
    """Tabel per halfuur voor alle gekozen systemen: opwek + weer uit 3 bronnen (_E, _F1, _F2), COSZ, DAY, SLOT,
    MONTH, SEASON, OK. Gecachet per tijdverschuiving (BESLISSINGEN #38)."""
    from .config import UKPV_CLEAN, UKPV_PERIOD, UKPV_WEATHER, UKPV_DA_MODEL, UKPV_SEASONS, MIN_COSZ
    start, end = UKPV_PERIOD
    cache = UKPV_CLEAN / f"ukpv_features_shift{rad_shift}.parquet"
    if cache.exists():
        data = pd.read_parquet(cache)
    else:
        sel = pd.read_parquet(UKPV_CLEAN / "ukpv_30min.parquet")
        parts = []
        for i, sid in enumerate(chosen.index, 1):
            lat, lon = chosen.latitude_rounded[sid], chosen.longitude_rounded[sid]
            tilt, az = float(chosen.tilt[sid]), float(chosen.orientation[sid])
            h, elev = era5_system(chosen, sid)
            fc = pd.read_csv(UKPV_WEATHER / "dagvooruit" / f"{UKPV_DA_MODEL}_{sid}.csv", parse_dates=["DATE_TIME"])
            src = {"E": h, "F1": forecast_hourly(fc, 1), "F2": forecast_hourly(fc, 2)}
            x = pd.concat([features_30min(hh, lat, lon, elev, rad_shift, tilt, az).add_suffix(f"_{s}")
                           for s, hh in src.items()], axis=1)
            x = x.join(pv_30min(sel, sid), how="left")
            x["PLANT"], x["KWP"] = sid, chosen.kWp[sid]
            parts.append(x.reset_index().rename(columns={"index": "DATE_TIME"}))
            print(f"  features {i}/{len(chosen)}")
        data = pd.concat(parts, ignore_index=True)
        data.to_parquet(cache, index=False)
    data = data[(data.DATE_TIME >= start) & (data.DATE_TIME < pd.Timestamp(end) + pd.Timedelta("1D"))].reset_index(drop=True)
    data["CS_GHI"] = data.CS_GHI_W_E                               # heldere hemel hangt niet af van de bron
    cz = []                                                        # cos(zenit) in het midden van elk halfuur
    for sid, g in data.groupby("PLANT"):
        t = pd.DatetimeIndex(g.DATE_TIME + pd.Timedelta("15min")).tz_localize("UTC")
        z = pvlib.solarposition.get_solarposition(t, chosen.latitude_rounded[sid], chosen.longitude_rounded[sid]).apparent_zenith
        cz.append(pd.Series(np.cos(np.radians(z.to_numpy())).clip(0), index=g.index))
    data["COSZ"] = pd.concat(cz)
    data["DAY"] = (data.DATE_TIME.dt.normalize() - pd.Timestamp(start)).dt.days + 1
    data["SLOT"] = data.DATE_TIME.dt.hour * 2 + data.DATE_TIME.dt.minute // 30
    data["MONTH"] = data.DATE_TIME.dt.month
    data["SEASON"] = data.MONTH.map(UKPV_SEASONS)
    data["KT_SENSOR"] = np.nan                                     # geen weersensor bij huizen
    need = ["POA", "EFF_T", "EFF_T_KT", "CLOUD", "T", "WS10", "PHYS"]
    data["OK"] = ((data.COSZ > MIN_COSZ) & data.P_KW.notna()
                  & data[[f"{c}_{s}" for c in need for s in ("E", "F1", "F2")]].notna().all(axis=1))
    return data


def horizon_features(rad_shift, chosen):
    """Single Runs (horizon) -> features per halfuur per systeem en run (POA, EFF_T, EFF_T_KT, PHYS). Gecachet."""
    from .config import UKPV_CLEAN, UKPV_WEATHER
    cache = UKPV_CLEAN / f"ukpv_horizon_features_shift{rad_shift}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    files = sorted((UKPV_WEATHER / "horizon").glob("*.csv"))
    for j, f in enumerate(files, 1):
        r = pd.read_csv(f, parse_dates=["DATE_TIME", "RUN"])
        for sid, g in r.groupby("ID"):
            x = features_30min(forecast_hourly(g, None), chosen.latitude_rounded[sid], chosen.longitude_rounded[sid],
                               float(era5_system(chosen, sid)[1]), rad_shift, float(chosen.tilt[sid]),
                               float(chosen.orientation[sid]))
            parts.append(x[["POA", "EFF_T", "EFF_T_KT", "PHYS"]].dropna().assign(PLANT=sid, RUN=g.RUN.iloc[0])
                         .reset_index())
        print(f"  horizon-features run {j}/{len(files)}")
    hzf = pd.concat(parts, ignore_index=True)
    hzf.to_parquet(cache, index=False)
    return hzf


def horizon_frame(data, hzf, tr_mask, cols, make, train_src="F1"):
    """Horizon (TAAK 7): model getraind op tr_mask (inputs cols + '_' + train_src), toegepast op de Single Runs.
    Geeft per (systeem, run, halfuur) FRAC, DAG (1 = 0-24 uur na de run) en de voorspellingen P_MODEL, P_GEMDAG,
    P_PERS en P_SMART (persistence/smart persistence = laatste dag vóór de run). Alleen OK-halve uren."""
    from .config import CAPACITY_QUANTILE, UKPV_PERIOD
    from .forecast_features import add_clear, add_daily
    cap = data[tr_mask & data.OK].groupby("PLANT").P_KW.quantile(CAPACITY_QUANTILE)
    dd = data.assign(FRAC=data.P_KW / data.PLANT.map(cap))
    kc = add_clear(dd, dd[tr_mask & dd.OK])
    k_tr = kc[tr_mask & dd.OK].groupby("PLANT").K.mean()
    tr = dd[tr_mask & dd.OK]
    model = make().fit(tr[[f"{c}_{train_src}" for c in cols]].values, tr.FRAC)
    hz = hzf.merge(kc[["PLANT", "DATE_TIME", "FRAC", "FRAC_CS", "COSZ", "OK", "DAY", "SLOT"]], on=["PLANT", "DATE_TIME"])
    hz = hz[hz.OK].copy()
    hz["DAG"] = ((hz.DATE_TIME - hz.RUN).dt.total_seconds() // 86400).astype(int) + 1
    hz["P_MODEL"] = np.clip(model.predict(hz[cols].values), 0, None)
    hz["P_GEMDAG"] = hz.FRAC_CS * hz.PLANT.map(k_tr)
    run_day = (hz.RUN.dt.normalize() - pd.Timestamp(UKPV_PERIOD[0])).dt.days + 1
    s = dd.set_index(["PLANT", "DAY", "SLOT"]).FRAC
    hz["P_PERS"] = s.reindex(list(zip(hz.PLANT, run_day - 1, hz.SLOT))).fillna(0).values
    kd = add_daily(kc).groupby(["PLANT", "DAY"]).KD.first()
    hz["P_SMART"] = kd.reindex(list(zip(hz.PLANT, run_day - 1))).fillna(float(k_tr.mean())).values * hz.FRAC_CS
    return hz


def pv_30min(sel, sid):
    """Opwek van één systeem per halfuur met label = BEGIN van het halfuur (bron: einde), slechte dagen -> NaN."""
    g = sel[sel.ss_id == sid]
    p = pd.Series(g.P_KW.where(~g.SLECHTE_DAG).to_numpy(), index=g.datetime_GMT - pd.Timedelta("30min"), name="P_KW")
    return p[~p.index.duplicated()]

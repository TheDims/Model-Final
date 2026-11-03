# -*- coding: utf-8 -*-
"""
Natuurkunde met pvlib: van ERA5-weer naar inputs op het paneel (fase 3)
======================================================================
Overgenomen uit V5 solarhome/physics.py (hourly_to_15min, soiling_loss, simulate, PARAMS) en
V5 solarhome/dayahead.py (clearsky_poa, add_features), ingekort tot wat het eindmodel nodig heeft.

  1. uur -> kwartier via de heldere-hemelindex (Ineichen, met de hoogte), met tijdverschuiving rad_shift_min;
  2. zonnestand (NREL SPA), DNI uit GHI en DHI;
  3. straling op het schuine paneel (Perez) + reflectie (Fresnel) + spectrum (First Solar) + stof (Kimber);
  4. paneeltemperatuur (Faiman) uit luchttemperatuur, straling en wind;
  5. V4-inputs: CS_POA, KT_POA, EFF_T = effectieve straling x (1 + gamma (T_cel - 25)), EFF_T_KT.
Alle getallen die niet uit het weer komen zijn AANNAMES (PARAMS), zie V4/V5-README.
"""
import numpy as np
import pandas as pd
import pvlib

from .config import TZ

RHO0 = 1.225
PARAMS = {
    "gamma_pdc": -0.0037, "dc_ac_ratio": 1.1, "eta_inv": 0.96, "albedo": 0.2, "module_type": "monosi",
    "soiling_rate": 0.0015, "clean_mm": 6.0, "max_soiling": 0.3, "wind_height_factor": 0.79,
    "u0": 25.0, "u1": 6.84, "bifacial_gain": 0.0,
    "losses": dict(soiling=0, shading=0, snow=0, mismatch=2, wiring=2, connections=0.5, lid=1.5,
                   nameplate_rating=1, age=0, availability=0),
}


def air_density(temp_c, rh, p_hpa):
    tk = temp_c + 273.15
    e = rh / 100 * 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    return (p_hpa - e) * 100 / (287.05 * tk) + e * 100 / (461.5 * tk)


def _clearsky(times, lat, lon, elev):
    return pvlib.location.Location(lat, lon, TZ, altitude=elev).get_clearsky(times, model="ineichen")


def hourly_to_15min(hourly, lat, lon, elev, rad_shift_min=0):
    """Uurtabel -> kwartiertabel (kwartier t = interval [t, t+15 min)). De straling van uur h hoort bij het
    interval met midden h + rad_shift_min (bepaald in fase 3). Via de heldere-hemelindex geïnterpoleerd."""
    h = hourly.set_index("DATE_TIME").sort_index()
    centers = h.index + pd.Timedelta(minutes=rad_shift_min)
    subs = [centers + pd.Timedelta(minutes=m) for m in (-22.5, -7.5, 7.5, 22.5)]
    cs_h = sum(_clearsky(s.tz_localize(TZ), lat, lon, elev).ghi.values for s in subs) / 4
    kt = np.where(cs_h > 10, h.shortwave_radiation / np.maximum(cs_h, 1), np.nan)
    dfrac = np.where(h.shortwave_radiation > 10, h.diffuse_radiation / np.maximum(h.shortwave_radiation, 1), np.nan)
    rad = pd.DataFrame({"kt": np.clip(kt, 0, 1.2), "dfrac": np.clip(dfrac, 0, 1)}, index=centers)
    grid = pd.date_range(h.index.min().normalize(), h.index.max().normalize() + pd.Timedelta("23h45min"), freq="15min")
    mid = grid + pd.Timedelta(minutes=7.5)

    def interp(s):
        s = s.dropna()
        return np.interp(mid.asi8, s.index.asi8, s.values) if len(s) else np.full(len(mid), np.nan)

    cs = _clearsky(mid.tz_localize(TZ), lat, lon, elev)
    out = pd.DataFrame(index=grid)
    out.index.name = "DATE_TIME"
    out["CS_GHI_W"] = cs.ghi.values
    out["KT_W"] = interp(rad.kt)
    out["GHI_W"] = out.KT_W * out.CS_GHI_W
    out["DHI_W"] = np.minimum(interp(rad.dfrac) * out.GHI_W, out.GHI_W)
    for c, new in [("temperature_2m", "T"), ("relative_humidity_2m", "RH"), ("surface_pressure", "P_HPA"),
                   ("wind_speed_10m", "WS10"), ("cloud_cover", "CLOUD")]:
        out[new] = interp(h[c])
    # neerslag: hoeveelheid in het uur -> elk kwartier van dat uur (mm per uur)
    out["PRECIP"] = h.precipitation.reindex(grid, method="ffill").values
    sp = pvlib.solarposition.get_solarposition(mid.tz_localize(TZ), lat, lon, altitude=elev,
                                               pressure=out.P_HPA.values * 100, temperature=out["T"].values)
    out["ZENITH"], out["AZIMUTH"] = sp.apparent_zenith.values, sp.azimuth.values
    cosz = np.cos(np.deg2rad(out.ZENITH)).clip(0)
    dni = np.where(cosz > 0.05, (out.GHI_W - out.DHI_W) / np.maximum(cosz, 0.05), 0)
    out["DNI_W"] = np.clip(dni, 0, cs.dni.values * 1.1)
    out[["GHI_W", "DHI_W", "KT_W"]] = out[["GHI_W", "DHI_W", "KT_W"]].fillna(0)
    return out.reset_index()


def soiling_loss(hourly, grid, p=PARAMS):
    """Kimber: stofverlies per kwartier; regen >= 6 mm/dag spoelt schoon."""
    r = hourly.set_index("DATE_TIME").precipitation.fillna(0)
    r.index = r.index.tz_localize(TZ)
    s = pvlib.soiling.kimber(r, cleaning_threshold=p["clean_mm"], soiling_loss_rate=p["soiling_rate"],
                             max_soiling=p["max_soiling"])
    return s.tz_localize(None).reindex(grid, method="ffill").fillna(0).values


def panel_inputs(w15, soiling, lat, lon, elev, tilt=None, azimuth=180.0, p=PARAMS):
    """Kwartiertabel -> straling op het paneel en de V4-inputs (EFF_T, KT_POA, ...) + PHYS (pvlib, kW per kWp)."""
    d = w15.copy()
    times = (pd.DatetimeIndex(d.DATE_TIME) + pd.Timedelta(minutes=7.5)).tz_localize(TZ)
    tilt = abs(lat) if tilt is None else tilt
    d["RHO"] = air_density(d["T"], d.RH, d.P_HPA)
    am_rel = pvlib.atmosphere.get_relative_airmass(d.ZENITH.values)
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, d.P_HPA.values * 100)
    pw = pvlib.atmosphere.gueymard94_pw(d["T"].values, d.RH.values)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    irr = pvlib.irradiance.get_total_irradiance(tilt, azimuth, d.ZENITH.values, d.AZIMUTH.values, d.DNI_W.values,
                                                d.GHI_W.values, d.DHI_W.values, dni_extra=dni_extra, airmass=am_rel,
                                                albedo=p["albedo"], model="perez")
    aoi = pvlib.irradiance.aoi(tilt, azimuth, d.ZENITH.values, d.AZIMUTH.values)
    iam_d = pvlib.iam.marion_diffuse("physical", tilt)
    eff = (np.nan_to_num(irr["poa_direct"]) * pvlib.iam.physical(aoi)
           + np.nan_to_num(irr["poa_sky_diffuse"]) * iam_d["sky"] + np.nan_to_num(irr["poa_ground_diffuse"]) * iam_d["ground"])
    cosz = np.cos(np.deg2rad(d.ZENITH)).clip(0)
    spec = pvlib.spectrum.spectral_factor_firstsolar(pw, am_abs, module_type=p["module_type"])
    d["SPECTRAL"] = np.where(cosz > 0.02, np.nan_to_num(spec, nan=1.0), 1.0)
    d["SOILING"] = soiling
    d["POA"] = np.nan_to_num(irr["poa_global"])
    d["EFF_IRR"] = eff * d.SPECTRAL * (1 - d.SOILING)
    u1 = p["u1"] * np.sqrt(d.RHO / RHO0)
    d["T_CELL"] = pvlib.temperature.faiman(d.POA, d["T"], d.WS10 * p["wind_height_factor"], u0=p["u0"], u1=u1)
    pdc = pvlib.pvsystem.pvwatts_dc(d.EFF_IRR, d.T_CELL, pdc0=1000, gamma_pdc=p["gamma_pdc"])
    pdc = pdc * (1 - pvlib.pvsystem.pvwatts_losses(**p["losses"]) / 100)
    pac = pvlib.inverter.pvwatts(pdc, pdc0=1000 / p["dc_ac_ratio"], eta_inv_nom=p["eta_inv"])
    d["PHYS"] = np.clip(np.nan_to_num(pac), 0, None) / 1000
    # V4-inputs (overgenomen uit V5 dayahead.add_features)
    cs = _clearsky(times, lat, lon, elev)
    cs_irr = pvlib.irradiance.get_total_irradiance(tilt, azimuth, d.ZENITH.values, d.AZIMUTH.values, cs.dni.values,
                                                   cs.ghi.values, cs.dhi.values, dni_extra=dni_extra, airmass=am_rel,
                                                   albedo=p["albedo"], model="perez")
    d["CS_POA"] = np.nan_to_num(np.asarray(cs_irr["poa_global"], dtype=float))
    d["KT_POA"] = np.where(d.CS_POA > 20, d.POA / np.maximum(d.CS_POA, 1), 0).clip(0, 1.3)
    d["EFF_T"] = d.EFF_IRR * (1 + p["gamma_pdc"] * (d.T_CELL - 25))
    d["EFF_T_KT"] = d.EFF_T * d.KT_POA
    return d

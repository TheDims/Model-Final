# -*- coding: utf-8 -*-
"""
Eén tabel per kwartier per plant: opgeschoonde plantdata + tijd/zon + ERA5-weer en natuurkunde (fase 3+)
======================================================================================================
Zelfde stappen als in fase3_weermodel.py, op één plek zodat fase 4 (en later de app) ze kan hergebruiken.
"""
import pandas as pd

from .config import CLEAN_FILE, PLANTS, PLANT_LOCATIONS, MIN_COSZ
from .features import add_time, add_solar, add_clearsky
from .weather import load_era5
from .physics import hourly_to_15min, soiling_loss, panel_inputs

WEATHER_COLS = ["GHI_W", "CS_GHI_W", "KT_W", "T", "RH", "WS10", "CLOUD", "PRECIP", "POA", "EFF_T", "KT_POA",
                "EFF_T_KT", "T_CELL", "PHYS", "SOILING", "CS_POA"]


def weather_frame(rad_shift_min):
    """Plantdata + ERA5 + pvlib-inputs. OK = overdag, opwek bekend, geen dip, sensor en weer bekend
    (precies dezelfde kwartieren als in fase 3)."""
    clean = add_clearsky(add_solar(add_time(pd.read_csv(CLEAN_FILE, parse_dates=["DATE_TIME"]))))
    parts = []
    for p in PLANTS:
        lat, lon = PLANT_LOCATIONS[p]
        h, elev = load_era5(lat, lon)
        w = hourly_to_15min(h, lat, lon, elev, rad_shift_min=rad_shift_min)
        d = panel_inputs(w, soiling_loss(h, pd.DatetimeIndex(w.DATE_TIME)), lat, lon, elev)
        parts.append(d[["DATE_TIME"] + WEATHER_COLS].assign(PLANT=p))
    data = clean.merge(pd.concat(parts), on=["DATE_TIME", "PLANT"], how="left")
    data["OK"] = ((data.COSZ > MIN_COSZ) & data.AC_KW.notna() & ~data.DIP & data.IRRADIATION.notna()
                  & data[WEATHER_COLS].notna().all(axis=1))
    return data

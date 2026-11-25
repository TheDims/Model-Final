# -*- coding: utf-8 -*-
"""
Profiel en instellingen van de gebruiker
========================================
Alles staat in st.session_state en wordt bewaard in gebruiker/profiel.json.
Hier staan ook de keuzes uit 'Settings' en hoe die doorwerken in het natuurkundige model.

Bronnen voor de standaardwaarden (zie README):
  - ~100 sq ft (≈10 m²) dak per kW (CSE, Vikaspedia; SECI: 130 sq ft)
  - panelen van 500-550 W, mono PERC / TOPCon (installateurs in India)
  - temperatuurcoëfficiënt PERC ~-0,35 %/°C, TOPCon ~-0,30 %/°C (PVEL-scorecard 2024)
  - stof: 0,15 %/dag (Kimber, pvlib-standaard); stoffig: 0,4 %/dag (gemeten door IIT Bombay)
  - CO2 van het Indiase net ~0,7 kg/kWh; een boom neemt ~21 kg CO2 per jaar op (AANNAMES)
"""
import json
import math

import numpy as np
import streamlit as st

from appcore.config import PROFILE_FILE, USER
from appcore.i18n import LANGS

CITIES = {   # (breedtegraad, lengtegraad)
    "New Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27), "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.02, 72.57), "Pune": (18.52, 73.86), "Jaipur": (26.91, 75.79),
    "Jodhpur": (26.24, 73.02), "Lucknow": (26.85, 80.95), "Bhopal": (23.26, 77.41),
    "Nagpur": (21.15, 79.09), "Kochi": (9.93, 76.27), "Guwahati": (26.14, 91.74),
    "Shimla": (31.10, 77.17), "Leh (Ladakh)": (34.15, 77.58),
    "Anantapur (near Plant 1)": (14.68, 77.60), "Telangana (near Plant 2)": (19.0, 79.0),
}
OTHER = "other"
FACING = {"S": 180, "SE": 135, "SW": 225, "E": 90, "W": 270}
# paneeltype: temperatuurcoëfficiënt, rendement (voor oppervlak), spectrum-type, extra licht achterkant
TECH = {"perc": (-0.0035, 0.21, "monosi", 0.0), "topcon": (-0.0030, 0.22, "monosi", 0.0),
        "bifacial": (-0.0030, 0.215, "monosi", 0.05), "poly": (-0.0040, 0.17, "multisi", 0.0)}
# dak: Faiman-koeling (u0, u1). Open frame = pvlib-standaard; dichter op het dak = warmer (AANNAME)
ROOF = {"rcc_high": (25.0, 6.84), "rcc_low": (20.0, 5.0), "metal": (15.0, 3.0), "tile": (17.0, 4.0)}
SHADE = {"sh_none": 0, "sh_little": 3, "sh_some": 8}                  # % verlies (AANNAME)
CLEAN = {"cl_week": 7, "cl_2week": 14, "cl_month": 30, "cl_rain": None}
DUST = {"dust_normal": 0.0015, "dust_high": 0.004}                    # verlies per droge dag
PANEL_W = [330, 400, 450, 500, 540, 550, 575, 600]
SQFT_PER_KW = 100                                                     # vuistregel India
CO2_KG_PER_KWH, TREE_KG_PER_YEAR = 0.7, 21.0
THEMES = {   # naam: (accent, licht accent, achtergrond)
    "Saffron": ("#ee7d1a", "#f5b301", "#fdf6ec"),
    "Peacock blue": ("#0e6e7e", "#4fb3bf", "#f1f7f7"),
    "Leaf green": ("#3d8b3d", "#9bd27d", "#f4f8ef"),
    "Royal navy": ("#13294b", "#f7a01b", "#f5f6fa"),
}
PAGES = ["home", "forecast", "usage", "tips", "settings"]
HOME_WIDGETS = ["health", "next3", "impact", "consumption", "action", "plan"]
VIEWS = ["today", "week", "month"]
DEFAULTS = dict(
    page="home", lang="en", name="",
    # locatie en panelen
    city="New Delhi", lat_custom=22.0, lon_custom=79.0, size_mode="m_panels", n_panels=6,
    panel_w=500, kwp=3.0, roof_area=300.0, area_unit="sq ft", tech="perc", inv_type="grid",
    inv_auto=True, inv_kw=2.5, roof="rcc_high", facing="S", auto_tilt=True, tilt=28,
    shade="sh_none", clean="cl_2week", dust="dust_normal", age=0, tariff=7.0,
    # mijn huis (verbruik)
    monthly_units=300.0, has_ac=True, ac_setpoint=24,
    # app
    theme="Saffron", unit="kWh", big_text=False, default_view="today", widgets=list(HOME_WIDGETS),
    simple_view=True, energy_style="both")
CHOICES = {"page": PAGES, "lang": list(LANGS), "city": list(CITIES) + [OTHER],
           "size_mode": ["m_panels", "m_kw", "m_area"], "panel_w": PANEL_W,
           "area_unit": ["sq ft", "m²"], "tech": list(TECH), "inv_type": ["grid", "hybrid", "offgrid"],
           "roof": list(ROOF), "facing": list(FACING), "shade": list(SHADE), "clean": list(CLEAN),
           "dust": list(DUST), "theme": list(THEMES), "unit": ["kWh", "units"], "default_view": VIEWS,
           "energy_style": ["kwh", "examples", "both"]}


def init_state():
    """Profiel laden (eerste keer) en widget-waarden bewaren bij het wisselen van pagina."""
    USER.mkdir(exist_ok=True)
    if "loaded" not in st.session_state:
        saved = json.loads(PROFILE_FILE.read_text(encoding="utf-8")) if PROFILE_FILE.exists() else {}
        for k, v in DEFAULTS.items():
            val = saved.get(k, v)
            if k in CHOICES and val not in CHOICES[k]:       # oud/ongeldig profiel -> standaard
                val = v
            if k == "widgets":
                val = [w for w in val if w in HOME_WIDGETS]
            st.session_state[k] = val
        st.session_state.loaded = True
        page = st.query_params.get("page")             # bv. ?page=tips (bookmark / link delen)
        if page in PAGES:
            st.session_state.page = page
    for k in DEFAULTS:
        st.session_state[k] = st.session_state[k]
    return st.session_state


def save_profile():
    cfg = st.session_state
    PROFILE_FILE.write_text(json.dumps({k: cfg[k] for k in DEFAULTS if k != "page"},
                                       indent=2, ensure_ascii=False), encoding="utf-8")


def reset_profile():
    """Callback (draait vóór het opnieuw tekenen, anders mag je widgets niet wijzigen)."""
    cfg = st.session_state
    for k, v in DEFAULTS.items():
        if k != "page":
            cfg[k] = v
    for k in HOME_WIDGETS:
        cfg.pop(f"w_{k}", None)
    PROFILE_FILE.unlink(missing_ok=True)


# ---------- setup -> getallen voor het model ----------
def location():
    cfg = st.session_state
    if cfg.city in CITIES:
        return (*CITIES[cfg.city], cfg.city)
    return cfg.lat_custom, cfg.lon_custom, f"{cfg.lat_custom:.2f}°N, {cfg.lon_custom:.2f}°E"


def system():
    """Grootte uit de gekozen invoer: (kWp, aantal panelen, W per paneel, sq ft, m²)."""
    cfg = st.session_state
    w = cfg.panel_w
    if cfg.size_mode == "m_panels":
        n = int(cfg.n_panels)
        kwp = n * w / 1000
    elif cfg.size_mode == "m_kw":
        kwp = float(cfg.kwp)
        n = math.ceil(kwp * 1000 / w)
    else:
        sqft = cfg.roof_area if cfg.area_unit == "sq ft" else cfg.roof_area / 0.0929
        n = max(1, int(sqft / SQFT_PER_KW * 1000 / w))             # wat past er op het dak
        kwp = n * w / 1000
    sqft = kwp * SQFT_PER_KW
    return kwp, n, w, sqft, sqft * 0.0929


def overrides(kwp):
    """Eigen setup -> parameters van het natuurkundige model (appcore/physics.py)."""
    cfg = st.session_state
    gamma, _, module, bif = TECH[cfg.tech]
    u0, u1 = ROOF[cfg.roof]
    ratio = 1.1 if cfg.inv_auto else float(np.clip(kwp / max(cfg.inv_kw, 0.1), 0.8, 1.5))
    return {"gamma_pdc": gamma, "module_type": module, "u0": u0, "u1": u1,
            "bifacial_gain": bif if cfg.roof == "rcc_high" else bif * 0.4,   # minder licht onder laag frame
            "dc_ac_ratio": ratio, "wash_every_days": CLEAN[cfg.clean],
            "soiling_rate": DUST[cfg.dust], "shading": SHADE[cfg.shade], "age": 0.5 * cfg.age}


def setup():
    """(lat, lon, plaatsnaam, kWp, hoek, richting, overrides) van het huis."""
    cfg = st.session_state
    lat, lon, place = location()
    kwp = system()[0]
    tilt = int(round(lat)) if cfg.auto_tilt else cfg.tilt
    return lat, lon, place, kwp, tilt, FACING[cfg.facing], overrides(kwp)

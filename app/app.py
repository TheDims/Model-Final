# -*- coding: utf-8 -*-
"""
SolarHome Forecast - de app (Streamlit)
=======================================
Starten (in deze map):
    streamlit run app.py
    (of: C:\\Users\\dimme\\anaconda3\\python.exe -m streamlit run app.py)

Eerst stap 1d, 1e en 2 draaien (maken modellen/model_v5.joblib en modellen/meldingsregels.json).
Model V5: zelfde model als V4, gekozen trainingsvenster en band uit stap 1d/1e (zie README).
Internet nodig: het weer komt live van Open-Meteo.

REALTIME
  - Het weer van NU (temperatuur, bewolking) en de voorspelling worden elke 15 minuten
    opnieuw opgehaald; Home, Forecast en My Usage verversen zich dan vanzelf.
  - Het verleden (deze week/maand/jaar) wordt berekend met het weer zoals het WAS.

Pagina's (in ui/): Home, Forecast, My Usage, Tips, Settings.
Gebruikersdata staat lokaal in gebruiker/ (profiel.json, mijn_panelen.json).
"""
from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))       # hoofdmap van de repo: solarcore/ (model, natuurkunde, weer)
sys.path.insert(0, str(APP_DIR))              # app/: ui/ en appcore/
from appcore.forecast import in_india   # noqa: E402
from ui.state import init_state, location, PAGES   # noqa: E402
from ui.style import css, brand, html, t, HOUSE_ART   # noqa: E402
from ui.data import weather, REFRESH_S   # noqa: E402
from ui import home, forecast, usage, tips, settings   # noqa: E402

st.set_page_config(page_title="SolarHome Forecast", page_icon="☀️", layout="wide")
cfg = init_state()
css()

ICONS = {"home": ":material/home:", "forecast": ":material/bar_chart:", "usage": ":material/power:",
         "tips": ":material/eco:", "settings": ":material/settings:"}
with st.sidebar:
    html(brand())
    st.write("")
    st.radio("Menu", PAGES, key="page", label_visibility="collapsed",
             format_func=lambda p: f"{ICONS[p]}&nbsp;&nbsp;{t('nav_' + p)}")
    st.write("")
    if st.button(t("refresh"), width="stretch"):
        weather.clear()
    html(f'<div style="margin-top:30px">{HOUSE_ART}</div>'
         f'<div class="label" style="margin:2px 0 0 8px">{t("slogan")}</div>')

RENDER = {"home": home.render, "forecast": forecast.render, "usage": usage.render,
          "tips": tips.render, "settings": settings.render}


@st.fragment(run_every=REFRESH_S)
def live_page():
    """Pagina's met live data verversen zich elke 15 minuten vanzelf."""
    try:
        RENDER[cfg.page]()
    except FileNotFoundError as e:                  # model nog niet gemaakt (stap 1)
        st.error(str(e))
    except Exception as e:                          # geen internet / API-fout
        st.error(t("no_weather", e=e))


lat, lon, _ = location()
if cfg.page == "settings":
    settings.render()
elif not in_india(lat, lon):
    st.error(t("outside"))
else:
    live_page()

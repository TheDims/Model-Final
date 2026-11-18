# -*- coding: utf-8 -*-
"""
Settings: 1) mijn zonnepanelen  2) mijn huis (verbruik)  3) app aanpassen
=========================================================================
Alles telt mee in de berekening (zie ui/state.py) en wordt bewaard in gebruiker/profiel.json.
"""
import pandas as pd
import streamlit as st

from appcore.i18n import LANGS
from .state import (CHOICES, PANEL_W, THEMES, HOME_WIDGETS, VIEWS, OTHER, location, system,
                    save_profile, reset_profile)
from .style import t, html, header, ICON, theme
from .data import now


def render():
    cfg = st.session_state
    A, _, _, _ = theme()
    lat, lon, place = location()
    header(t("settings_title"), t("settings_sub"), ICON["gear"].replace('width="24" height="24"', 'width="58" height="58"'),
           place, now().date(), None)
    tab1, tab2, tab3 = st.tabs([t("tab_setup"), t("tab_home"), t("tab_app")])
    with tab1:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f"##### {t('sec_loc')}")
            st.text_input(t("name"), key="name", placeholder="e.g. Priya")
            st.selectbox(t("where"), CHOICES["city"], key="city", format_func=lambda c: t("other") if c == OTHER else c)
            if cfg.city == OTHER:
                st.number_input(t("lat"), 6.0, 37.5, step=0.01, format="%.2f", key="lat_custom")
                st.number_input(t("lon"), 68.0, 97.5, step=0.01, format="%.2f", key="lon_custom")
            lat, lon, _ = location()
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=4, height=200, color=A)
            st.markdown(f"##### {t('sec_panels')}")
            st.radio(t("size_how"), CHOICES["size_mode"], key="size_mode", horizontal=True, format_func=t)
            if cfg.size_mode == "m_panels":
                c1, c2 = st.columns(2)
                c1.number_input(t("n_panels"), 1, 60, step=1, key="n_panels")
                c2.selectbox(t("panel_w"), PANEL_W, key="panel_w", help=t("panel_w_h"))
            elif cfg.size_mode == "m_kw":
                st.slider(t("m_kw"), 1.0, 10.0, step=0.5, key="kwp", help=t("kw_h"))
                st.selectbox(t("panel_w"), PANEL_W, key="panel_w", help=t("panel_w_h"))
            else:
                c1, c2 = st.columns([2, 1])
                c1.number_input(t("area"), 20.0, 5000.0, step=10.0, key="roof_area", help=t("area_h"))
                c2.radio(t("area_unit"), CHOICES["area_unit"], key="area_unit")
                st.selectbox(t("panel_w"), PANEL_W, key="panel_w", help=t("panel_w_h"))
            kwp, n, w, sqft, m2 = system()
            html(f'<div class="card orangish" style="padding:10px 14px">{t("summary", kw=f"{kwp:.2f}", n=n, w=w, sqft=f"{sqft:,.0f}", m2=f"{m2:,.0f}")}</div>')
            st.selectbox(t("tech"), CHOICES["tech"], key="tech", format_func=t, help=t("tech_h"))
        with b:
            st.markdown(f"##### {t('sec_inv')}")
            st.selectbox(t("inv_type"), CHOICES["inv_type"], key="inv_type", format_func=t)
            st.toggle(t("inv_auto"), key="inv_auto")
            if not cfg.inv_auto:
                st.number_input(t("inv_kw"), 0.5, 15.0, step=0.5, key="inv_kw")
            st.markdown(f"##### {t('sec_roof')}")
            st.selectbox(t("roof"), CHOICES["roof"], key="roof", format_func=t, help=t("roof_h"))
            st.selectbox(t("facing"), CHOICES["facing"], key="facing", format_func=t, help=t("facing_h"))
            st.toggle(t("auto_tilt"), key="auto_tilt", help=t("auto_tilt_h", t=round(lat)))
            if not cfg.auto_tilt:
                st.slider(t("tilt"), 0, 45, key="tilt")
            st.markdown(f"##### {t('sec_care')}")
            st.radio(t("shade"), CHOICES["shade"], key="shade", format_func=t)
            st.selectbox(t("clean"), CHOICES["clean"], key="clean", format_func=t)
            st.radio(t("dust"), CHOICES["dust"], key="dust", format_func=t, horizontal=True, help=t("dust_h"))
            st.slider(t("age"), 0, 25, key="age", help=t("age_h"))
            st.markdown(f"##### {t('sec_money')}")
            st.number_input(t("price"), 1.0, 15.0, step=0.5, key="tariff", help=t("price_h"))
    with tab2:
        a, b = st.columns(2, gap="large")
        with a:
            st.number_input(t("monthly_units"), 30.0, 3000.0, step=10.0, key="monthly_units", help=t("monthly_units_h"))
            st.toggle(t("has_ac"), key="has_ac")
            if cfg.has_ac:
                st.slider(t("ac_setpoint"), 18, 30, key="ac_setpoint", help=t("ac_setpoint_h"))
        with b:
            html(f'<div class="card blueish">{t("home_explain")}</div>')
    with tab3:
        a, b = st.columns(2, gap="large")
        with a:
            st.selectbox("🌐 " + t("language"), CHOICES["lang"], key="lang", format_func=lambda x: LANGS[x])
            st.toggle(t("simple_view"), key="simple_view", help=t("simple_view_h"))
            st.radio(t("energy_style"), CHOICES["energy_style"], key="energy_style", horizontal=True,
                     format_func=lambda x: t("es_" + x), help=t("energy_style_h"))
            st.radio(t("theme"), CHOICES["theme"], key="theme", horizontal=True)
            html("".join(f'<span style="display:inline-block;width:28px;height:28px;border-radius:50%;'
                         f'background:{THEMES[x][0]};margin-right:8px;'
                         f'outline:{"3px solid #2a1f17" if x == cfg.theme else "none"};outline-offset:2px"></span>'
                         for x in THEMES))
            st.write("")
            st.radio(t("unit"), CHOICES["unit"], key="unit", horizontal=True, help=t("unit_h"),
                     format_func=lambda x: "kWh" if x == "kWh" else t("units"))
            st.toggle(t("big_text"), key="big_text")
            st.radio(t("chart_opens"), VIEWS, key="default_view", horizontal=True,
                     format_func=lambda v: t({"today": "today", "week": "this_week", "month": "this_month"}[v]))
        with b:
            st.markdown(f"**{t('show_home')}**")
            cfg.widgets = [k for k in HOME_WIDGETS
                           if st.checkbox(t("w_" + k), value=k in cfg.widgets, key=f"w_{k}")]
    st.write("")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button(t("save"), type="primary", width="stretch"):
        save_profile()
        st.success(t("saved"))
    c2.button(t("reset"), width="stretch", on_click=reset_profile)
    st.caption(t("local"))

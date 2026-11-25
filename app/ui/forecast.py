# -*- coding: utf-8 -*-
"""
Forecast: de komende 16 dagen (weersvoorspelling Open-Meteo) + details per dag
==============================================================================
Ook: wat beïnvloedt je panelen op deze plek, en hoe nauwkeurig is de voorspelling.
"""
import pandas as pd
import streamlit as st

from appcore.i18n import date_long
from .state import TECH
from .style import t, html, energy, examples, header, ICON, lang, unit_name, updated
from .data import bundle, model_info, best_window
from .charts import day_curve, day_bars


def render():
    cfg = st.session_state
    B = bundle()
    d, daily, meta, today, now = B["d"], B["daily"], B["meta"], B["today"], B["now"]
    fut = daily[daily.future]
    header(t("forecast_title"), t("forecast_sub"), ICON["bars"].replace('width="26" height="26"', 'width="58" height="58"'),
           B["place"], today, B["cur"])

    # tegels
    tomorrow = fut.iloc[1] if len(fut) > 1 else fut.iloc[0]
    week = fut.head(7)
    best = fut.kWh.idxmax()
    tom_sub = t("likely", a=f"{tomorrow.p10_kWh:.1f}", b=f"{tomorrow.p90_kWh:.1f}")
    if cfg.energy_style != "kwh":
        tom_sub = f"≈ {examples(tomorrow.kWh)}"
    tiles = [(t("tomorrow"), energy(tomorrow.kWh), tom_sub),
             (t("next_days", n=7), energy(week.kWh.sum(), 0), t("avg_day", x=f"{week.kWh.mean():.1f}", u=unit_name())),
             (t("next_days", n=len(fut)), energy(fut.kWh.sum(), 0), f"≈ ₹{fut.kWh.sum() * cfg.tariff:,.0f}"),
             (t("best_day"), date_long(pd.Timestamp(best), lang()).split(",")[0], energy(fut.kWh.max()))]
    for col, (lab, val, sm) in zip(st.columns(4), tiles):
        html(f'<div class="card"><div class="label">{lab}</div><div class="mid">{val}</div>'
             f'<div class="small">{sm}</div></div>', col)

    # eindmodel v2: hoe goed is de voorspelling al afgestemd op dit huis?
    cal = meta["calib"]
    if cal["fallback"]:
        st.warning(t("model_fallback"))
    st.caption(t("calib_on", n=cal["n_days"], f=f"{cal['f']:.2f}") if cal["n_days"] else t("calib_none"))

    # 16 dagen
    with st.container(key="panel_fc16"):
        rough = fut.index[fut.rough]
        start = date_long(pd.Timestamp(rough[0]), lang()).split(",")[0] if len(rough) else "–"
        html(f'<h4>{t("fc16_title", n=len(fut))}</h4><div class="small">{t("fc16_sub", d=start)}</div>')
        day_bars(fut, today, height=300)

    # één dag in detail
    with st.container(key="panel_fcday"):
        days = list(fut.index)
        sel = st.select_slider(t("pick_day"), options=days, value=days[1] if len(days) > 1 else days[0],
                               format_func=lambda x: date_long(pd.Timestamp(x), lang()), key="fc_day")
        x = d[d.DATE_TIME.dt.date == sel].set_index("DATE_TIME")
        r = daily.loc[sel]
        if r.rough:
            st.caption(f"⚠️ {t('rough_day')}")
        html(f'<div class="row" style="gap:24px"><div class="mid">{energy(r.kWh)}</div><div class="label">'
             f'{t("likely", a=f"{r.p10_kWh:.1f}", b=f"{r.p90_kWh:.1f}")} · 🌡️ {r.t_min:.0f}–{r.t_max:.0f} °C · '
             f'☁️ {r.cloud:.0f}% · 🌧️ {r.rain_prob:.0f}%</div></div>')
        day_curve(x, sel, now=now if sel == today else None, band=True, window=best_window(x.pred_kW), height=300)

    # wat beïnvloedt je panelen
    st.markdown(f"#### {t('factors')}")
    rho_pct = (meta["air_density"] / 1.225 - 1) * 100
    gamma = abs(TECH[cfg.tech][0]) * 100
    t0 = daily.loc[today]
    f = [(t("f_height"), f'{meta["elevation_m"]:.0f} m', t("f_height_t")),
         (t("f_press"), f'{meta["pressure_hpa"]:.0f} hPa', t("f_press_t")),
         (t("f_dens"), f'{meta["air_density"]:.2f} kg/m³', t("f_dens_t", p=f"{rho_pct:+.0f}")),
         (t("f_temp"), t("f_temp_v", t=f"{t0.t_cell_max:.0f}"), t("f_temp_t", g=f"{gamma:.2f}")),
         (t("f_dust"), f'−{meta["soiling_now_%"]:.1f}%', t("f_dust_t")),
         (t("f_rain"), f'{t0.rain_prob:.0f}%', t("f_rain_t", p=f"{fut.rain_prob.iloc[1]:.0f}")),
         (t("f_vap"), f'{meta["water_vapour_cm"]:.1f} cm', t("f_vap_t", p=f"{meta['spectral_%']:+.1f}")),
         (t("f_cloud"), f'{t0.cloud:.0f}%', t("f_cloud_t"))]
    for row in (f[:4], f[4:]):
        for col, (name, val, txt) in zip(st.columns(4), row):
            html(f'<div class="card"><div class="label">{name}</div><div style="font-size:21px;font-weight:700">'
                 f'{val}</div><div class="small">{txt}</div></div>', col)

    # hoe nauwkeurig: scores van eindmodel v2 (UK-test, resultaten/taak5/ukpv/eindmodel_v2/model_info_v2.json)
    info = model_info()
    da = info.get("dagvooruit", {})
    pick = lambda prefix: next((v for k, v in da.items() if k.startswith(prefix)), {})   # noqa: E731
    nan = float("nan")
    with st.expander(t("how")):
        st.markdown(t("how_body",
                      err=f"{pick('v2: + mengen').get('dagfout_%', nan):.0f}",
                      base=f"{pick('v2: alleen gemiddelde dag').get('dagfout_%', nan):.0f}",
                      cov=f"{info.get('band_dekking_%', nan):.0f}") + "\n" + t("how_16"))
    html(f'<div class="foot">{meta["source"]} · {updated(B["fetched"])} · {t("auto_refresh")}</div>')

# -*- coding: utf-8 -*-
"""Onderdelen die op meerdere pagina's terugkomen: statuskaart, navigeren, meetwaarden invullen."""
import pandas as pd
import streamlit as st

from appcore.config import PANELS_FILE
from appcore.alerts import dust_reminder, demo_log
from appcore.i18n import date_long
from .style import t, html, theme, lang, unit_name
from .data import save_readings, health, rules_info, now
from .charts import health_chart


def go_to(page):
    """Callback voor knoppen die naar een andere pagina gaan."""
    st.session_state.page = page


def status_card(s, meta=None, extra=""):
    """Statuskaart van de panelen (kleur per niveau) + eventueel stof-herinnering uit het model."""
    _, _, _, S = theme()
    colors = {"ok": "#3d8b3d", "clean": "#d68a00", "service": "#c0392b", "learn": "#0e6e7e", "nodata": "#8a7a6b"}
    lvl = s["level"]
    if lvl == "ok":
        title, body = t("st_ok_t"), t("st_ok_b", n=s["n_window"], r=f"{s['r_window'] * 100:.0f}")
    elif lvl == "clean":
        title, body = t("st_clean_t"), t("st_clean_b", n=s["n_window"], r=f"{s['r_window'] * 100:.0f}",
                                         d=s["dry_days"] if s["dry_days"] is not None else "45+")
    elif lvl == "service" and s["reason"] == "low_day":
        title, body = t("st_service_t"), t("st_service_low", date=date_long(pd.Timestamp(s["last_date"]), lang()),
                                           r=f"{s['r_last'] * 100:.0f}")
    elif lvl == "service":
        title, body = t("st_service_t"), t("st_service_wash", r=f"{s['r_window'] * 100:.0f}")
    elif lvl == "learn":
        title, body = t("st_learn_t"), t("st_learn_b", n=s["n"])
    else:
        title, body = t("st_nodata_t"), t("st_nodata_b")
    tip = ""
    if meta and lvl in ("ok", "nodata", "learn") and dust_reminder(meta["soiling_now_%"], meta.get("dry_days")):
        txt = t("dust_tip", d=meta.get("dry_days") or "45+", s=f'{meta["soiling_now_%"]:.1f}')
        tip = f'<div class="small" style="margin-top:6px">{txt}</div>'
    c = colors[lvl]
    more = f' <span class="small">{extra}</span>' if extra and lvl != "ok" else ""
    html(f'<div class="card" style="border-left:6px solid {c}"><div style="font-weight:700;font-size:'
         f'{19 * S:.0f}px;color:{c}">{title}</div><div>{body}{more}</div>{tip}</div>')


def readings_section(meta):
    """Gemeten opwek en verbruik invullen + gezondheidscheck van de panelen."""
    df, s, rules, data, exp = health()
    status_card(s, meta)
    if data.get("demo"):
        st.caption("🧪 " + t("demo"))
    if df is not None and len(df):
        st.markdown(f"**{t('hist')}**")
        health_chart(df, rules)
    today = now().date()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"##### {t('add_title')}")
        day = st.date_input(t("date"), value=today - pd.Timedelta(days=1), max_value=today, key="add_date")
        a, b = st.columns(2)
        sol = a.number_input(t("kwh_day", u=unit_name()), 0.0, 500.0, step=0.1, key="add_kwh")
        use = b.number_input(t("use_day", u=unit_name()), 0.0, 500.0, step=0.1, key="add_use",
                             help=t("use_day_h"))
        if st.button(t("add"), type="primary"):
            entry = {"solar": float(sol) if sol > 0 else None, "use": float(use) if use > 0 else None}
            data["log"][str(day)] = {k: v for k, v in entry.items() if v is not None}
            data["demo"] = False
            save_readings(data)
            st.toast(t("added"))
            st.rerun()
    with c2:
        up = st.file_uploader(t("upload"), type=["csv"])
        if up is not None and st.session_state.get("last_upload") != up.file_id:
            try:
                f = pd.read_csv(up)
                cols = {c.lower().strip(): c for c in f.columns}
                dcol = cols["date"]
                kcol = next(c for k, c in cols.items() if "kwh" in k or "solar" in k)
                ucol = next((c for k, c in cols.items() if "use" in k or "consumption" in k), None)
                n = 0
                for i, row in f.iterrows():
                    e = {"solar": row[kcol]} | ({"use": row[ucol]} if ucol else {})
                    e = {k: float(v) for k, v in e.items() if pd.notna(v)}
                    if e:
                        data["log"][str(pd.Timestamp(row[dcol]).date())] = e
                        n += 1
                data["demo"] = False
                save_readings(data)
                st.session_state.last_upload = up.file_id
                st.success(t("upload_ok", n=n))
            except Exception:
                st.error(t("upload_bad"))
        if st.button(t("cleaned_btn")):
            data["cleaned"] = sorted(set(data["cleaned"]) | {str(today)})
            save_readings(data)
            st.success(t("cleaned_ok"))
    with st.expander(t("demo")):
        sc = st.radio(" ", ["demo_healthy", "demo_dusty", "demo_fault"], format_func=t, horizontal=True,
                      label_visibility="collapsed", key="demo_scenario")
        d1, d2 = st.columns(2)
        if d1.button(t("demo_btn")):
            scen = sc.replace("demo_", "")
            log = demo_log(exp, scen)
            save_readings({"log": {str(k): {"solar": float(v)} for k, v in log.items()}, "cleaned": [], "demo": scen})
            st.rerun()
        if d2.button(t("clear")):
            PANELS_FILE.unlink(missing_ok=True)
            st.rerun()
    with st.expander(t("how_alert")):
        tst = rules_info().get("test_dag21_34", {})
        st.markdown(t("how_alert_body", s=f"{rules['t_service'] * 100:.0f}", c=f"{rules['t_clean'] * 100:.0f}",
                      det=f"{tst.get('detectie_%', float('nan')):.0f}", fa=f"{tst.get('vals_alarm_%', float('nan')):.0f}"))

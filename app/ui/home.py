# -*- coding: utf-8 -*-
"""
Home (concept: 'Good morning!')
===============================
Volgorde van boven naar beneden:
  1. MELDING (alleen als er iets te doen is): één zin in eenvoudige taal + één actie.
  2a. EENVOUDIGE WEERGAVE (standaard aan): één hoofdboodschap ("Vandaag is een goede
      zonnedag" + hoeveel energie, ook als herkenbare voorbeelden) en hooguit 3 onderdelen:
      beste moment, morgen (met waarschuwing voor zonarme dagen), grafiek van vandaag.
      Al het andere staat onder "Meer details" - er is niets weggegooid.
  2b. VOLLEDIG DASHBOARD (eenvoudige weergave uit, of onder "Meer details"): opwek
      vandaag/week/maand, komende 3 dagen, milieu-impact, verbruik, aanbevolen actie, plan.
Realtime: deze pagina ververst elke 15 minuten vanzelf (zie app.py).
"""
import pandas as pd
import streamlit as st

from appcore.alerts import dust_reminder
from appcore.i18n import time_str, weekday_short, day_month, date_long
from .state import CO2_KG_PER_KWH, TREE_KG_PER_YEAR
from .style import (t, html, energy, energy_main, energy_line, theme, header, svg_weather, day_icon, ICON,
                    lang, unit_name, mini_bars, updated, GREEN)
from .data import bundle, flows, health, measured, best_window, load_readings, save_readings
from .charts import day_curve, day_bars
from .common import go_to


# ---------- hulpfuncties ----------
def _period(daily, today, view):
    """Dagen van de gekozen periode (kalenderweek ma-zo / kalendermaand) die we kennen."""
    t0 = pd.Timestamp(today)
    if view == "week":
        start, end = t0 - pd.Timedelta(days=t0.weekday()), t0 + pd.Timedelta(days=6 - t0.weekday())
    elif view == "month":
        start, end = t0.replace(day=1), t0 + pd.offsets.MonthEnd(0)
    else:
        start = end = t0
    sel = daily[(daily.index >= start.date()) & (daily.index <= end.date())]
    return sel, start, end


def _verdict(kwh, ref):
    """Goede / gemiddelde / zonarme dag t.o.v. een zonnige dag op deze plek (90e percentiel)."""
    r = kwh / ref if ref else 1
    return "good" if r >= 0.8 else "avg" if r >= 0.55 else "poor"


def _window(d, today, now):
    """Zonnigste blok van 4 uur; is dat vandaag al voorbij, dan dat van morgen."""
    x = d[d.DATE_TIME.dt.date == today].set_index("DATE_TIME").pred_kW
    s, e = best_window(x)
    if now > e:
        x = d[d.DATE_TIME.dt.date == today + pd.Timedelta(days=1)].set_index("DATE_TIME").pred_kW
        s, e = best_window(x)
        return s, e, "tomorrow"
    return s, e, "today"


def _now_kw(d, now):
    v = d.loc[(d.DATE_TIME <= now) & (d.DATE_TIME > now - pd.Timedelta(minutes=15)), "pred_kW"]
    return float(v.iloc[0]) if len(v) and v.iloc[0] > 0.01 else None


# ---------- 1. melding bovenaan ----------
def _clean_done():
    data = load_readings()
    today = str(pd.Timestamp.now(tz="Asia/Kolkata").date())
    data["cleaned"] = sorted(set(data["cleaned"]) | {today})
    save_readings(data)
    st.session_state.cleaned_msg = True


def alert_top(meta):
    """Eén zin in eenvoudige taal + één actie. Alleen als er iets te doen is."""
    _, _, _, S = theme()
    try:
        s = health()[1]
    except Exception:
        s = {"level": "nodata"}
    if st.session_state.pop("cleaned_msg", False):
        st.success(t("al_cleaned"))
        return
    if s["level"] == "service":
        r = s["r_last"] if s["reason"] == "low_day" else s["r_window"]      # % van normaal
        msg, color = t("al_service" if s["reason"] == "low_day" else "al_service_wash", p=f"{r * 100:.0f}"), "#c0392b"
    elif s["level"] == "clean":
        msg, color = t("al_clean", p=f"{(1 - s['r_window']) * 100:.0f}"), "#d68a00"
    elif dust_reminder(meta["soiling_now_%"], meta.get("dry_days")):
        msg, color = t("al_dust", d=meta.get("dry_days") or "45+", s=f'{meta["soiling_now_%"]:.0f}'), "#b07a1a"
    else:
        return
    a, b = st.columns([3.2, 1])
    html(f'<div class="card" style="border-left:6px solid {color};padding:14px 18px;margin-bottom:0">'
         f'<div style="font-size:{19 * S:.0f}px;font-weight:600">{msg}</div></div>', a)
    with b:
        st.write("")
        if s["level"] == "service":
            with st.popover(t("al_service_btn"), width="stretch"):
                st.markdown(t("al_service_steps"))
        else:
            st.button(t("al_clean_btn"), on_click=_clean_done, width="stretch", type="primary")
    st.write("")


# ---------- 2a. eenvoudige weergave ----------
def simple(B, daily):
    cfg = st.session_state
    A, _, _, S = theme()
    d, today, now = B["d"], B["today"], B["now"]
    ref = daily.kWh.quantile(0.9)
    kwh = daily.kWh.get(today, 0)
    v = _verdict(kwh, ref)
    icon = {"good": "sun", "avg": "partly", "poor": "cloud"}[v]
    nk = _now_kw(d, now)
    live = f'<span class="live">⚡ {t("now")}: {nk:.1f} kW</span>' if nk else ""
    # hoofdboodschap
    html(f'<div class="card orangish hero" style="padding:22px 26px"><div class="row" style="gap:26px">'
         f'{svg_weather(icon, 96)}<div style="flex:1"><div class="title">{t("verdict_" + v)}</div>'
         f'<div style="margin-top:6px">{energy_main(kwh)}</div><div style="margin-top:6px">{live}</div>'
         f'</div></div></div>')
    # 1. beste moment  2. morgen
    c1, c2 = st.columns(2, gap="medium")
    s_, e_, when = _window(d, today, now)
    battery = cfg.inv_type in ("hybrid", "offgrid")
    icon = ICON["battery"] if battery else ICON["clock"].replace('width="24" height="24"', 'width="40" height="40"')
    html(f'<div class="card action"><div class="row" style="align-items:flex-start"><div>{icon}</div><div>'
         f'<h4>{t("best_now")}{"" if when == "today" else " (" + t("v_tomorrow") + ")"}</h4>'
         f'<div class="what">{time_str(s_, lang())} – {time_str(e_, lang())}</div>'
         f'<div style="font-size:{16 * S:.0f}px">{t("do_batt") if battery else t("do_grid")}</div></div></div></div>', c1)
    nxt = [x for x in daily.index if x > today][:3]
    tm = daily.loc[nxt[0]]
    vt = _verdict(tm.kWh, ref)
    low = [x for x in nxt if _verdict(daily.loc[x].kWh, ref) == "poor"]
    warn = ""
    if low:
        dname = date_long(pd.Timestamp(low[0]), lang())
        warn = (f'<div style="margin-top:8px;font-weight:600;color:#b0461b">'
                f'{t("low_ahead", d=dname, x=t("low_batt" if battery else "low_grid"))}</div>')
    html(f'<div class="card"><div class="row" style="gap:16px">{svg_weather(day_icon(tm.cloud, tm.rain_prob), 64)}'
         f'<div><h4>{t("tom_" + vt)}</h4><div style="font-size:{18 * S:.0f}px;font-weight:600">{energy_line(tm.kWh)}</div>'
         f'</div></div>{warn}</div>', c2)
    # 3. grafiek van vandaag
    with st.container(key="panel_simplechart"):
        html(f'<h4>{t("prod_today")}</h4>')
        x = d[d.DATE_TIME.dt.date == today].set_index("DATE_TIME")
        day_curve(x, today, now=now, band=False, window=best_window(x.pred_kW), height=240)


# ---------- 2b. volledig dashboard ----------
def full(B, daily):
    cfg = st.session_state
    A, _, _, S = theme()
    d, today, now = B["d"], B["today"], B["now"]
    show = set(cfg.widgets)
    left, right = st.columns([2.25, 1], gap="medium")
    with left:
        with st.container(key="panel_prod"):
            view = st.session_state.get("home_view") or cfg.default_view
            sel, start, end = _period(daily, today, view)
            total = sel.kWh.sum()
            a, b = st.columns([1.45, 1])
            with b:
                st.segmented_control("period", ["today", "week", "month"], default=cfg.default_view,
                                     key="home_view", label_visibility="collapsed",
                                     format_func=lambda v: t({"today": "today", "week": "this_week", "month": "this_month"}[v]))
            if view == "today":
                ref = daily.kWh.get(today - pd.Timedelta(days=1))
                nk = _now_kw(d, now)
                extra = f'<span class="live">⚡ {t("now")}: {nk:.1f} kW</span>' if nk else ""
                cmp_txt = t("vs_yesterday")
            else:
                n_days = len(sel)
                prev_sel = daily[(daily.index < start.date())].tail(n_days) if view == "week" else None
                ref = prev_sel.kWh.sum() if prev_sel is not None and len(prev_sel) == n_days else None
                extra = (f'<span class="small">{t("period_info", p=int((sel.index < today).sum()), f=int((sel.index > today).sum()), n=(end - start).days + 1)}</span>')
                cmp_txt = t("vs_last_week")
            cmp_html = ""
            if ref:
                ch = (total / ref - 1) * 100
                arrow = f'<span class="up">↑ {ch:.0f}%</span>' if ch >= 0 else f'<span class="down">↓ {abs(ch):.0f}%</span>'
                cmp_html = f'{arrow} <span class="subtitle">{cmp_txt}</span>'
            title = {"today": "prod_today", "week": "prod_week", "month": "prod_month"}[view]
            icon = svg_weather(day_icon(sel.cloud.mean(), sel.rain_prob.max()), 76) if view == "today" else ""
            html(f'<h4 style="font-size:{22 * S:.0f}px;margin:0">{t(title)}</h4>'
                 f'<div class="row" style="gap:30px"><div>{energy_main(total)}'
                 f'<div style="margin-top:4px;font-size:{20 * S:.0f}px">{cmp_html}</div></div>{icon}</div>'
                 f'<div style="margin-top:6px">{extra}</div>', a)
            if view == "today":
                x = d[d.DATE_TIME.dt.date == today].set_index("DATE_TIME")
                day_curve(x, today, now=now, band=False, window=best_window(x.pred_kW), height=300)
            else:
                day_bars(sel, today, height=300, fmt=(lambda ts: str(ts.day)) if view == "month" else None)
        st.write("")
        c1, c2 = st.columns(2, gap="medium")
        if "next3" in show:
            cells = ""
            for day in [x for x in daily.index if x > today][:3]:
                r = daily.loc[day]
                ts = pd.Timestamp(day)
                cells += (f'<div style="flex:1;text-align:center"><b>{weekday_short(ts, lang())}</b>'
                          f'<div class="small">{day_month(ts, lang())}</div>'
                          f'{svg_weather(day_icon(r.cloud, r.rain_prob), 46)}'
                          f'<div style="font-weight:700">{r.kWh:.1f} {unit_name()}</div></div>')
            html(f'<div class="card"><div class="row" style="gap:10px">{ICON["bars"]}<h4>{t("next3")}</h4></div>'
                 f'<div style="display:flex;margin-top:8px">{cells}</div></div>', c1)
        if "impact" in show:
            kg = total * CO2_KG_PER_KWH
            html(f'<div class="card"><div class="row" style="gap:10px">{ICON["leaf"]}<h4>{t("impact")}</h4></div>'
                 f'<div class="row" style="margin-top:8px">{ICON["trees"]}<div>'
                 f'<div class="mid">{kg:.1f} <span>kg CO₂</span></div>'
                 f'<div class="label">{t("avoided_" + view)}</div>'
                 f'<div class="label">{t("trees", n=f"{kg / TREE_KG_PER_YEAR:.1f}")}</div></div></div></div>', c2)
    with right:
        f, _ = flows(d[d.DATE_TIME.dt.date == today], B["readings"])
        use_today = f.use.sum() * 0.25
        solar_today = daily.kWh.get(today, 0)
        if "consumption" in show:
            html(f'<div class="card greenish"><div class="row" style="align-items:flex-start">'
                 f'<div style="background:#e3f2e3;border-radius:14px;padding:10px">{ICON["house"]}</div>'
                 f'<div style="flex:1"><h4>{t("est_use")}</h4><div class="mid">{energy(use_today)}</div>'
                 f'<div class="label">{t("est_use_sub")}</div></div>'
                 f'<div style="align-self:flex-end">{mini_bars(GREEN)}</div></div></div>')
        if "action" in show:
            s_, e_, when = _window(d, today, now)
            battery = cfg.inv_type in ("hybrid", "offgrid")
            html(f'<div class="card action"><div class="row" style="align-items:flex-start">'
                 f'<div>{ICON["battery"]}</div><div><h4>{t("rec_action")}</h4>'
                 f'<div class="what">{t("act_charge") if battery else t("act_run")}</div>'
                 f'<div style="font-size:{18 * S:.0f}px">{t("between")} <b style="color:{A}">'
                 f'{time_str(s_, lang())} – {time_str(e_, lang())}</b>'
                 f'{"" if when == "today" else " (" + t("v_tomorrow") + ")"}</div>'
                 f'<div class="label">{t("act_charge_sub") if battery else t("act_run_sub")}</div></div></div></div>')
        if "plan" in show:
            surplus = solar_today - use_today
            tip = (t("plan_surplus", x=f"{surplus:.1f}", u=unit_name()) if surplus > 0
                   else t("plan_short", p=f"{solar_today / max(use_today, 0.1) * 100:.0f}"))
            html(f'<div class="card plan"><div class="row"><div style="background:rgba(255,255,255,.18);'
                 f'border-radius:16px;padding:10px">{ICON["bolt"]}</div><div><h3>{t("plan_title")}</h3>'
                 f'<div class="small">{tip}</div></div></div></div>')
            st.button(t("see_tips") + " →", on_click=go_to, args=("tips",), width="stretch")


def render():
    cfg = st.session_state
    B = bundle()
    daily = B["daily"].copy()
    for day, v in measured(B["readings"], "solar").items():    # gemeten opwek gaat voor
        if day in daily.index and day < B["today"]:
            daily.loc[day, "kWh"] = v
    hour = B["now"].hour
    greet = t("good_morning" if hour < 12 else "good_afternoon" if hour < 17 else "good_evening")
    greet += f", {cfg.name}!" if cfg.name.strip() else "!"
    header(greet, t("home_sub"), svg_weather("sun", 66), B["place"], B["today"], B["cur"])
    if "health" in cfg.widgets:
        alert_top(B["meta"])
    if cfg.simple_view:
        simple(B, daily)
        with st.expander(t("more_details")):
            full(B, daily)
    else:
        full(B, daily)
    st.caption(f"{B['meta']['source']} · PV model: pvlib · {t('demo_note')}")
    html(f'<div class="foot">{updated(B["fetched"])} · {t("auto_refresh")}</div>')

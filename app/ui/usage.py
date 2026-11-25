# -*- coding: utf-8 -*-
"""
My Usage (concept: verbruik, uit zon / uit net, apparaten, trend, inzichten)
============================================================================
Verbruik is een SCHATTING (appcore/usage.py) tenzij de gebruiker meterstanden invult.
Opwek in het verleden = berekend met het weer zoals het was (of gemeten, als ingevuld).
Onderaan: meetwaarden invullen + gezondheidscheck van de panelen.
"""
import pandas as pd
import streamlit as st

from appcore.i18n import date_short, day_month, month_short, weekday_short
from .style import t, html, energy, header, ICON, lang, unit_name, mini_bars, theme, svg_weather, GREEN, BLUE
from .data import bundle, flows, year, measured
from .charts import usage_chart, donut, trend
from .common import readings_section

CAT_COLORS = {"cooling": "#f39a3b", "kitchen": "#4caf6a", "lighting": "#5b9be0", "water": "#38b6c9",
              "electronics": "#f2c94c", "other": "#c7c2bb"}


def _range(view, today, last_day):
    """(begin, eind, vorige begin, vorige eind). Week = ma-zo, maand = kalendermaand; dagen die nog
    komen zijn een schatting met de weersvoorspelling (tot de laatste voorspelde dag)."""
    t0 = pd.Timestamp(today)
    if view == "today":
        return t0, t0, t0 - pd.Timedelta(days=1), t0 - pd.Timedelta(days=1)
    if view == "week":
        s = t0 - pd.Timedelta(days=t0.weekday())
        return s, s + pd.Timedelta(days=6), s - pd.Timedelta(days=7), s - pd.Timedelta(days=1)
    if view == "month":
        s = t0.replace(day=1)
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(last_day))
        ps = (s - pd.Timedelta(days=1)).replace(day=1)
        return s, e, ps, min(ps + (e - s), s - pd.Timedelta(days=1))
    return t0.replace(month=1, day=1), t0, None, None


def _totals(f):
    return {k: f[k].sum() * 0.25 for k in ("use", "solar", "from_solar", "from_grid", "export")}


def render():
    A, LI, _, S = theme()
    B = bundle()
    d, today = B["d"], B["today"]
    header(t("usage_title"), t("usage_sub"), ICON["plug"].replace('width="40" height="40"', 'width="58" height="58"'),
           B["place"], today, B["cur"])

    top_l, top_r = st.columns([2, 1])
    with top_l:
        view = st.segmented_control("period", ["today", "week", "month", "year"], default="week", key="usage_view",
                                    label_visibility="collapsed",
                                    format_func=lambda v: t({"today": "today", "week": "this_week", "month": "this_month",
                                                             "year": "this_year"}[v])) or "week"
    start, end, pstart, pend = _range(view, today, d.DATE_TIME.max().date())
    html(f'<div style="text-align:right;padding-top:6px" class="label">📅 {day_month(start, lang())} – '
         f'{day_month(end, lang())} {end.year}</div>', top_r)

    # data voor deze periode en de vorige
    dd = d.assign(date=d.DATE_TIME.dt.date)
    if view == "year":
        with st.spinner(t("loading_year")):
            y = year(*B["key"])
        cur15 = pd.concat([y, d[d.DATE_TIME.dt.date == today]], ignore_index=True)
    else:
        cur15 = dd[(dd.date >= start.date()) & (dd.date <= end.date())]
    f, cats = flows(cur15, B["readings"])
    cur = _totals(f)
    prev = None
    if pstart is not None and pstart.date() >= dd.date.min():
        pf, _ = flows(dd[(dd.date >= pstart.date()) & (dd.date <= pend.date())], B["readings"])
        prev = _totals(pf)

    # drie tegels
    def change(k, good_down):
        if not prev or prev[k] <= 0:
            return '<span class="small">–</span>'
        ch = (cur[k] / prev[k] - 1) * 100
        good = (ch < 0) if good_down else (ch > 0)
        cls = "up" if good else "down"
        return (f'<span class="{cls}">{"↓" if ch < 0 else "↑"} {abs(ch):.0f}%</span> '
                f'<span class="small">{t("vs_prev_" + view)}</span>')
    tiles = [("greenish", ICON["house"], t("total_use"), cur["use"], change("use", True), GREEN),
             ("orangish", svg_weather("sun", 34), t("from_solar"), cur["from_solar"], change("from_solar", False), A),
             ("blueish", ICON["grid"], t("from_grid"), cur["from_grid"], change("from_grid", True), BLUE)]
    for col, (cls, ic, lab, val, ch, colr) in zip(st.columns(3), tiles):
        html(f'<div class="card {cls}"><div class="row" style="align-items:flex-start"><div>{ic}</div>'
             f'<div style="flex:1"><h4>{lab}</h4><div class="mid">{energy(val)}</div><div>{ch}</div></div>'
             f'<div style="align-self:flex-end">{mini_bars(colr)}</div></div></div>', col)
    measured_days = len(measured(B["readings"], "use"))
    note = t("use_estimated") if measured_days == 0 else t("use_partly_measured", n=measured_days)
    if end.date() > today:
        note += " " + t("incl_forecast")
    st.caption(note)

    # grafiek + verdeling per apparaat
    g1, g2 = st.columns([1.6, 1], gap="medium")
    with g1:
        with st.container(key="panel_usechart"):
            html(f'<h4>{t("use_vs_solar")}</h4>')
            if view == "today":
                agg = f.groupby(f.index.hour).sum() * 0.25
                lab = [f"{h:02d}:00" for h in agg.index]
            elif view == "year":
                agg = f.groupby(f.index.to_period("M")).sum() * 0.25
                lab = [month_short(p.to_timestamp(), lang()) for p in agg.index]
            else:
                agg = f.groupby(f.index.date).sum() * 0.25
                lab = [date_short(pd.Timestamp(x), lang()) if view == "week" else str(pd.Timestamp(x).day)
                       for x in agg.index]
            usage_chart(agg, lab, height=320)
    with g2:
        with st.container(key="panel_donut"):
            html(f'<h4>{t("appliances")}</h4>')
            per = cats.loc[(cats.index >= start.date()) & (cats.index <= end.date())].sum()
            tot = per.sum()
            a, b = st.columns([1, 1.15])
            with a:
                donut(per.values, [t("cat_" + c) for c in per.index], [CAT_COLORS[c] for c in per.index],
                      f"<b>{tot:.0f}</b><br>{unit_name()}", height=220)
            rows = "".join(f'<div class="row" style="gap:8px;margin:5px 0"><span style="width:10px;height:10px;'
                           f'border-radius:50%;background:{CAT_COLORS[c]};display:inline-block"></span>'
                           f'<span style="flex:1">{t("cat_" + c)}</span><b>{v / tot * 100:.0f}%</b>'
                           f'<span class="small" style="width:62px;text-align:right">{v:.1f}</span></div>'
                           for c, v in per.items())
            html(f'<div style="font-size:{14 * S:.0f}px;padding-top:10px">{rows}</div>', b)
            st.caption(t("appliances_src"))

    # trend + inzichten
    t1, t2 = st.columns([1.2, 1], gap="medium")
    with t1:
        with st.container(key="panel_trend"):
            html(f'<h4>{t("trend")}</h4>')
            if view == "year":
                labels, vals = lab, list(agg.use)
            else:                                      # laatste 4 kalenderweken (ma-zo), uit de afgelopen 45 dagen
                monday = pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).weekday())
                labels, vals = [], []
                for k in range(3, -1, -1):
                    ws = monday - pd.Timedelta(days=7 * k)
                    we = min(ws + pd.Timedelta(days=6), pd.Timestamp(today))
                    if ws.date() < dd.date.min():
                        continue
                    wf, _ = flows(dd[(dd.date >= ws.date()) & (dd.date <= we.date())], B["readings"])
                    labels.append(f"{day_month(ws, lang())} – {day_month(we, lang())}")
                    vals.append(wf.use.sum() * 0.25)
            trend(labels, vals, avg=sum(vals) / len(vals) if vals else None, height=260)
    with t2:
        with st.container(key="panel_insights"):
            html(f'<div class="row" style="gap:8px">{ICON["bulb"]}<h4>{t("insights")}</h4></div>')
            ins = []
            if prev and prev["use"] > 0:
                ch = (cur["use"] / prev["use"] - 1) * 100
                ins.append(("🌿", t("ins_less" if ch < 0 else "ins_more", p=f"{abs(ch):.0f}", w=t("prev_" + view)),
                            t("ins_total", x=f"{cur['use']:.1f}", u=unit_name())))
            if view == "today":
                h = f.use.groupby(f.index.hour).sum().idxmax()
                ins.append(("🕒", t("ins_peak_hour", h=f"{h:02d}:00"), t("ins_peak_hour_sub")))
            else:
                daily_use = f.use.groupby(f.index.date).sum() * 0.25
                top = daily_use.idxmax()
                lbl = (f"{weekday_short(pd.Timestamp(top), lang())} {day_month(pd.Timestamp(top), lang())}")
                ins.append(("🕒", t("ins_peak_day", d=lbl, x=f"{daily_use.max():.1f}", u=unit_name()),
                            t("ins_peak_day_sub")))
            share = cur["from_solar"] / cur["use"] * 100 if cur["use"] else 0
            ins.append(("🏠", t("ins_share", p=f"{share:.0f}"),
                        t("ins_export", x=f"{cur['export']:.1f}", u=unit_name())))
            for ic, ttl, sub in ins:
                html(f'<div class="insight"><div style="font-size:22px">{ic}</div><div><b>{ttl}</b>'
                     f'<div class="small">{sub}</div></div></div>')

    # historie downloaden (per dag: opwek, verbruik, uit zon, van net; gemeten of geschat)
    per_day = (f.groupby(f.index.date)[["solar", "use", "from_solar", "from_grid", "export"]].sum() * 0.25).round(2)
    per_day.index.name = "date"
    per_day["solar_measured"] = per_day.index.isin(measured(B["readings"], "solar").index)
    per_day["use_measured"] = per_day.index.isin(measured(B["readings"], "use").index)
    st.download_button(t("download"), per_day.to_csv().encode("utf-8"), file_name=f"solarhome_{view}_{today}.csv",
                       mime="text/csv", help=t("download_h"))

    st.markdown(f"### 🔧 {t('readings_title')}")
    readings_section(B["meta"])

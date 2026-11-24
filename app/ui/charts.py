# -*- coding: utf-8 -*-
"""Grafieken (plotly) in de stijl van de concepten, gedeeld door alle pagina's."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from appcore.i18n import date_short, time_str
from .style import t, theme, rgba, lang, unit_name, NAVY, BLUE

CFG = {"displayModeBar": False}


def finish(fig, height=320, legend=True):
    _, _, _, S = theme()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=legend, hovermode="x unified",
                      font=dict(family="Poppins, Noto Sans KR, Noto Sans Devanagari", color="#5a4c40", size=12 * S),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    fig.update_yaxes(gridcolor="#f0e8de", rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch", config=CFG)


def _time_axis(fig, day):
    twelve = lang() in ("en", "hi")
    t0 = pd.Timestamp(day)
    ticks = [t0 + pd.Timedelta(hours=h) for h in range(0, 25, 6)]
    text = (["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"] if twelve
            else [f"{h:02d}:00" for h in range(0, 25, 6)])
    fig.update_xaxes(range=[t0, t0 + pd.Timedelta(hours=24)], tickvals=ticks, ticktext=text,
                     hoverformat="%-I:%M %p" if twelve else "%H:%M", showgrid=True, gridcolor="#f5efe7")


def day_curve(x, day, now=None, band=True, window=None, height=320):
    """Vermogen over één dag (kwartieren): vloeiende curve met verloop, piek-label, band, 'nu'."""
    A, LI, _, _ = theme()
    fig = go.Figure()
    if band:
        fig.add_scatter(x=x.index, y=x.p90_kW, line=dict(width=0, shape="spline"), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=x.index, y=x.p10_kW, fill="tonexty", fillcolor=rgba(LI, .18), line=dict(width=0, shape="spline"),
                        name=t("likely_range"), hoverinfo="skip")
    fig.add_scatter(x=x.index, y=x.pred_kW, name=t("forecast"), fill="tozeroy",
                    fillgradient=dict(type="vertical", colorscale=[[0, rgba(A, .02)], [1, rgba(A, .38)]]),
                    line=dict(color=A, width=3, shape="spline", smoothing=0.6),
                    hovertemplate="%{y:.2f} kW<extra></extra>")
    if x.pred_kW.max() > 0:
        pk, pv = x.pred_kW.idxmax(), x.pred_kW.max()
        fig.add_shape(type="line", x0=pk, x1=pk, y0=0, y1=pv, line=dict(color=rgba(A, .5), dash="dot"))
        fig.add_scatter(x=[pk], y=[pv], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=12, color=A, line=dict(color="white", width=2)))
        fig.add_annotation(x=pk, y=pv, yshift=34, showarrow=False, align="left",
                           text=f'<span style="color:{A}">{time_str(pk, lang())}</span><br><b>{pv:.1f} kW</b>',
                           bgcolor="#fff7ef", bordercolor="#f8dcc0", borderpad=6)
        top = max(pv, x.p90_kW.max() if band else 0) * 1.35
        fig.update_yaxes(range=[0, top])
    if window is not None:
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor=LI, opacity=0.12, line_width=0)
    if now is not None and x.index.min() <= now <= x.index.max():
        fig.add_vline(x=now, line=dict(color=NAVY, dash="dot", width=1))
        fig.add_annotation(x=now, y=1, yref="paper", text=t("now"), showarrow=False, font=dict(color=NAVY))
    _time_axis(fig, day)
    fig.update_yaxes(title=t("power_kw"))
    finish(fig, height, legend=band)


def day_bars(daily, today, height=320, field="kWh", title=None, fmt=None):
    """Staaf per dag: verleden (berekend met het weer zoals het was), vandaag, voorspelling."""
    A, LI, _, _ = theme()
    u = unit_name()
    lab = [fmt(pd.Timestamp(d)) if fmt else date_short(pd.Timestamp(d), lang()) for d in daily.index]
    past = [d < today for d in daily.index]
    color = [A if d == today else (rgba(A, .55) if p else rgba(LI, .75)) for d, p in zip(daily.index, past)]
    fig = go.Figure(go.Bar(x=lab, y=daily[field], marker_color=color, marker_cornerradius=8,
                           marker_pattern_shape=["" if p or d == today else "/" for d, p in zip(daily.index, past)],
                           marker_pattern_fgcolor="rgba(255,255,255,.6)", name=t("forecast"),
                           customdata=[t("estimated") if p else t("forecast") for p in past],
                           hovertemplate="%{x}: %{y:.1f} " + u + " (%{customdata})<extra></extra>"))
    if "p10_kWh" in daily:
        fut = [not p for p in past]
        fig.add_scatter(x=[l_ for l_, f in zip(lab, fut) if f], y=daily.p90_kWh[fut], mode="markers",
                        marker=dict(symbol="line-ew-open", color="#a99c8e", size=14), hoverinfo="skip", showlegend=False)
    fig.update_yaxes(title=title or t("per_day", u=u))
    fig.update_layout(bargap=0.35)
    finish(fig, height, legend=False)


def usage_chart(df, xlab, height=330):
    """My Usage: lijnen opwek en verbruik, staven 'van het net' (zoals in het concept)."""
    A, _, _, _ = theme()
    u = unit_name()
    fig = go.Figure()
    fig.add_bar(x=xlab, y=df.from_grid, name=t("grid_usage"), marker_color=rgba(BLUE, .35), marker_cornerradius=6,
                hovertemplate="%{y:.1f} " + u + "<extra></extra>")
    fig.add_scatter(x=xlab, y=df.solar, name=t("solar_gen"), mode="lines+markers", line=dict(color=A, width=3, shape="spline"),
                    fill="tozeroy", fillcolor=rgba(A, .08), hovertemplate="%{y:.1f} " + u + "<extra></extra>")
    fig.add_scatter(x=xlab, y=df.use, name=t("total_use"), mode="lines+markers",
                    line=dict(color=NAVY, width=3, shape="spline"), hovertemplate="%{y:.1f} " + u + "<extra></extra>")
    fig.update_yaxes(title=f"{t('energy')} ({u})")
    fig.update_layout(bargap=0.4)
    finish(fig, height)


def donut(values, labels, colors, center, height=300):
    fig = go.Figure(go.Pie(values=values, labels=labels, hole=0.62, marker=dict(colors=colors),
                           sort=False, textinfo="none", hovertemplate="%{label}: %{value:.1f} (%{percent})<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(size=18))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config=CFG)


def trend(labels, values, highlight_last=True, avg=None, height=250):
    A, _, _, _ = theme()
    u = unit_name()
    colors = [rgba(BLUE, .3)] * (len(values) - 1) + [A if highlight_last else rgba(BLUE, .3)]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, marker_cornerradius=6, text=[f"{v:.0f}" for v in values],
                           textposition="outside", hovertemplate="%{x}: %{y:.1f} " + u + "<extra></extra>"))
    if avg is not None:
        fig.add_hline(y=avg, line=dict(color=BLUE, dash="dash", width=1),
                      annotation_text=f"{t('avg')} {avg:.0f} {u}", annotation_position="top left")
    fig.update_yaxes(title=f"{t('energy')} ({u})")
    finish(fig, height, legend=False)


def health_chart(df, rules):
    """Gemeten opwek tegen 'normaal voor dit huis' (laatste 30 dagen)."""
    A, _, _, _ = theme()
    u = unit_name()
    last = df.tail(30)
    normal = last.exp_kWh * float((df.raw / df.r).iloc[0])
    lab = [date_short(pd.Timestamp(x), lang()) for x in last.index]
    fig = go.Figure()
    fig.add_bar(x=lab, y=last.act, name=t("measured"), marker_cornerradius=6,
                marker_color=[A if v else "#cfc6bb" for v in last.valid],
                customdata=[t("cloudy_day") if not v else "" for v in last.valid],
                hovertemplate="%{x}: %{y:.1f} " + u + " %{customdata}<extra></extra>")
    fig.add_scatter(x=lab, y=normal, name=t("expected"), mode="lines+markers", line=dict(color=NAVY, width=2))
    fig.add_scatter(x=lab, y=normal * rules["t_clean"], name=f"{rules['t_clean'] * 100:.0f}%", mode="lines",
                    line=dict(color="#c0392b", width=1, dash="dot"), hoverinfo="skip")
    fig.update_yaxes(title=t("per_day", u=u))
    fig.update_layout(bargap=0.3)
    finish(fig, 300)

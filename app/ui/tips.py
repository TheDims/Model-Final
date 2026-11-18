# -*- coding: utf-8 -*-
"""
Tips (concept: uitgelichte tip, optimalisatiepotentieel, aanbevolen, snelle tips, seizoenen)
==========================================================================================
De getallen zijn berekend voor dit huis en de komende 7 dagen:
  - schoonmaken:   huidig stofverlies (stofmodel, of de meting via 'My Usage')
  - middaggebruik: extra eigen zonnestroom als pomp/geiser/wasmachine in het zonnigste blok
                   draaien (appcore/usage.shift_gain)
  - airco:         BEE: ~6 % minder stroom per °C hoger (4-5 % rond 24-26 °C)
Bronnen staan in de README.
"""
import pandas as pd
import streamlit as st

from appcore import usage as umodel
from appcore.i18n import time_str
from .style import (t, html, header, ICON, lang, unit_name, theme, illus_panel, illus_washer, illus_ac,
                    rgba, GREEN, NAVY)
from .data import bundle, flows, health, best_window
from .common import go_to

SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post", 11: "post"}
SEASON_STYLE = {"summer": ("☀️", "#fff3df"), "monsoon": ("🌧️", "#e8f1fb"), "post": ("🌤️", "#fdeee8"),
                "winter": ("❄️", "#eef3fb")}


def _numbers(B):
    """Alle getallen voor de tips (komende 7 dagen)."""
    cfg = st.session_state
    d, today = B["d"], B["today"]
    fut = d[(d.DATE_TIME.dt.date >= today) & (d.DATE_TIME.dt.date < today + pd.Timedelta(days=7))]
    f, cats = flows(fut, B["readings"])
    days = max(fut.DATE_TIME.dt.date.nunique(), 1)
    solar = f.solar.sum() * 0.25 / days
    selfuse = f.from_solar.sum() * 0.25 / days
    shift = umodel.shift_gain(fut.pred_kW.values, cats, fut.DATE_TIME) / days
    soil = B["meta"]["soiling_now_%"]
    try:
        s = health()[1]
    except Exception:
        s = {"level": "nodata"}
    if s["level"] == "clean":                          # gemeten: duidelijk minder dan normaal
        soil = max(soil, (1 - s["r_window"]) * 100)
    clean_gain = solar * soil / max(100 - soil, 1)     # extra opwek als de panelen schoon zijn
    use_ratio = selfuse / solar if solar else 0
    optimized = selfuse + shift + clean_gain * use_ratio
    cooling = cats.cooling.mean() if len(cats) else 0
    sp = cfg.ac_setpoint
    ac_save = cooling * (1 - 0.94 ** max(26 - sp, 0)) if cfg.has_ac else 0
    x = d[d.DATE_TIME.dt.date == today].set_index("DATE_TIME").pred_kW
    return dict(solar=solar, selfuse=selfuse, shift=shift, soil=soil, clean_gain=clean_gain,
                optimized=optimized, ac_save=ac_save, cooling=cooling, window=best_window(x), status=s)


def _optim_svg(cur, opt):
    """Twee staafjes 'nu' en 'geoptimaliseerd' met een pijl (zoals in het concept)."""
    h1 = 60 * cur / max(opt, 0.01)
    return (f'<svg width="130" height="100" viewBox="0 0 130 100">'
            f'<rect x="12" y="{88 - h1:.0f}" width="34" height="{h1:.0f}" rx="4" fill="#e3e0dc"/>'
            f'<rect x="78" y="28" width="34" height="60" rx="4" fill="{GREEN}"/>'
            f'<path d="M30 {78 - h1:.0f} Q55 20 90 14" stroke="{GREEN}" stroke-width="3" fill="none"/>'
            f'<polygon points="92,8 96,18 86,16" fill="{GREEN}"/></svg>')


def render():
    cfg = st.session_state
    A, LI, _, S = theme()
    B = bundle()
    header(t("tips_title"), t("tips_sub"), ICON["leaf"].replace('width="28" height="28"', 'width="62" height="62"'),
           B["place"], B["today"], B["cur"])
    N = _numbers(B)
    u = unit_name()
    w0, w1 = (time_str(x, lang()) for x in N["window"])

    # uitgelichte tips (carrousel)
    feats = [("clean", illus_panel(h=230), t("feat_clean_t"), t("feat_clean_b", s=f"{N['soil']:.1f}",
                                                                   g=f"{N['clean_gain']:.1f}", u=u), t("how_clean")),
             ("midday", illus_washer(230), t("feat_mid_t"), t("feat_mid_b", a=w0, b=w1, x=f"{N['shift']:.1f}", u=u),
              t("how_mid")),
             ("ac", illus_ac(230, f"{cfg.ac_setpoint}°C"), t("feat_ac_t"), t("feat_ac_b"), t("how_ac"))]
    if N["soil"] < 1 and N["status"]["level"] != "clean":   # nog schoon: middaggebruik eerst
        feats = feats[1:] + feats[:1]
    idx = st.session_state.get("tip_idx", 0) % len(feats)
    key, art, title, body, how = feats[idx]
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        with st.container(key="panel_feat"):
            a, b = st.columns([1, 1.25])
            html(f'<div class="kicker">{t("featured")}</div><div class="feat-title">{title}</div>'
                 f'<div style="margin:10px 0">{body}</div>', a)
            with a.popover(t("learn_how") + " →"):
                st.markdown(how)
            html(f'<div style="border-radius:18px;overflow:hidden">{art}</div>', b)
            c1, c2, c3 = b.columns([1, 1, 1])
            c1.button("‹", key="tip_prev", on_click=lambda: st.session_state.update(tip_idx=idx - 1))
            c2.markdown("<div style='text-align:center'>" + " ".join("●" if i == idx else "○" for i in range(len(feats)))
                        + "</div>", unsafe_allow_html=True)
            c3.button("›", key="tip_next", on_click=lambda: st.session_state.update(tip_idx=idx + 1))
    with right:
        pot = (N["optimized"] / N["selfuse"] - 1) * 100 if N["selfuse"] else 0
        rupees = (N["optimized"] - N["selfuse"]) * cfg.tariff * 30
        detail = t("optim_detail", a=f'{N["selfuse"]:.1f}', b=f'{N["optimized"]:.1f}', u=u, r=f"{rupees:,.0f}")
        html(f'<div class="card greenish"><div class="row" style="gap:10px">{ICON["bulb"]}<div><h4>{t("optim")}</h4>'
             f'<div class="small">{t("optim_sub")}</div></div></div>'
             f'<div class="row" style="justify-content:space-between;margin-top:10px"><div>'
             f'<div style="font-size:{44 * S:.0f}px;font-weight:800;color:{GREEN}">+{pot:.0f}%</div>'
             f'<div class="label">{t("optim_what")}</div></div>{_optim_svg(N["selfuse"], N["optimized"])}</div>'
             f'<div class="small">{detail}</div></div>')

    # aanbevolen + snelle tips
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        html(f'<div class="row" style="gap:10px;margin-top:6px">{ICON["target"]}<h3 style="margin:0">{t("recommended")}</h3></div>')
        cards = [(illus_panel(h=120), t("rec_clean_t"), t("rec_clean_b"), t("badge_gen", p=f"{N['soil']:.1f}")),
                 (illus_washer(120), t("rec_mid_t"), t("rec_mid_b", a=w0, b=w1),
                  t("badge_save", r=f"{N['shift'] * cfg.tariff * 30:,.0f}")),
                 (illus_ac(120), t("rec_ac_t"), t("rec_ac_b"),
                  t("badge_ac", x=f"{N['ac_save']:.1f}", u=u) if N["ac_save"] > 0.05 else t("badge_ac_ok"))]
        for col, (art, ttl, txt, badge) in zip(st.columns(3), cards):
            html(f'<div class="card" style="padding:0;overflow:hidden"><div>{art}</div><div style="padding:12px 14px">'
                 f'<h4>{ttl}</h4><div class="small" style="min-height:58px">{txt}</div>'
                 f'<div style="margin-top:8px"><span class="pill">{badge}</span></div></div></div>', col)
    with right:
        rows = [(ICON["clock"], t("q_sun", a=w0, b=w1)), ("🔌", t("q_standby")), ("⭐", t("q_star")),
                ("🧺", t("q_shift")), (ICON["gear"], t("q_check"))]
        items = "".join(f'<div class="list-row"><div style="width:26px;text-align:center">{ic}</div>'
                        f'<div style="flex:1">{tx}</div><div class="small">›</div></div>' for ic, tx in rows)
        html(f'<div class="card"><div class="row" style="gap:8px">{ICON["bulb"]}<h4>{t("quick")}</h4></div>{items}</div>')
        st.button(t("check_panels") + " →", on_click=go_to, args=("usage",), width="stretch")

    # seizoenen + wist je dat
    left, right = st.columns([2.2, 1], gap="medium")
    with left:
        html(f'<div class="row" style="gap:10px">{ICON["calendar"]}<h3 style="margin:0">{t("seasonal")}</h3></div>')
        now_season = SEASON_OF_MONTH[B["today"].month]
        for col, s in zip(st.columns(4), ["summer", "monsoon", "post", "winter"]):
            ic, bg = SEASON_STYLE[s]
            border = f"2px solid {A}" if s == now_season else "1px solid transparent"
            now_tag = f'<div class="small" style="margin-top:6px"><b>{t("now_season")}</b></div>' if s == now_season else ""
            html(f'<div class="season" style="background:{bg};border:{border}"><div class="row" style="gap:8px">'
                 f'<span style="font-size:26px">{ic}</span><b>{t("s_" + s)}</b></div>'
                 f'<div class="small" style="margin-top:4px">{t("s_" + s + "_months")}</div>'
                 f'<div style="font-size:{14 * S:.0f}px;margin-top:6px">{t("s_" + s + "_tip")}</div>'
                 f'{now_tag}</div>', col)
    with right:
        html(f'<div class="card greenish" style="background:linear-gradient(180deg,#f3faf3,{rgba(GREEN, .12)})">'
             f'<div class="row" style="gap:8px">{ICON["leaf"]}<h4>{t("dyk")}</h4></div>'
             f'<div style="margin-top:6px">{t("dyk_body")}</div><div class="small" style="margin-top:6px">{t("dyk_src")}</div></div>')
    html(f'<div class="foot" style="color:{NAVY}">{t("tips_note")}</div>')

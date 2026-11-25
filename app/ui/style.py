# -*- coding: utf-8 -*-
"""
Uiterlijk van de app: vertaalfunctie, kleuren/CSS, kop van elke pagina, iconen (SVG)
====================================================================================
Ontwerp volgt de concepten (Home, My Usage, Tips): warme achtergrond, witte kaarten met
ronde hoeken, oranje accent. Alle iconen en illustraties zijn eigen SVG's.
Indiase motieven (uit het eerste prototype, Model/app.py): toran (slinger van goudsbloemen en
mangobladeren) boven elke pagina, jali-patroon in de zijbalk, mandala's als watermerk in de
hoofdkaarten, een Indiaas huis met plat dak, watertank en kokospalm.
"""
from urllib.parse import quote

import pandas as pd
import streamlit as st

from appcore.i18n import t as _t, date_long, time_str
from appcore.equivalents import split, fmt_count, DEFAULT_PAIR, BATTERY_PAIR
from .state import THEMES

NAVY, GREEN, BLUE, INK, MUTED = "#13294b", "#3d8b3d", "#4a8fe0", "#2a1f17", "#6e6156"
ORANGE, MARIGOLD, PEACOCK = "#ee7d1a", "#f5b301", "#0e6e7e"     # goudsbloem, pauwblauw
# hoofdkaarten met een mandala als watermerk (st.container(key=...))
MANDALA_PANELS = ["panel_prod", "panel_simplechart", "panel_fc16", "panel_usechart", "panel_feat"]


# ---------- tekst ----------
def lang():
    return st.session_state.get("lang", "en")


def t(key, **kw):
    return _t(key, lang(), **kw)


def unit_name():
    return "kWh" if st.session_state.unit == "kWh" else t("units")


def energy(x, digits=1):
    """Bv. '24.5 <span>kWh</span>' (of 'units', zoals op de Indiase stroomrekening)."""
    return f"{x:,.{digits}f} <span>{unit_name()}</span>"


def examples(kwh):
    """Bv. 'enough for 12 washing machine loads + 8 h of water pump' (appcore/equivalents.py).
    Met een batterij: uren ventilatoren + lampen bij stroomuitval."""
    battery = st.session_state.inv_type in ("hybrid", "offgrid")
    parts = [t("eq_" + k, n=fmt_count(n)) for k, n in split(kwh, BATTERY_PAIR if battery else DEFAULT_PAIR)]
    return t("enough_for", x=" + ".join(parts)) if parts else ""


def energy_main(kwh):
    """Grote weergave van een hoeveelheid energie, volgens de instelling (kWh / voorbeelden / beide)."""
    style = st.session_state.energy_style
    if style == "kwh":
        return f'<div class="big">{energy(kwh)}</div>'
    if style == "examples":
        return f'<div class="mid">{examples(kwh)}</div>'
    return f'<div class="big">{energy(kwh)}</div><div class="subtitle">≈ {examples(kwh)}</div>'


def energy_line(kwh):
    """Korte weergave in één regel, volgens de instelling."""
    style = st.session_state.energy_style
    if style == "kwh":
        return energy(kwh)
    return examples(kwh) if style == "examples" else f"{energy(kwh)} · {examples(kwh)}"


def html(s, target=st):
    target.markdown(s, unsafe_allow_html=True)


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def theme():
    """(accent, licht accent, achtergrond, tekstschaal)."""
    a, l_, bg = THEMES[st.session_state.theme]
    return a, l_, bg, 1.15 if st.session_state.big_text else 1.0


def mandala(color, opacity=1.0):
    """Eenvoudige mandala (twee kransen bloemblaadjes), gebruikt als watermerk."""
    outer = "".join(f'<path d="M50 3C58 13 58 23 50 30C42 23 42 13 50 3Z" transform="rotate({a} 50 50)"/>'
                    for a in range(0, 360, 30))
    inner = "".join(f'<path d="M50 20C55 27 55 33 50 38C45 33 45 27 50 20Z" transform="rotate({a} 50 50)"/>'
                    for a in range(15, 375, 30))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><g fill="none" '
            f'stroke="{color}" stroke-width="1" opacity="{opacity}">{outer}{inner}'
            f'<circle cx="50" cy="50" r="9"/><circle cx="50" cy="50" r="5"/>'
            f'<circle cx="50" cy="50" r="41"/></g></svg>')


def jali(color):
    """Jali: Indiaas raampatroon van overlappende cirkels (tegel voor de zijbalk)."""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='44' height='44'><g fill='none' "
            f"stroke='{color}' stroke-opacity='.10' stroke-width='1.2'><circle cx='0' cy='0' r='15.5'/>"
            f"<circle cx='44' cy='0' r='15.5'/><circle cx='0' cy='44' r='15.5'/>"
            f"<circle cx='44' cy='44' r='15.5'/><circle cx='22' cy='22' r='15.5'/></g></svg>")


def data_uri(svg):
    return "data:image/svg+xml;utf8," + quote(svg)


def svg_toran():
    """Toran: slinger van goudsbloemen en mangobladeren boven de deur (feestelijk welkom)."""
    A, LI, _, _ = theme()
    parts = ['<path d="M0 3H1200" stroke="#c9822b" stroke-width="2"/>']
    for i, x in enumerate(range(15, 1200, 30)):
        if i % 2 == 0:
            kleur = A if i % 4 == 0 else LI
            parts.append(f'<line x1="{x}" y1="3" x2="{x}" y2="13" stroke="#c9822b"/>'
                         f'<circle cx="{x}" cy="19" r="7" fill="{kleur}"/><circle cx="{x}" cy="19" r="2.8" fill="#fde7a6"/>')
        else:
            parts.append(f'<path d="M{x} 3C{x + 6} 10 {x + 4} 19 {x} 25C{x - 4} 19 {x - 6} 10 {x} 3Z" fill="#4f9a3f"/>')
    return (f'<div class="toran"><svg width="100%" height="30" viewBox="0 0 1200 30" '
            f'preserveAspectRatio="xMidYMin slice">{"".join(parts)}</svg></div>')


def css():
    A, LI, BG, S = theme()
    panels = ",".join(f".st-key-{k}" for k in MANDALA_PANELS)
    html(f"""<style>
/* Indiase motieven: jali in de zijbalk, mandala als watermerk, toran bovenaan */
section[data-testid="stSidebar"]{{background-color:#fffaf4;background-image:url("{data_uri(jali(A))}")}}
{panels}{{background-image:url("{data_uri(mandala(A, .13))}")!important;background-repeat:no-repeat!important;
background-size:320px!important;background-position:right -100px top -110px!important}}
.plan{{background:url("{data_uri(mandala('#ffffff', .18))}") no-repeat right -60px bottom -70px/180px,
linear-gradient(135deg,#0b5566,{PEACOCK} 55%,#178f8a)!important;position:relative;overflow:hidden}}
.hero{{background:url("{data_uri(mandala(A, .14))}") no-repeat right -90px top -100px/300px,#fff6ec!important}}
.toran{{margin:-0.6rem -1rem 10px -1rem;line-height:0}}
</style>""")
    html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;700&family=Noto+Sans+Devanagari:wght@400;700&display=swap');
.stApp{{background:{BG};font-family:'Poppins','Noto Sans KR','Noto Sans Devanagari',sans-serif;font-size:{16 * S:.0f}px;color:{INK}}}
header[data-testid="stHeader"]{{background:transparent}}
.block-container{{padding-top:1.4rem;max-width:1320px}}
section[data-testid="stSidebar"]{{border-right:1px solid #f1e6da}}
section[data-testid="stSidebar"] div[role="radiogroup"]{{gap:4px}}
section[data-testid="stSidebar"] div[role="radiogroup"] label{{padding:11px 16px;border-radius:14px;width:100%;margin:0;cursor:pointer}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{{display:none}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{{background:{rgba(A, .08)}}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){{background:{rgba(A, .18)};box-shadow:inset 4px 0 0 {A}}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{{color:{A};font-weight:600}}
section[data-testid="stSidebar"] div[role="radiogroup"] p{{font-size:16px}}
.card{{background:#fffdf9;border:1px solid #f3e4cf;border-radius:20px;padding:18px 20px;box-shadow:0 2px 12px rgba(150,80,10,.06);height:100%;margin-bottom:14px}}
.card h4{{margin:0 0 4px 0;font-size:{16 * S:.0f}px;font-weight:600;color:{INK};padding:0}}
[class*="st-key-panel"]{{background-color:#fffdf9;border:1px solid #f3e4cf!important;border-radius:20px;padding:18px 20px 8px 20px;box-shadow:0 2px 12px rgba(150,80,10,.06)}}
.row{{display:flex;align-items:center;gap:14px}}
.label{{color:{MUTED};font-size:{14 * S:.0f}px}}
.big{{font-size:{50 * S:.0f}px;font-weight:700;line-height:1.05;letter-spacing:-1px;color:{INK}}}
.big span{{font-size:{28 * S:.0f}px;font-weight:600;letter-spacing:0}}
.mid{{font-size:{30 * S:.0f}px;font-weight:700;line-height:1.15;color:{INK}}}
.mid span{{font-size:{18 * S:.0f}px;font-weight:600}}
.small{{color:#8a7a6b;font-size:{13 * S:.0f}px}}
.up{{color:{GREEN};font-weight:700}}.down{{color:#c0392b;font-weight:700}}
.title{{font-size:{38 * S:.0f}px;font-weight:700;line-height:1.1;color:{INK}}}
.subtitle{{font-size:{18 * S:.0f}px;color:#5a4c40}}
.hdr{{display:flex;align-items:center;justify-content:flex-end;gap:20px}}
.hdr .sep{{width:1px;height:42px;background:#eadfd2}}
.avatar{{width:46px;height:46px;border-radius:50%;background:#efe9e2;display:flex;align-items:center;justify-content:center}}
.live{{display:inline-block;background:{rgba(A, .1)};color:{A};border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}}
.pill{{display:inline-block;background:{rgba(GREEN, .1)};color:{GREEN};border-radius:10px;padding:4px 12px;font-size:{13 * S:.0f}px;font-weight:600}}
.action{{background:#fff3e6;border:1px solid #fbd9b8}}
.action .what{{font-size:{32 * S:.0f}px;font-weight:700;color:{A};line-height:1.1}}
.plan{{border:none;color:#fff}}
.plan h3{{color:#fff;margin:0;font-size:{22 * S:.0f}px;font-weight:600;padding:0}}
.plan .small{{color:#dff1f2}}
.greenish{{background:#f3faf3;border:1px solid #dcefdc}}
.blueish{{background:#f2f7fd;border:1px solid #dce9f7}}
.orangish{{background:#fff6ec;border:1px solid #fbe1c6}}
.kicker{{color:{A};font-weight:700;letter-spacing:1px;font-size:{13 * S:.0f}px}}
.feat-title{{font-size:{36 * S:.0f}px;font-weight:800;line-height:1.1;color:{INK}}}
.list-row{{display:flex;align-items:center;gap:12px;padding:10px 4px;border-bottom:1px solid #f3ebe2}}
.list-row:last-child{{border-bottom:none}}
.season{{border-radius:16px;padding:14px;height:100%}}
.insight{{display:flex;gap:12px;align-items:flex-start;background:#f7faf5;border-radius:12px;padding:10px 12px;margin-bottom:8px}}
.foot{{color:#a08f7e;font-size:12px;margin-top:6px}}
</style>""")


# ---------- iconen en illustraties ----------
def svg_logo(size=56):
    """Ons logo (zon + huis met zonnepanelen + stijgende pijl), nagetekend als SVG."""
    rays = "".join(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}"/>' for a, b, c, d in
                   [(38, 3, 38, 9), (23, 10, 27, 14), (53, 10, 49, 14), (16, 25, 22, 25), (21, 40, 26, 36)])
    cells = "".join(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}"/>' for a, b, c, d in
                    [(30, 41, 52, 41), (27, 46, 49, 46), (38, 35, 31, 52), (46, 35, 39, 52)])
    return (f'<svg width="{size}" height="{size * 0.8:.0f}" viewBox="0 0 100 80">'
            f'<g stroke="#f7a01b" stroke-width="3.5" stroke-linecap="round">{rays}</g>'
            f'<path d="M26 30a12 12 0 0 1 24 0Z" fill="#f7a01b"/>'
            f'<polygon points="23,52 46,52 63,35 40,35" fill="#0b3a73"/>'
            f'<g stroke="#e8edf5" stroke-width="1">{cells}</g>'
            f'<g fill="none" stroke="{NAVY}" stroke-width="4">'
            f'<path d="M47 54 L65 35 L81 52"/><path d="M27 62V55H49V62"/><path d="M76 48V62"/></g>'
            f'<path d="M65 41 L70 46 L80 35" fill="none" stroke="{NAVY}" stroke-width="3.5"/>'
            f'<polygon points="76,33 85,29 83,38" fill="{NAVY}"/>'
            f'<g fill="{NAVY}"><rect x="58" y="49" width="4" height="4"/><rect x="63" y="49" width="4" height="4"/>'
            f'<rect x="58" y="54" width="4" height="4"/><rect x="63" y="54" width="4" height="4"/></g></svg>')


def brand():
    return (f'<div style="text-align:center">{svg_logo(140)}<div style="font-weight:800;font-size:26px;line-height:1">'
            f'<span style="color:{NAVY}">Solar</span><span style="color:#f7a01b">Home</span></div>'
            f'<div style="color:{NAVY};font-size:15px;letter-spacing:3px">— Forecast —</div>'
            f'<div style="color:{NAVY};font-size:10px;letter-spacing:1px">Powering Brighter Days</div></div>')


# Indiaas huis: plat dak met zonnepanelen en zwarte watertank, boogdeur, kokospalm, goudsbloemen
HOUSE_ART = f'''<svg viewBox="0 0 240 160" width="100%">
<circle cx="184" cy="58" r="28" fill="#fcd9a0"/>
<path d="M0 120 Q60 84 120 106 T240 96 V160 H0Z" fill="#fbe0bd"/>
<path d="M0 140 Q120 124 240 138 V160 H0Z" fill="#b9d98f"/>
<rect x="80" y="86" width="100" height="52" fill="#f6c98e"/>
<rect x="76" y="80" width="108" height="8" rx="1" fill="#e59a52"/>
<polygon points="86,80 116,80 122,64 92,64" fill="#24406e"/><polygon points="120,80 150,80 156,64 126,64" fill="#24406e"/>
<line x1="104" y1="64" x2="101" y2="80" stroke="#5d7bb0"/><line x1="138" y1="64" x2="135" y2="80" stroke="#5d7bb0"/>
<rect x="160" y="62" width="16" height="18" rx="2" fill="#2b2b2b"/><ellipse cx="168" cy="62" rx="8" ry="2.5" fill="#4a4a4a"/>
<path d="M97 138V119a9 9 0 0 1 18 0V138Z" fill="#8a4b2a"/>
<path d="M134 110V100a8 8 0 0 1 16 0V110Z" fill="{PEACOCK}"/><path d="M157 110V100a8 8 0 0 1 16 0V110Z" fill="{PEACOCK}"/>
<g>{"".join(f'<circle cx="{x}" cy="111" r="1.8" fill="{ORANGE if j % 2 else MARIGOLD}"/>' for j, x in enumerate(range(96, 118, 4)))}</g>
<path d="M42 142 Q37 112 48 84" stroke="#8a5a32" stroke-width="5" fill="none" stroke-linecap="round"/>
<g stroke="#3d8b3d" stroke-width="4.5" fill="none" stroke-linecap="round">
<path d="M48 84 Q30 76 18 90"/><path d="M48 84 Q34 64 20 68"/><path d="M48 84 Q50 62 64 58"/>
<path d="M48 84 Q66 72 76 86"/><path d="M48 84 Q44 68 36 58"/></g>
<circle cx="46" cy="88" r="2.6" fill="#6b4a2a"/><circle cx="51" cy="88" r="2.6" fill="#6b4a2a"/>
<ellipse cx="208" cy="136" rx="17" ry="9" fill="#5f9e4a"/>
<circle cx="200" cy="132" r="2.5" fill="{ORANGE}"/><circle cx="210" cy="129" r="2.5" fill="{MARIGOLD}"/><circle cx="217" cy="134" r="2.5" fill="{ORANGE}"/>
</svg>'''

_SUN = ('<g stroke="#f6a623" stroke-width="4" stroke-linecap="round">'
        + "".join(f'<line x1="32" y1="4" x2="32" y2="13" transform="rotate({a} 32 32)"/>' for a in range(0, 360, 45))
        + '</g><circle cx="32" cy="32" r="13" fill="#fbb935"/>')
_CLOUD = '<path d="M22 54h26a10 10 0 0 0 0-20a14 14 0 0 0-27 3a8.5 8.5 0 0 0 1 17z" fill="#eef0f4" stroke="#d6dbe3" stroke-width="2"/>'
_RAIN = ('<g stroke="#5b8def" stroke-width="3" stroke-linecap="round"><line x1="26" y1="58" x2="24" y2="63"/>'
         '<line x1="36" y1="58" x2="34" y2="63"/><line x1="46" y1="58" x2="44" y2="63"/></g>')
_WEATHER = {"sun": _SUN, "partly": f'<g transform="translate(-6,-8) scale(.8)">{_SUN}</g>{_CLOUD}',
            "cloud": _CLOUD, "rain": _CLOUD + _RAIN}
SKY_ICON = {"clear": "sun", "mostly_clear": "partly", "partly": "partly", "overcast": "cloud",
            "drizzle": "rain", "rain": "rain", "storm": "rain"}


def svg_weather(kind="sun", size=56):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 64 64">{_WEATHER[kind]}</svg>'


def day_icon(cloud, rain_prob):
    """Icoon voor een dag uit bewolking (%) en regenkans (%)."""
    if rain_prob is not None and rain_prob >= 60:
        return "rain"
    if cloud is None or cloud < 20:
        return "sun"
    return "partly" if cloud < 60 else "cloud"


ICON = {
    "house": f'<svg width="28" height="28" viewBox="0 0 24 24"><path d="M3 11.5 12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z" fill="{GREEN}"/><rect x="10.5" y="15" width="3" height="6" fill="#fff"/></svg>',
    "battery": '<svg width="34" height="44" viewBox="0 0 34 44"><rect x="12" y="1" width="10" height="5" rx="1.5" fill="#ee7d1a"/><rect x="3" y="5" width="28" height="37" rx="5" fill="#fff" stroke="#ee7d1a" stroke-width="3"/><rect x="8" y="18" width="18" height="19" rx="2" fill="#f5b301"/></svg>',
    "leaf": f'<svg width="28" height="28" viewBox="0 0 24 24"><path d="M20 4C9 4 4 9 4 16c0 1.4.3 2.6.8 3.6C7 13 11 10 16 8c-4 3-7.6 6.6-9.6 11.6 1.2.3 2.4.4 3.6.4C17 20 20 14 20 4z" fill="{GREEN}"/></svg>',
    "bolt": '<svg width="34" height="34" viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z" fill="#fff"/></svg>',
    "pin": '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7z" fill="#ee7d1a"/><circle cx="12" cy="9" r="2.6" fill="#fff"/></svg>',
    "user": '<svg width="26" height="26" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4.2" fill="#6e6a64"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7z" fill="#6e6a64"/></svg>',
    "bars": '<svg width="26" height="26" viewBox="0 0 24 24"><rect x="3" y="13" width="4.5" height="8" rx="1" fill="#f6a623"/><rect x="9.75" y="8" width="4.5" height="13" rx="1" fill="#ee7d1a"/><rect x="16.5" y="3" width="4.5" height="18" rx="1" fill="#f6a623"/></svg>',
    "trees": '<svg width="96" height="84" viewBox="0 0 96 84"><rect x="45" y="44" width="5" height="38" fill="#7a5a3a"/><ellipse cx="47.5" cy="34" rx="20" ry="28" fill="#5fae5f"/><rect x="17" y="56" width="4" height="26" fill="#7a5a3a"/><ellipse cx="19" cy="50" rx="13" ry="18" fill="#7cc276"/><rect x="75" y="58" width="4" height="24" fill="#7a5a3a"/><ellipse cx="77" cy="52" rx="12" ry="16" fill="#7cc276"/></svg>',
    "grid": f'<svg width="30" height="30" viewBox="0 0 24 24"><path d="M12 2 7 22M12 2l5 20M8.5 9h7M7.5 13h9M6.5 17h11" stroke="{BLUE}" stroke-width="1.8" fill="none" stroke-linecap="round"/></svg>',
    "plug": '<svg width="40" height="40" viewBox="0 0 24 24"><path d="M8 2v5M16 2v5M5 7h14v4a7 7 0 0 1-14 0zM12 18v4" stroke="#ee7d1a" stroke-width="2.2" fill="none" stroke-linecap="round"/></svg>',
    "bulb": '<svg width="34" height="34" viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2V16h5v-.1c0-.8.4-1.5 1-2A6 6 0 0 0 12 3z" fill="#fbb935" stroke="#f6a623"/></svg>',
    "target": '<svg width="28" height="28" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="#ee7d1a" stroke-width="2.2"/><circle cx="12" cy="12" r="4.5" fill="none" stroke="#ee7d1a" stroke-width="2.2"/><circle cx="12" cy="12" r="1.5" fill="#ee7d1a"/></svg>',
    "calendar": '<svg width="26" height="26" viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="16" rx="2" fill="none" stroke="#ee7d1a" stroke-width="2"/><path d="M3 10h18M8 3v4M16 3v4" stroke="#ee7d1a" stroke-width="2"/></svg>',
    "clock": '<svg width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="#ee7d1a" stroke-width="2"/><path d="M12 7v5l3 2" stroke="#ee7d1a" stroke-width="2" fill="none"/></svg>',
    "gear": '<svg width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2" fill="none" stroke="#6e6a64" stroke-width="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1 7 17M17 7l2.1-2.1" stroke="#6e6a64" stroke-width="2"/></svg>',
}


def mini_bars(color, size=40):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><rect x="3" y="13" width="4.5" height="8" rx="1" '
            f'fill="{color}" opacity=".45"/><rect x="9.75" y="9" width="4.5" height="12" rx="1" fill="{color}" '
            f'opacity=".7"/><rect x="16.5" y="4" width="4.5" height="17" rx="1" fill="{color}"/></svg>')


def illus_panel(w=240, h=130, brush=True):
    """Zonnepaneel met glinsteringen (en borstel) voor de tip 'panelen schoonmaken'."""
    cells = "".join(f'<line x1="{40 + i * 36}" y1="118" x2="{100 + i * 36}" y2="20" stroke="#bcd3f2" stroke-width="2"/>' for i in range(1, 4))
    rows = "".join(f'<line x1="{40 + k * 15}" y1="{118 - k * 24}" x2="{184 + k * 15}" y2="{118 - k * 24}" stroke="#bcd3f2" stroke-width="2"/>' for k in range(1, 4))
    star = '<path d="M0-9L2-2 9 0 2 2 0 9-2 2-9 0-2-2Z" fill="#ffd34d"/>'
    br = ('<g transform="rotate(-25 175 60)"><rect x="168" y="20" width="8" height="70" rx="3" fill="#c8915b"/>'
          '<rect x="145" y="78" width="55" height="16" rx="4" fill="#e9b65f"/>'
          '<g stroke="#8c6a3a" stroke-width="2">' + "".join(f'<line x1="{150 + i * 6}" y1="94" x2="{150 + i * 6}" y2="102"/>' for i in range(8)) + '</g></g>') if brush else ""
    return (f'<svg viewBox="0 0 260 130" width="100%" height="{h}" preserveAspectRatio="xMidYMid slice">'
            f'<rect width="260" height="130" fill="#dcecfb"/><circle cx="230" cy="20" r="14" fill="#fbd36b"/>'
            f'<ellipse cx="30" cy="125" rx="60" ry="22" fill="#9fd08a"/><ellipse cx="240" cy="128" rx="50" ry="18" fill="#9fd08a"/>'
            f'<polygon points="40,118 184,118 244,20 100,20" fill="#2f5aa8"/>{cells}{rows}'
            f'<g transform="translate(90 60)">{star}</g><g transform="translate(200 45) scale(.7)">{star}</g>'
            f'<g transform="translate(150 100) scale(.6)">{star}</g>{br}</svg>')


def illus_washer(h=130):
    return (f'<svg viewBox="0 0 260 130" width="100%" height="{h}" preserveAspectRatio="xMidYMid slice">'
            f'<rect width="260" height="130" fill="#fde9cf"/><rect x="170" y="10" width="70" height="60" fill="#fff5e6" stroke="#f2c98f" stroke-width="3"/>'
            f'<circle cx="205" cy="40" r="14" fill="#fbb935"/>'
            f'<rect x="70" y="18" width="90" height="108" rx="8" fill="#fbfbfb" stroke="#d9d4cd" stroke-width="2"/>'
            f'<rect x="70" y="18" width="90" height="20" rx="8" fill="#ece8e2"/><circle cx="115" cy="80" r="30" fill="#c9d6e3" stroke="#9aa9b8" stroke-width="5"/>'
            f'<circle cx="115" cy="80" r="18" fill="#7f97b0"/><rect x="20" y="96" width="34" height="30" rx="4" fill="#e9a35b"/>'
            f'<ellipse cx="37" cy="92" rx="16" ry="10" fill="#7cc276"/></svg>')


def illus_ac(h=130, temp="24°C"):
    vents = "".join(f'<line x1="70" y1="{78 + i * 7}" x2="200" y2="{78 + i * 7}" stroke="#cfd8e0" stroke-width="2"/>' for i in range(3))
    return (f'<svg viewBox="0 0 260 130" width="100%" height="{h}" preserveAspectRatio="xMidYMid slice">'
            f'<rect width="260" height="130" fill="#e3f0fb"/>'
            f'<rect x="55" y="20" width="160" height="78" rx="12" fill="#ffffff" stroke="#d4dde6" stroke-width="2"/>'
            f'<rect x="80" y="32" width="70" height="30" rx="5" fill="#2b3440"/><text x="115" y="53" font-size="17" '
            f'font-family="Arial" fill="#ffffff" text-anchor="middle">{temp}</text>{vents}'
            f'<rect x="22" y="94" width="20" height="30" rx="3" fill="#e9a35b"/><ellipse cx="32" cy="88" rx="16" ry="14" fill="#7cc276"/></svg>')


# ---------- kop van elke pagina ----------
def header(title, subtitle, icon_html, place, today, cur):
    """Links: icoon + titel + ondertitel. Rechts: locatie + datum | weer nu | avatar."""
    html(svg_toran())
    left, right = st.columns([1.55, 1])
    html(f'<div class="row" style="gap:16px">{icon_html}<div><div class="title">{title}</div>'
         f'<div class="subtitle">{subtitle}</div></div></div>', left)
    if cur:
        sky = cur["sky"] if cur["is_day"] or cur["sky"] != "clear" else "clear_night"
        weather = (f'<div class="row" style="gap:8px">{svg_weather(SKY_ICON.get(cur["sky"], "partly"), 46)}'
                   f'<div><b style="font-size:21px">{cur["temp"]:.0f}°C</b><br>'
                   f'<span class="label">{t("sky_" + sky)}</span></div></div><div class="sep"></div>')
    else:
        weather = ""
    html(f'<div class="hdr"><div class="row" style="gap:8px">{ICON["pin"]}<div><b>{place}, India</b><br>'
         f'<span class="label">{date_long(pd.Timestamp(today), lang())}</span></div></div><div class="sep"></div>'
         f'{weather}<div class="avatar">{ICON["user"]}</div></div>', right)
    st.write("")


def updated(ts):
    return f'<span class="live">● {t("live", t=time_str(ts, lang()))}</span>'

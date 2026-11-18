# -*- coding: utf-8 -*-
"""
De app testen met de persona's (Streamlit AppTest)  -  eindmodel v2 (BESLISSINGEN #60)
======================================================================================
Draait app.py zonder browser, voor elke persona van de wiki (Design Thinking) en elke pagina.
Controleert: geen foutmelding of crash, en wat de persona nodig heeft.
Extra scenario's: "nieuw huis zonder data" (f = 1, uitleg kalibratie) en "huis met 6 weken data"
(nagebootste dagopbrengst, GEEN echte metingen -> kalibratie aan). Forecast-pagina in alle 6 talen,
en 'How accurate?' toont de v2-scores uit model_info_v2.json.
Gebruikt een tijdelijke gebruikersmap (SOLARHOME_USER_DIR): het echte profiel wordt niet aangeraakt.

Runnen: F5 in Spyder of VS Code (Anaconda). Internet nodig (live weer). ~3-6 minuten.
"""

# %% 0. Instellingen
from pathlib import Path
import json
import os
import sys
import tempfile
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd() / "app"
TMP = Path(tempfile.mkdtemp(prefix="solarhome_test_"))
os.environ["SOLARHOME_USER_DIR"] = str(TMP)          # vóór de eerste import van appcore.config
sys.path[:0] = [str(ROOT.parent), str(ROOT)]
from streamlit.testing.v1 import AppTest              # noqa: E402
from appcore.config import MODEL_INFO, PANELS_FILE    # noqa: E402
from appcore.i18n import t as tr, LANGS               # noqa: E402
from ui.state import DEFAULTS, PAGES                  # noqa: E402

assert PANELS_FILE.parent == TMP, "test schrijft niet naar de tijdelijke map"
PERSONAS = {   # persona's van de wiki -> instellingen in de app
    "Anita (Pune, middel)": dict(city="Pune", simple_view=True, energy_style="both"),
    "Rahul (Bengaluru, hoog)": dict(city="Bengaluru", simple_view=False, energy_style="kwh"),
    "Lakshmi (Hyderabad, laag)": dict(city="Hyderabad", lang="hi", energy_style="examples", big_text=True),
    "Vikram (Jaipur, batterij, stof)": dict(city="Jaipur", inv_type="hybrid", dust="dust_high",
                                            clean="cl_rain"),
}
info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
pick = lambda prefix: next(v for k, v in info["dagvooruit"].items() if k.startswith(prefix))   # noqa: E731
ERR = f"{pick('v2: + mengen')['dagfout_%']:.0f}"
BASE = f"{pick('v2: alleen gemiddelde dag')['dagfout_%']:.0f}"
NONE_TXT = tr("calib_none", "en")
ON_TXT = tr("calib_on", "en").split("{n}")[0]        # vaste tekst vóór het aantal dagen


def write_log(n_days, kwh=11.0):
    """n_days nagebootste dagopbrengsten (kWh) tot en met gisteren; 0 dagen -> geen bestand."""
    PANELS_FILE.unlink(missing_ok=True)
    if n_days:
        days = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
                             - pd.Timedelta(days=1), periods=n_days).date
        log = {str(d): {"solar": kwh} for d in days}
        PANELS_FILE.write_text(json.dumps({"log": log, "cleaned": [], "demo": False}), encoding="utf-8")


def run_page(settings, page):
    """App draaien met deze instellingen op deze pagina -> (alle tekst, fouten)."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    for k, v in {**DEFAULTS, **settings, "page": page}.items():
        at.session_state[k] = v
    at.session_state["loaded"] = True                 # niet het opgeslagen profiel laden
    at.run()
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [e.label for e in at.expander] + [w.value for w in at.warning])
    errors = [e.value for e in at.error] + [str(x.value) for x in at.exception]
    return text, errors


# %% 1. Elke persona, elke pagina (nieuw huis: geen ingevulde opwek)
rows = []
write_log(0)
for persona, s in PERSONAS.items():
    lang = s.get("lang", "en")
    for page in PAGES:
        text, errors = run_page(s, page)
        rows.append({"scenario": persona, "pagina": page, "check": "geen fout", "ok": not errors,
                     "detail": "; ".join(errors)[:200]})
        checks = []
        if page == "home" and persona.startswith("Anita"):
            checks = [("goede/gemiddelde/weinig zon?", any(tr(k, lang) in text for k in
                                                          ("verdict_good", "verdict_avg", "verdict_poor"))),
                      ("beste moment", tr("best_now", lang) in text),
                      ("morgen", tr("v_tomorrow", lang) in text),
                      ("rest onder 'Meer details'", tr("more_details", lang) in text)]
        if page == "forecast" and persona.startswith("Rahul"):
            checks = [("'How accurate' aanwezig", tr("how", lang) in text),
                      (f"v2-dagfout {ERR}% getoond", f"**{ERR}%**" in text),
                      (f"gemiddelde dag {BASE}% getoond", f"**{BASE}%**" in text)]
        if page == "home" and persona.startswith("Lakshmi"):
            checks = [("Hindi", tr("best_now", "hi") in text)]
        if page == "home" and persona.startswith("Vikram"):
            checks = [("advies batterij laden", tr("do_batt", lang) in text)]
        rows += [{"scenario": persona, "pagina": page, "check": c, "ok": ok, "detail": ""} for c, ok in checks]
    print("klaar:", persona)

# %% 2. Nieuw huis zonder data en huis met 6 weken data (alle pagina's)
for name, n in [("nieuw huis zonder data", 0), ("huis met 6 weken data", 42)]:
    write_log(n)
    for page in PAGES:
        text, errors = run_page({"city": "Bengaluru"}, page)
        rows.append({"scenario": name, "pagina": page, "check": "geen fout", "ok": not errors,
                     "detail": "; ".join(errors)[:200]})
        if page == "forecast":
            ok = (NONE_TXT in text and ON_TXT not in text) if n == 0 else (ON_TXT in text and "28" in text)
            rows.append({"scenario": name, "pagina": page, "ok": ok, "detail": "",
                         "check": "uitleg: nog niet afgestemd (f = 1)" if n == 0 else "afgestemd met 28 dagen (4 weken)"})
    print("klaar:", name)
write_log(0)

# %% 3. Forecast-pagina in alle talen
for lang in LANGS:
    text, errors = run_page({"lang": lang}, "forecast")
    rows.append({"scenario": f"taal {lang}", "pagina": "forecast", "check": "geen fout + v2-scores",
                 "ok": not errors and f"**{ERR}%**" in text, "detail": "; ".join(errors)[:200]})

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(f"\n{res.ok.sum()} van {len(res)} checks OK")
assert res.ok.all(), "niet alle checks OK"

# UI handoff – SolarHome Forecast app

Status on 24 Nov 2026, for the teammate who continues the UI (step 6 of SCRUM-104).

The app already runs on **final model v2** (PR #4, branch `app-v2`). This document lists what works, how the code
is organised, and what should be improved in the UI. The model and the forecast numbers are done: **step 6 is only
about how things look and read**, not about the calculations.

## 1. Run and test

```
pip install -r requirements.txt
python -m streamlit run app/app.py     # opens http://localhost:8501 (or double-click start_app.bat)
python app/test_app.py                 # 47 checks, all pages, 4 personas, 6 languages – must stay 47/47
```

- Needs internet for the live weather forecast (Open-Meteo).
- User data is stored in `app/gebruiker/` (ignored by git). Never commit it. The tests use a temporary folder.
- Anaconda Python 3.13 is the tested environment.

## 2. Where things are

| File | What it does |
|---|---|
| `app/app.py` | Entry point: page navigation, sidebar |
| `app/ui/home.py` | Home page (today, best time to use power, today's curve) |
| `app/ui/forecast.py` | Forecast page (16 days, pick a day, "what affects your panels") |
| `app/ui/usage.py` | My Usage (consumption vs solar, appliances, trend, insights, readings form) |
| `app/ui/tips.py` | Tips (featured tip, recommended, quick tips, seasonal tips) |
| `app/ui/settings.py` | Settings (solar setup, my home, customise app) |
| `app/ui/charts.py` | All plotly charts, shared by the pages |
| `app/ui/style.py` | Colours, CSS, page header, SVG icons |
| `app/ui/common.py` | Parts used on several pages (status card, readings form) |
| `app/ui/state.py` | User profile and settings (`st.session_state`, saved to `app/gebruiker/profiel.json`) |
| `app/ui/data.py` | Cached live data (weather + forecast) |
| `app/appcore/i18n.py` | **All visible text**, in 6 languages (en, nl, ko, sv, ...). Use `t(key, lang)` |
| `app/appcore/forecast.py`, `usage.py`, `alerts.py`, `equivalents.py` | Calculations – do not change in step 6 |

Code comments are in Dutch (project rule); visible text is English and goes through `i18n.py`.

## 3. Screenshots (19 Nov 2026, 17:49 IST, demo house New Delhi, 3 kW)

Desktop (1400 px, full page) and mobile (390 px, top of the page only) in [`docs/ui-review/`](ui-review/):
`home`, `forecast`, `usage`, `tips`, `settings` × `_desktop.png` / `_mobile.png`. No page showed an error.

## 4. What to improve (most important first)

### Wrong or confusing content
1. **Home after sunset** – at 17:49 the main card still says "Today is a good solar day, 13.2 kWh", while the day is
   over. After sunset, tomorrow should be the main card and today a small line. (`ui/home.py`)
2. **Forecast "Best day"** – picks today (already almost over). Only pick from tomorrow on; use a short date
   ("Fri 20 Nov") at the same size as the other cards. (`ui/forecast.py`)
3. **Forecast 16-day chart** – the grey dashes above the bars have no legend; "Striped = forecast" but today's bar
   (also a forecast) is solid; rotated date labels are hard to read. (`ui/charts.py`)
4. **"+0.0%" and "−0.0%"** – Tips badge "Clean your solar panels: +0.0% generation" and Forecast card "Dust on
   panels −0.0%" look like bugs. Below ~0.5 %: hide the badge or say "little effect now". (`ui/tips.py`, `ui/forecast.py`)
5. **My Usage "Usage Trend"** – value labels on top of the bars are cut off (the "53" is not visible); the
   "avg 47 kWh" label lies on top of a bar. (`ui/charts.py`)
6. **My Usage "Energy Usage vs. Solar Generation"** – a flat consumption line (~7 kWh), grid bars (~4.3 kWh) and a
   solar line (~13 kWh) are hard to relate for a non-expert. Suggestion: stacked bars "solar used at home" +
   "from grid" (= consumption), solar production as a line. "This week" mixes past days with future (estimated)
   days, and the KPI cards show "↓9 % vs previous week" in red without saying it is an estimate. Mark future days
   lighter and label the cards "estimated" while there are no meter readings. (`ui/usage.py`, `ui/charts.py`)

### Too technical for the personas (Anita, Lakshmi)
7. **Forecast "What affects your panels here"** – 8 cards incl. air density (kg/m³), water vapour (cm) and air
   pressure (hPa). Show 4 by default (clouds, rain, panel temperature, dust), the rest in "More details".
8. The same equivalent ("13 washing machine loads + 8.5 h of water pump") appears 3× on Home and Forecast. Once per
   page is enough.

### Layout
9. **Header** – "Thursday 19 November" wraps over two lines on every page; Settings has no weather block while the
   other pages do. (`ui/style.py`)
10. **Appliance donut legend** – "Cooling (AC, fans)" wraps over 3 lines; numbers (7.0, 19.6) have no unit (kWh).
11. **Settings** – "Your name" sits under "Location"; the map first shows an empty grey tile; the profile note shows
    the Dutch path `gebruiker/profiel.json` – say "saved on this computer only".
12. **Contrast** – small grey text and orange text on the cream background look weaker than 4.5:1 (WCAG AA).
    Estimated by eye, **not measured**. Measure and adjust the colour tokens in `ui/style.py`.
13. **Mobile (< 600 px)** – the 4 KPI cards on Forecast stack and fill the whole first screen (2×2 grid is better);
    the big sun icon on Home takes a lot of space.
14. **Reliability band (FR9)** – the likely range (P10–P90) is shown on Forecast but not on Home's chart.

Not checked: whether the "›" arrows in Quick Tips do something; the lower part of the mobile pages.

## 5. Rules for step 6

- Do not change the model, `appcore/forecast.py` or other calculations. If a number looks wrong, ask Dimme first.
- One change at a time; after each change run `python app/test_app.py` (47/47) and take a before/after screenshot.
- New visible text: add it to `app/appcore/i18n.py` in all languages.
- Own branch per task, pull request to `main`, someone else reviews and merges.
- Keep the existing look (colours and fonts in `ui/style.py`) unless the team agrees otherwise.

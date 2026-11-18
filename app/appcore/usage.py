# -*- coding: utf-8 -*-
"""
Huishoudverbruik schatten en vergelijken met de opwek (voor 'My Usage' en 'Tips')
================================================================================
De app heeft geen slimme meter. Daarom SCHATTEN we het verbruik, tenzij de gebruiker
zelf meterstanden invult (die gaan dan voor).

  1. Gemiddeld verbruik per dag = 'units' per maand van de stroomrekening / 30.
  2. Verdeling over groepen apparaten volgens het CLASP/BEE-onderzoek (2024, 4.321
     huishoudens in India): koeling 40 %, keuken 28 %, verlichting 11 %, overig 21 %
     (overig verdeeld als: elektronica 8 %, water (pomp/geiser) 7 %, rest 6 %).
  3. KOELING (ventilator, airco, cooler) hangt af van de temperatuur: meer bij warm weer
     (factor 1 bij een daggemiddelde van 28 °C) en van de airco-stand: ~6 % minder per °C
     warmer dan 24 °C (BEE). Dit is een eenvoudige AANNAME, geen gemeten verband.
  4. Verdeling over de dag met een vast profiel per groep (ochtend- en avondpiek).
  5. Per kwartier: uit zon = min(opwek, verbruik); de rest komt van het net.
"""
import numpy as np
import pandas as pd

CATS = ["cooling", "kitchen", "lighting", "water", "electronics", "other"]
SHARES = {"cooling": 0.40, "kitchen": 0.28, "lighting": 0.11, "water": 0.07,
          "electronics": 0.08, "other": 0.06}
T_BASE, T_REF = 22.0, 28.0          # koeling: basistemperatuur en 'normale' warme dag
FLEXIBLE = {"water": 1.0, "other": 0.5}     # deel dat je naar zonnige uren kunt verplaatsen


def _shape(weights):
    w = np.array(weights, dtype=float)
    return w / w.sum()


# verdeling over 24 uur (0..23 h); som = 1
SHAPE = {
    "cooling": _shape([4, 4, 3, 3, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7, 7, 6, 5, 5, 5, 6, 6, 5, 5]),
    "kitchen": _shape([2, 2, 2, 2, 2, 3, 6, 8, 7, 4, 3, 4, 6, 5, 3, 3, 3, 4, 6, 8, 7, 4, 3, 2]),
    "lighting": _shape([1, 1, 1, 1, 1, 2, 3, 2, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 3, 7, 9, 9, 8, 5, 2]),
    "water": _shape([0, 0, 0, 0, 0, 3, 8, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1, 0, 0]),
    "electronics": _shape([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 4, 6, 8, 8, 7, 4, 2]),
    "other": _shape([1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]),
}


def daily_use(t_mean, monthly_units=300.0, ac_setpoint=24, has_ac=True):
    """Geschat verbruik (kWh) per dag en per groep. t_mean: Series datum -> gem. temperatuur."""
    avg = monthly_units / 30
    t = pd.Series(t_mean, dtype=float)
    cool = (np.clip(t - T_BASE, 0, None) + 2) / (T_REF - T_BASE + 2)
    if has_ac:
        cool = cool * 0.94 ** (ac_setpoint - 24)              # BEE: ~6 % per °C
    out = pd.DataFrame({c: avg * SHARES[c] for c in CATS}, index=t.index)
    out["cooling"] = avg * SHARES["cooling"] * cool
    return out


def load_kw(daily, times):
    """Verbruik per kwartier (kW) voor de tijden in 'times', uit het dagverbruik per groep."""
    times = pd.DatetimeIndex(times)
    day = daily.reindex(times.date)
    hour = times.hour.values
    kw = np.zeros(len(times))
    for c in CATS:
        kw += day[c].values * SHAPE[c][hour]               # kWh in dat uur = gemiddelde kW
    return pd.Series(kw, index=times)


def split(solar_kw, load):
    """Per kwartier: uit zon, naar het net (overschot) en van het net (kW)."""
    s = pd.Series(np.asarray(solar_kw, dtype=float), index=load.index)
    from_solar = np.minimum(s, load)
    return pd.DataFrame({"solar": s, "use": load, "from_solar": from_solar,
                         "export": s - from_solar, "from_grid": load - from_solar})


def shift_gain(solar_kw, daily, times, window_hours=4):
    """Hoeveel meer zonnestroom je zelf gebruikt (kWh) als je de verplaatsbare apparaten
    (waterpomp/geiser, de helft van 'overig': wasmachine, strijkijzer) laat draaien in de
    zonnigste 4 uur van elke dag."""
    times = pd.DatetimeIndex(times)
    base = split(solar_kw, load_kw(daily, times))
    moved = daily.copy()
    flex = sum(daily[c] * f for c, f in FLEXIBLE.items())
    for c, f in FLEXIBLE.items():
        moved[c] = daily[c] * (1 - f)
    load2 = load_kw(moved, times)
    s = pd.Series(np.asarray(solar_kw, dtype=float), index=times)
    for day, g in s.groupby(times.date):
        best = g.rolling(window_hours * 4).sum().idxmax()      # einde van het zonnigste blok
        block = g.loc[best - pd.Timedelta(minutes=15 * (window_hours * 4 - 1)): best].index
        load2.loc[block] += flex.get(day, 0) / window_hours    # kWh gelijk verdeeld over het blok
    after = split(s, load2)
    return float((after.from_solar.sum() - base.from_solar.sum()) * 0.25)

# -*- coding: utf-8 -*-
"""
Energie in herkenbare voorbeelden (voor gebruikers voor wie 'kWh' niets zegt)
=============================================================================
Bv. "≈ 12 kWh · genoeg voor 12 wasbeurten + 8 uur waterpomp".

Per apparaat: energie per keer / per uur / per km, met de BRON of het label 'aanname'.
Waar een bron een bereik geeft, kiezen we een gemiddelde waarde. Het blijft een indicatie:
echte apparaten verschillen per merk, stand en gebruik.
"""
import math

# key: (kWh per eenheid, stap waarop we afronden, bron / aanname)
ITEMS = {
    "washing": (0.5, 1, "AANNAME: front-load 7 kg ~0,42-0,48 kWh/was (verkoper, 5 sterren); "
                         "BEE-norm 60 °C: 0,09 kWh/kg/was -> 0,63 kWh. Koud wassen (top-load) gebruikt minder."),
    "pump": (0.75, 0.5, "AANNAME: waterpomp 1 pk = 0,746 kW, 1 uur draaien (omrekening pk -> kW)."),
    "fan": (0.075, 1, "BRON: gewone plafondventilator 75 W (Crompton; 5-sterren BLDC-ventilator: 28-50 W)."),
    "geyser": (0.5, 1, "AANNAME: geiser 2 kW, 15 minuten opwarmen voor één warme douche."),
    "scooter": (0.033, 5, "BRON: elektrische scooter in de stad ~33 Wh/km (LBNL-onderzoek voor India; "
                          "Ather rekent met 30 Wh/km)."),
    "fridge": (1.7, 1, "BRON: koelkast 250 L frost-free, 3 sterren: 626 kWh/jaar (BEE-tabel) = 1,7 kWh/dag."),
    "ac": (1.5, 0.5, "AANNAME: airco 1,5 ton, ~1,5 kW opgenomen vermogen, 1 uur."),
    "phone": (0.015, 10, "AANNAME: telefoon opladen ~15 Wh (accu ~4000 mAh x 3,85 V, plus verlies)."),
    "backup": (0.19, 0.5, "AANNAME: bij stroomuitval 2 plafondventilatoren (2 x 75 W) + 4 LED-lampen (4 x 9 W) "
                          "= ~0,19 kW, 1 uur."),
}

# welke voorbeelden bij welk huis: batterij -> uren back-up bij stroomuitval
DEFAULT_PAIR = ("washing", "pump")
BATTERY_PAIR = ("backup", "washing")


def _floor(x, step):
    return math.floor(x / step) * step


def split(kwh, keys=DEFAULT_PAIR):
    """Verdeel de energie over twee voorbeelden: de helft voor het eerste (afgerond naar beneden),
    de rest voor het tweede. Geeft [(key, aantal), ...] zonder nullen."""
    out, rest = [], max(kwh, 0.0)
    for i, k in enumerate(keys):
        per, step, _ = ITEMS[k]
        share = rest / 2 if i < len(keys) - 1 else rest
        n = _floor(share / per, step)
        if n > 0:
            out.append((k, n))
            rest -= n * per
    if not out:                                        # te weinig voor beide: één klein voorbeeld
        per, step, _ = ITEMS["fan"]
        n = _floor(kwh / per, step)
        if n > 0:
            out.append(("fan", n))
    return out


def fmt_count(n):
    return f"{n:g}"

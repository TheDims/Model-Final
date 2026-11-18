# -*- coding: utf-8 -*-
"""App-onderdelen van SolarHome Forecast (overgenomen uit Model V5 solarhome/, alleen wat de app gebruikt).

Natuurkunde, weer en het model komen uit solarcore/ (één bron van waarheid, zelfde code als bij het trainen):
  forecast.py     dun laagje op solarcore: weer -> verwacht vermogen (kW) + band, per kwartier
  alerts.py       melding schoonmaken / storing (gemeten opwek t.o.v. verwacht)
  usage.py        huishoudverbruik schatten (CLASP/BEE 2024)
  equivalents.py  energie in herkenbare voorbeelden
  i18n.py         teksten in 6 talen
  config.py       paden van de app (gebruikersmap, meldingsregels)
"""

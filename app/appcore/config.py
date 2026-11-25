# -*- coding: utf-8 -*-
"""Paden en instellingen van de app (overgenomen uit V5 solarhome/config.py; alleen wat de app gebruikt).
Model, natuurkunde en weer staan in solarcore/ (solarcore/config.py: APP_MODEL_FILE)."""
import os
from pathlib import Path

from solarcore.config import MODELS, APP_MODEL_FILE, APP_MODEL_INFO, TZ   # noqa: F401  (één bron van waarheid)

APP = Path(__file__).resolve().parent.parent
# profiel en ingevulde opwek van de gebruiker (lokaal, NIET in git: GDPR); de test gebruikt een tijdelijke map
USER = Path(os.environ.get("SOLARHOME_USER_DIR", APP / "gebruiker"))
MODEL_FILE = APP_MODEL_FILE
MODEL_INFO = APP_MODEL_INFO
RULES_FILE = MODELS / "meldingsregels.json"
PROFILE_FILE = USER / "profiel.json"
PANELS_FILE = USER / "mijn_panelen.json"

INDIA = dict(lat=(6.0, 37.5), lon=(68.0, 97.5))          # ruwe grenzen: de app werkt alleen in India (IST)

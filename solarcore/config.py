# -*- coding: utf-8 -*-
"""Instellingen op één plek: paden, plants, split en constanten. Geen magische getallen in de scripts."""
import os
from pathlib import Path

# ---------- run (AGENTS.md §6.0, BESLISSINGEN #16) ----------
# Welke run draait? Standaard de hoofdrun 'split70'. Voor een andere run (bv. de regressietest) zet je vóór het
# starten de omgevingsvariabele SOLAR_RUN, bv. in PowerShell: $env:SOLAR_RUN = "split30_check".
RUNS = {"split30": 0.30,                 # run 1: eerste 30 % kiezen, laatste 70 % testen (archief)
        "split30_check": 0.30,           # regressietest: moet run 1 exact teruggeven
        "split70": 0.70}                 # run 2 (hoofdrun): eerste 70 % trainen, laatste 30 % testen
RUN_NAME = os.environ.get("SOLAR_RUN", "split70")
if RUN_NAME == "split30":                # archief van run 1: nooit opnieuw draaien (AGENTS.md §4)
    raise RuntimeError("resultaten/split30 is het archief van run 1. Gebruik SOLAR_RUN=split30_check.")
TRAIN_FRACTION = RUNS[RUN_NAME]
# Per run: van welke fases de testdagen al bekeken mogen worden (na akkoord van Dimme). Zo kan een run nooit per
# ongeluk de testdagen tonen vóór de keuze vaststaat (AGENTS.md §6.2).
TEST_VRIJGEGEVEN = {"split30": {1, 2, 3}, "split30_check": {1, 2, 3},
                    "split70": {1, 3}}   # fase 1 en 3: akkoord Dimme 2026-10-29


def toon_test(fase):
    """True als de testdagen van deze fase in deze run bekeken mogen worden."""
    return fase in TEST_VRIJGEGEVEN.get(RUN_NAME, set())


# ---------- mappen ----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                     # Kaggle-csv's + ERA5-cache (kopie uit Model V5/data)
CLEAN = DATA / "clean" / RUN_NAME        # opgeschoonde data per run (de dip-vlag hangt van de trainingsdagen af)
RESULTS = ROOT / "resultaten" / RUN_NAME
MODELS = ROOT / "modellen"
CLEAN_FILE = CLEAN / "plant_15min.csv"
MODEL_FILE = MODELS / f"model_final_{RUN_NAME}.joblib"

# ---------- plants ----------
PLANTS = (1, 2)
# GESCHATTE locaties (beste correlatie plant-sensor <-> ERA5, zie V2/wiki), geen echte adressen
PLANT_LOCATIONS = {1: (14.5, 77.5), 2: (19.0, 79.0)}
TZ = "Asia/Kolkata"                      # alle tijden zijn lokale tijd India (IST), zonder label
# datumformaat per bestand (Plant 1 generation gebruikt dag-maand-jaar)
DATE_FORMATS = {("gen", 1): "%d-%m-%Y %H:%M", ("gen", 2): "%Y-%m-%d %H:%M:%S",
                ("sensor", 1): "%Y-%m-%d %H:%M:%S", ("sensor", 2): "%Y-%m-%d %H:%M:%S"}
PERIOD = ("2020-05-15 00:00", "2020-06-17 23:45")
FREQ = "15min"

# ---------- opschonen (zelfde regels als V5 / wiki) ----------
DC_FIX_PLANT1 = 10.0                     # Plant 1 meldt DC_POWER ~10x te hoog
MIN_INVERTERS = 8                        # < 8 actieve omvormers in een kwartier -> AC_KW = NaN
OUTAGE_MEDIAN_FRAC = 0.05                # uitval: AC <= 0 terwijl de mediaan > 5 % van het plant-maximum is
MAX_GAP_QUARTERS = 4                     # sensorgaten tot 1 uur interpoleren, langer -> NaN
# plant-brede dip (BESLISSINGEN #10): sensor ziet zon, maar de hele plant levert veel te weinig
DIP_MIN_IRR = 0.2                        # kW/m²
DIP_MAX_RATIO = 0.3                      # opwek < 30 % van wat de straling verwacht (verhouding uit de trainingsdagen)
SENSOR_COLS = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]

# ---------- split (op tijd, hele dagen) ----------
# n_train = round(TRAIN_FRACTION x aantal dagen); validatievouwen afgeleid in split.val_folds (BESLISSINGEN #17)
VAL_SHARE = 0.40                         # validatie = laatste 40 % van de trainingsdagen
VAL_FOLDS_SMALL, VAL_FOLDS_LARGE, LARGE_FROM = 2, 3, 15   # 2 vouwen bij < 15 trainingsdagen, anders 3
BAND_CAL_START = 11                      # band: walk-forward-kalibratie vanaf dag 11 (binnen de training, §6.3)
LEARNING_SIZES = (10, 17, 24, 27)        # leercurve: aantal trainingsdagen ...
LEARNING_TEST = (28, 34)                 # ... altijd getest op dag 28-34 (= de 80/20-split)

# ---------- target en evaluatie ----------
CAPACITY_QUANTILE = 0.99                 # capaciteit = 99e percentiel van AC_KW overdag in de training
MIN_COSZ = 0.02                          # 'overdag' = cos(zenit) > 0,02 (evaluatie alleen overdag)
HOUSE_KWP = 5.0                          # voorbeeldhuis voor fouten in kW en kWh
GAMMA_PDC = -0.004                       # temperatuurcoëfficiënt (1/°C) - AANNAME tot getest

# ---------- keuzeregel (BESLISSINGEN #2) ----------
TIE_PP = 0.5                             # verschil < 0,5 pp nRMSE -> eenvoudigste model
COMPLEX_PP = 1.0                         # complex model alleen als >= 1 pp beter en op alle vouwen beter
N_BOOTSTRAP = 1000                       # bootstrap over testdagen (90 %-interval)
RANDOM_STATE = 42

# ---------- grafieken ----------
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29d"
INK, INK2 = "#0b0b0b", "#52514e"
PLANT_COLORS = {1: BLUE, 2: ORANGE}
DPI = 150

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

# ---------- TAAK 5a: externe validatie HKUST (Hong Kong, 2021-2023) ----------
# Bron: Lin et al. (2025), Scientific Data; Dryad doi:10.5061/dryad.m37pvmd99, versie 8 sep 2024, CC0.
HKUST_RAW = DATA / "extern" / "hkust" / "raw" / "Dataset"
HKUST_TTL = HKUST_RAW / "Metadata" / "PV generation system metadata.ttl"
HKUST_TS = HKUST_RAW / "Time series dataset"
HKUST_INV = HKUST_TS / "PV generation dataset" / "PV stations with panel level optimizer" / "Inverter level dataset"
HKUST_MET = HKUST_TS / "Meteorological dataset"
HKUST_RESULTS = ROOT / "resultaten" / "taak5" / "hkust"
HKUST_TZ = "Asia/Hong_Kong"
HKUST_LOCATION = (22.3363, 114.2634)     # weerstation campus (README); de .ttl geeft afgeronde, afwijkende coördinaten
HKUST_YEARS = (2021, 2022, 2023)         # retrain-opzet: train 2021, valideer 2022, test 2023
HKUST_DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M")   # beide komen voor (per bestand consequent)
# csv-naam -> naam in de Brick-.ttl als normaliseren (kleine letters, alleen letters/cijfers) niet genoeg is
HKUST_NAME_ALIAS = {"indoorsportscentre": "indoorsportcenter"}
# stationkeuze: drempels (vooraf vast, vóór het modelleren)
HKUST_MIN_GHI = 200.0                    # W/m²: "er is zon" voor uitval- en schaduwcontrole
HKUST_MIN_ELEV = 15.0                    # graden zonnehoogte voor de schaduwcontrole
HKUST_OUTAGE_FRAC = 0.01                 # uitval: P < 1 % van kWp terwijl GHI > HKUST_MIN_GHI
HKUST_MIN_COVERAGE = 0.95                # min. deel van de daguren met data, per jaar
HKUST_MIN_SHADE_INDEX = 0.80             # PR in de slechtste uur-klasse / PR rond de middag
HKUST_MIN_R = 0.90                       # correlatie opwek <-> GHI weerstation (5 min, overdag)
# gekozen stations (BESLISSINGEN #25) en bekende storingsmaanden (PR t.o.v. ERA5 < 85 % van de mediaan)
HKUST_STATIONS = ("SQ_Apartment37_48", "LSK_South", "SQ1", "SQ4", "SQ567")
HKUST_FAULT_MONTHS = {"SQ1": ("2021-05", "2023-05")}
# stralingssensor alleen betrouwbaar in deze periode (BESLISSINGEN #26)
HKUST_SENSOR_OK = ("2021-05-01", "2023-02-14 23:59")
HKUST_CLEAN = DATA / "extern" / "hkust" / "clean"
HKUST_MIN_SUB = (3, 10)                  # min. aantal 5-min PV-waarden / 1-min weerwaarden per kwartier
HKUST_WET_MONTHS = (5, 6, 7, 8, 9)       # moesson (nat) mei-sep; droog = okt-apr
# testjaar 2023 pas tonen na de keuze op 2022 en akkoord Dimme (BESLISSINGEN #28, AGENTS.md §6.2)
HKUST_TEST_VRIJGEGEVEN = True           # akkoord Dimme 2026-11-11 (na keuze #29)

# ---------- TAAK 5b: dag-vooruit en horizon, Open Climate Fix UK PV (2024-2025) ----------
# Bron: huggingface.co/datasets/openclimatefix/uk_pv (CC BY 4.0, doi:10.57967/hf/0878), data van Sheffield Solar.
# Toegang: gratis account + voorwaarden accepteren + HF_TOKEN als omgevingsvariabele (staat nooit in de code).
UKPV_HF = "hf://datasets/openclimatefix/uk_pv/30_minutely"
UKPV_RAW = DATA / "extern" / "uk_pv" / "raw"             # metadata.csv, bad_data.csv, README.md
UKPV_CLEAN = DATA / "extern" / "uk_pv" / "clean"
UKPV_RESULTS = ROOT / "resultaten" / "taak5" / "ukpv"
UKPV_PERIOD = ("2024-01-01", "2025-03-31")                # 15 hele maanden (april 2025 is maar half)
UKPV_N_SYSTEMS = 40                                       # taak: 20-50 systemen
UKPV_KWP = (1.5, 10.0)                                    # huis
UKPV_TILT = (10.0, 60.0)
UKPV_ORIENT = (90.0, 270.0)                               # oost .. zuid .. west (graden, 180 = zuid)
UKPV_MIN_ROWS = 0.98                                      # per maand >= 98 % van de halve uren aanwezig
UKPV_MAX_BAD_DAYS = 0.01                                  # < 1 % dagen met nachtopwek of nul-opwek bij zon
UKPV_YIELD = (550.0, 1150.0)                              # plausibele jaaropbrengst (kWh per kWp), BESLISSINGEN #35
UKPV_PMAX = (0.6, 1.2)                                    # plausibel maximum vermogen / kWp (klopt kWp?)
UKPV_POOL = 150                                          # kandidaten die we echt ophalen voor de kwaliteitscheck
UKPV_LAT_BANDS = (49.5, 51.0, 52.0, 53.0, 54.5, 61.0)    # 5 breedtebanden (zuid -> noord) voor de spreiding
# weer (BESLISSINGEN #37): cache per locatie / per run
UKPV_WEATHER = DATA / "extern" / "uk_pv" / "weer"
UKPV_DA_MODEL = "ecmwf_ifs025"                            # Previous Runs: dag-vooruit vanaf 6 mrt 2024
UKPV_HZ_MODEL = "ecmwf_ifs"                               # Single Runs: horizon, vanaf 14 mrt 2024
UKPV_HZ_RUNS = ("2024-03-14", "2025-03-31", "4D")         # run 00 UTC elke 4e dag
# tests (BESLISSINGEN #36, #38)
UKPV_FC_START = "2024-03-06"                              # eerste dag met dag-vooruit-voorspelling
UKPV_TRAIN_END = "2024-06-30"
UKPV_FOLDS = (("2024-03-06", "2024-04-30", "2024-05-01", "2024-05-31"),
              ("2024-03-06", "2024-05-31", "2024-06-01", "2024-06-30"))
UKPV_SEASONS = {7: "zomer", 8: "zomer", 9: "zomer", 10: "herfst", 11: "herfst", 12: "winter", 1: "winter",
                2: "winter", 3: "voorjaar"}
UKPV_TEST_VRIJGEGEVEN = True                              # akkoord Dimme 2026-11-13 (na keuze #39)

# ---------- TAAK 8: feedback pitch week 9 (BESLISSINGEN #67-#70, #75) ----------
TAAK8_RULE_PP = 0.3                                       # overnemen: >= 0,3 pp beter dan v2 en interval < 0
# Wanneer is de opbrengst van een dag bekend? lag = dagen tussen de laatste bekende dag en de doeldag.
# avond: voorspelling om 21:00 (na zonsondergang), vandaag is al bekend -> lag 1 (de app, AGENTS §13.3)
# strikt: voorspelling 24 uur vóór het doel (zoals de UK-test), vandaag is nog niet af -> lag 2
TAAK8_SCENARIOS = {"avond": 1, "strikt": 2}

# ---------- app (Streamlit, app/) ----------
# Officieel eindmodel voor de app (BESLISSINGEN #60): eindmodel v2. Ontbreekt het bestand, dan rekent de app met
# de standaardwaarden (pvlib zonder menggewichten) en toont een melding.
APP_MODEL_FILE = MODELS / "model_final_v2.joblib"
APP_MODEL_INFO = UKPV_RESULTS / "eindmodel_v2" / "model_info_v2.json"
APP_ROUGH_FROM_DAY = 5           # vanaf dag 5 vooruit (vandaag = dag 1) is de skill ~0 (UK-horizon, #40) -> "rough estimate"

# ---------- grafieken ----------
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29d"
INK, INK2 = "#0b0b0b", "#52514e"
PLANT_COLORS = {1: BLUE, 2: ORANGE}
DPI = 150

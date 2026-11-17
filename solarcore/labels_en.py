# -*- coding: utf-8 -*-
"""
Grafiektekst Nederlands -> Engels (titels, assen, legenda's, labels). Gebruikt door plots.save().
=================================================================================================
De scripts en de csv's houden hun Nederlandse namen; alleen wat in de grafiek staat wordt vertaald (de groep en de
docenten lezen Engels). Eerst hele zinsdelen (langste eerst), daarna losse woorden als vangnet. Teksten waarin na het
vertalen nog Nederlandse woorden staan, worden gelogd als de omgevingsvariabele SOLAR_TEXT_LOG een bestand noemt.
"""
import os
import re

PHRASES = {
    # modelnamen en baselines
    "Baseline: gemiddelde dag (K x heldere hemel)": "Baseline: mean day (K × clear sky)",
    "Baseline: persistence (gisteren)": "Baseline: persistence (yesterday)",
    "Baseline: gemiddelde dag": "Baseline: mean day",
    "Smart persistence (K blijft gelijk)": "Smart persistence (K stays the same)",
    "Fase 1: LR straling (sensor, bovengrens)": "Phase 1: LR irradiance (sensor, upper limit)",
    "Fase 2: smart persistence (day-ahead, geen weer)": "Phase 2: smart persistence (day-ahead, no weather)",
    "Fase 2: smart persistence (geen weer)": "Phase 2: smart persistence (no weather)",
    "LR straling + moduletemp.": "LR irradiance + module temp.",
    "LR fysisch: straling x (1 + γ(T-25))": "LR physical: irradiance × (1 + γ(T−25))",
    "Ridge (alle sensor-inputs + tijd)": "Ridge (all sensor inputs + time)",
    "Polynoom gr. 2 + Ridge (straling, moduletemp.)": "Polynomial deg. 2 + Ridge (irradiance, module temp.)",
    "Random Forest (alle)": "Random Forest (all)",
    "Gradient Boosting (alle)": "Gradient Boosting (all)",
    "LR op eerdere K": "LR on earlier K",
    "Ridge (K + sensor + tijd)": "Ridge (K + sensor + time)",
    "LR POA (straling op het paneel)": "LR POA (irradiance on the panel)",
    "LR EFF_T (POA + reflectie, spectrum, stof, temp.)": "LR EFF_T (POA + reflection, spectrum, soiling, temp.)",
    "e5: EFF_T, KT_POA, CS_POA, COSZ (LR met intercept)": "e5: EFF_T, KT_POA, CS_POA, COSZ (LR with intercept)",
    "e6: EFF_T x (a + b KT_POA), zonder intercept": "e6: EFF_T × (a + b KT_POA), no intercept",
    "e6: EFF_T x (a + b KT_POA)": "e6: EFF_T × (a + b KT_POA)",
    "Ridge e6 (zonder intercept)": "Ridge e6 (no intercept)",
    "e6 (zonder intercept)": "e6 (no intercept)",
    "Ridge e6 + bewolking, regen, vochtigheid": "Ridge e6 + cloud cover, rain, humidity",
    "Ridge e6 + bewolking, temp., wind": "Ridge e6 + cloud cover, temp., wind",
    "Ridge e6 + extra weer": "Ridge e6 + extra weather",
    "Sensor: LR straling": "Sensor: LR irradiance",
    "Sensor: Ridge (straling, temp., tijd)": "Sensor: Ridge (irradiance, temp., time)",
    "LR straling (plant-sensor, bovengrens)": "LR irradiance (plant sensor, upper limit)",
    "LR straling": "LR irradiance",
    "Zero-shot 2a: Kaggle-eindmodel (LR POA)": "Zero-shot 2a: Kaggle final model (LR POA)",
    "Zero-shot 2a: Kaggle-e6": "Zero-shot 2a: Kaggle e6",
    "Zero-shot 2b: pvlib-natuurkunde": "Zero-shot 2b: pvlib physics",
    "Zero-shot pvlib (kWp, dakhoek, richting)": "Zero-shot pvlib (kWp, tilt, orientation)",
    "LR GHI (satelliet, geschatte locatie)": "LR GHI (satellite, estimated location)",
    "LR GHI (satelliet, beste punt)": "LR GHI (satellite, best grid point)",
    "LR GHI (satelliet)": "LR GHI (satellite)",
    "Ridge (satelliet + ERA5)": "Ridge (satellite + ERA5)",
    "Kaggle-eindmodel x f": "Kaggle final model × f",
    "LR GHI alleen op de eerste N weken": "LR GHI on the first N weeks only",
    "LR GHI op 2021 + 2022 (stap 3)": "LR GHI on 2021 + 2022 (step 3)",
    "Gemiddelde dag (2021-2022)": "Mean day (2021–2022)",
    "pvlib x orakel-f": "pvlib × oracle f",
    "pvlib x f, doorlopend (alles sinds installatie)": "pvlib × f, rolling (everything since installation)",
    "pvlib x f, doorlopend (laatste": "pvlib × f, rolling (last",
    "pvlib x f: P10-P90 over huizen": "pvlib × f: P10–P90 over houses",
    "pvlib x f": "pvlib × f",
    "pvlib zonder kalibratie": "pvlib without calibration",
    "vaste f uit 13 weken": "fixed f from 13 weeks",
    "LR POA ECMWF (stap 3)": "LR POA ECMWF (step 3)",
    "LR gemiddelde POA (3 modellen)": "LR mean POA (3 models)",
    "LR op 3 POA's": "LR on 3 POAs",
    "LR POA getraind op ERA5": "LR POA trained on ERA5",
    "LR POA (stap 3, trainingsperiode nodig)": "LR POA (step 3, needs a training period)",
    "LR POA (zonder mengen)": "LR POA (no blending)",
    "LR POA gemengd (stap 4)": "LR POA blended (step 4)",
    "v2: + mengen met gemiddelde dag": "v2: + blending with mean day",
    "v2: alleen gemiddelde dag (4 weken)": "v2: mean day only (4 weeks)",
    "v2: pvlib x f (4 weken)": "v2: pvlib × f (4 weeks)",
    "v2 (gemengd)": "v2 (blended)",
    "alleen model (LR POA)": "model only (LR POA)",
    "alleen gemiddelde dag": "mean day only",
    "regel: model t/m dag": "rule: model up to day",
    "daarna gem. dag": "then mean day",
    "mengen (w_d uit training)": "blending (w_d from training)",
    "met voorspelling dag 2": "with forecast day 2",
    "met ERA5 (perfect weer)": "with ERA5 (perfect weather)",
    "Bovengrens:": "Upper limit:",
    "gemiddelde dag (trainingsperiode)": "mean day (training period)",
    "gemiddelde dag (geen weer)": "mean day (no weather)",
    "smart persistence (geen weer)": "smart persistence (no weather)",
    "standaard (u0 25, u1 6,84)": "default (u0 25, u1 6.84)",
    "gefit met ERA5-wind": "fitted with ERA5 wind",
    "gefit zonder wind (u1 = 0)": "fitted without wind (u1 = 0)",
    "gemeten wind (weerstation)": "measured wind (weather station)",
    "standaard u0 = 25": "default u0 = 25",
    # titels en zinnen
    "Fout per horizon - validatie binnen": "Error per horizon – validation within",
    "Fout per horizon - test": "Error per horizon – test",
    "Fout van het eindmodel": "Error of the final model",
    "EINDMODEL": "FINAL MODEL",
    "Eindvergelijking": "Final comparison",
    "(vgl.)": "(for comparison)",
    "mean of beide plants": "mean of both plants",
    "niet geschikt": "not suitable",
    "zonnestand (cos zenit)": "sun position (cos zenith)",
    "meer trainingsdagen": "more training days",
    "ERA5 (achteraf)": "ERA5 (hindsight)",
    "per testdag en per plant": "per test day and per plant",
    "bovengrens: sensor op hetzelfde moment (fase 1)": "upper limit: sensor at the same moment (phase 1)",
    "Correlatie van alle features (Pearson, overdag, beide plants, 34 dagen)":
        "Correlation of all features (Pearson, daytime, both plants, 34 days)",
    "Correlatie overdag (plant-sensor, beide plants)": "Correlation during the day (plant sensor, both plants)",
    "Dagprofiel van de opwek (alle 34 dagen)": "Daily profile of the power (all 34 days)",
    "Dagopbrengst per plant met de split-grens": "Daily yield per plant with the split boundary",
    "Dagopbrengst per station (7-daags gemiddelde); stippellijn = jaargrens (train / valideer / test)":
        "Daily yield per station (7-day mean); dashed line = year boundary (train / validate / test)",
    "DC/AC-verhouding per dag (Plant 1 na de fix DC / 10)": "DC/AC ratio per day (Plant 1 after the fix DC / 10)",
    "Hoeveel warmer is het paneel dan de lucht? (gemiddeld per kwartier)":
        "How much warmer is the panel than the air? (mean per quarter-hour)",
    ": opwek per omvormer per dag (ruwe data, inclusief uitval)": ": power per inverter per day (raw data, incl. failures)",
    ": straling tegen opwek": ": irradiance vs power",
    ": voorspelling voor morgen": ": forecast for tomorrow",
    ": fout per model en testdag": ": error per model and test day",
    "laatste 5 testdagen, met de band": "last 5 test days, with the band",
    "laatste 5 testdagen (overdag)": "last 5 test days (daytime)",
    "Autocorrelatie residu": "Autocorrelation of residuals",
    "(overdag, op volgorde)": "(daytime, in order)",
    "voorspeld - werkelijk": "forecast − actual",
    "Residu tegen tijd van de dag": "Residual vs time of day",
    "Residu tegen voorspelde waarde": "Residual vs forecast value",
    "Residu tegen helderheid (KT, sensor)": "Residual vs clearness (KT, sensor)",
    "Residu tegen helderheid ERA5 (KT)": "Residual vs ERA5 clearness (KT)",
    "Residu tegen bewolking (ERA5)": "Residual vs cloud cover (ERA5)",
    "Residu per seizoen": "Residual per season",
    "Verdeling van de residuen": "Distribution of the residuals",
    "Residuen van": "Residuals of",
    "KT = gemeten / heldere hemel (-)": "KT = measured / clear sky (-)",
    "Fase 3: ERA5-modellen tegen baselines en fase 1/2": "Phase 3: ERA5 models vs baselines and phases 1/2",
    "Hoe gevoelig voor fouten in de weersvoorspelling?": "How sensitive to errors in the weather forecast?",
    "gesimuleerde fout op de voorspelde straling (sigma, per dag x per uur)":
        "simulated error on the forecast irradiance (sigma, per day × per hour)",
    "Tijdafspraak ERA5: welke verschuiving past het best?": "ERA5 time alignment: which shift fits best?",
    "Tijdverschuiving ERA5 (alleen 2021)": "ERA5 time shift (2021 only)",
    "verschuiving ERA5-straling (minuten)": "shift of the ERA5 irradiance (minutes)",
    "correlatie met de opwek (r)": "correlation with the power (r)",
    "correlatie met de sensor (r)": "correlation with the sensor (r)",
    "Fout per model en plant": "Error per model and plant",
    "nRMSE per model en plant": "nRMSE per model and plant",
    "Leercurve: zelfde testdagen": "Learning curve: same test days",
    "aantal trainingsdagen (aandeel van 34)": "number of training days (share of 34)",
    "HKUST test 2023: retrain (blauw = gekozen), zero-shot (oranje) en baselines":
        "HKUST test 2023: retrain (blue = chosen), zero-shot (orange) and baselines",
    "nRMSE per maand in 2023 (lichtblauw = moesson)": "nRMSE per month in 2023 (light blue = monsoon)",
    "PR per uur t.o.v. de middag (lager = schaduw of ongunstige richting)":
        "PR per hour relative to noon (lower = shading or unfavourable orientation)",
    "PR / PR middag (-)": "PR / PR at noon (-)",
    "Ontbrekende 5-min-waarden overdag (na de start van het station)":
        "Missing 5-min values during the day (after the station started)",
    "Stralingssensor weerstation HKUST: per maand t.o.v. heldere hemel en ERA5":
        "Irradiance sensor HKUST weather station: per month vs clear sky and ERA5",
    "sensor P99 / heldere hemel (max)": "sensor P99 / clear sky (max)",
    "sensor / ERA5 (maandsom)": "sensor / ERA5 (monthly sum)",
    "einde betrouwbare sensor": "end of reliable sensor",
    "Windsnelheid per maand": "Wind speed per month",
    "Luchttemperatuur per maand": "Air temperature per month",
    "weerstation (10 m mast)": "weather station (10 m mast)",
    "Nieuw huis: hoeveel weken eigen meetdata zijn nodig? (HKUST 2023, 30 huizen)":
        "New house: how many weeks of own data are needed? (HKUST 2023, 30 houses)",
    "weken eigen meetdata na installatie": "weeks of own data after installation",
    "Test 2023: nRMSE per bron": "Test 2023: nRMSE per source",
    "Kaggle: moduletemperatuur, test dag 25-34": "Kaggle: module temperature, test days 25–34",
    "HKUST: vermogensfout vs u0 (validatie 2022)": "HKUST: power error vs u0 (validation 2022)",
    "UK PV: gekozen systemen": "UK PV: chosen systems",
    "na metadata-filter": "after metadata filter",
    "UK PV: correlatie opwek met straling per uur (40 systemen)":
        "UK PV: correlation of power with irradiance per hour (40 systems)",
    "UK PV, test jul 2024 - mrt 2025: echte dag-vooruit-voorspelling (40 huizen)":
        "UK PV, test Jul 2024 – Mar 2025: real day-ahead forecast (40 houses)",
    "nRMSE per maand (jul 2024 - mrt 2025)": "nRMSE per month (Jul 2024 – Mar 2025)",
    "Horizon: hoe ver vooruit helpt de weersvoorspelling? (ECMWF, testperiode)":
        "Horizon: how far ahead does the weather forecast help? (ECMWF, test period)",
    "dag vooruit (dag 1 = 0-24 uur na de run van 00 UTC)": "days ahead (day 1 = 0–24 hours after the 00 UTC run)",
    "Horizon: voorspelling mengen met de gemiddelde dag (UK, testperiode)":
        "Horizon: blending the forecast with the mean day (UK, test period)",
    "UK, test jul 2024 - mrt 2025: welk weermodel voor morgen? (blauw = gekozen)":
        "UK, test Jul 2024 – Mar 2025: which weather model for tomorrow? (blue = chosen)",
    "Dekking van de band per dag vooruit (test)": "Band coverage per day ahead (test)",
    "band per dag vooruit": "band per day ahead",
    "dag-1-band": "day-1 band",
    "voorspelling dag 1": "forecast day 1",
    "halfuur (alleen overdag)": "half hour (daytime only)",
    "Eindmodel v2 tegen LR POA: nRMSE per dag vooruit (UK, test)": "Final model v2 vs LR POA: nRMSE per day ahead (UK, test)",
    "India (Kaggle), test dag": "India (Kaggle), test days",
    "straling uit ERA5, satelliet of sensor": "irradiance from ERA5, satellite or sensor",
    "(wisselvallig)": "(changeable)",
    "Richting (180 = zuid)": "Orientation (180 = south)",
    "(gemiddelde, band = 25-75 %)": "(mean, band = 25–75 %)",
    "instraling plant-sensor (kW/m²)": "irradiance plant sensor (kW/m²)",
    "omvormer (gesorteerd op mediaan)": "inverter (sorted by median)",
    "AC-vermogen per omvormer (kW)": "AC power per inverter (kW)",
    "opwek per omvormer (kWh/dag)": "power per inverter (kWh/day)",
    "dag (1 = 15 mei 2020, 34 = 17 juni 2020)": "day (1 = 15 May 2020, 34 = 17 June 2020)",
    "tijd van de dag (uur, IST)": "time of day (hour, IST)",
    "module - lucht (°C)": "module − air (°C)",
    "vermogen (kW, huis 5 kWp)": "power (kW, house 5 kWp)",
    "nRMSE (% van capaciteit) - lager is beter": "nRMSE (% of capacity) – lower is better",
    "nRMSE (%) - lager is beter": "nRMSE (%) – lower is better",
    "% van capaciteit": "% of capacity",
    "% van de capaciteit": "% of capacity",
    "opbrengst (kWh per kW AC)": "yield (kWh per kW AC)",
    "aantal kwartieren": "number of quarter-hours",
    "hoe ver vooruit": "how far ahead",
    "dag vooruit": "days ahead",
    "droog (okt-apr)": "dry (Oct–Apr)",
    "moesson (mei-sep)": "monsoon (May–Sep)",
    "half bewolkt": "partly cloudy",
    "plant-sensor": "plant sensor",
    "ERA5-weer": "ERA5 weather",
    "ERA5-wind": "ERA5 wind",
    "eigen opwek": "own power",
    "sensor (fase 1)": "sensor (phase 1)",
    "voorspeld dag": "forecast day",
    "uur (IST)": "hour (IST)",
    "uur (HKT)": "hour (HKT)",
    "(geen weer)": "(no weather)",
}

WORDS = {
    "gemiddelde": "mean", "gemiddeld": "mean", "dag": "day", "dagen": "days", "uur": "hour", "uren": "hours",
    "tegen": "vs", "opwek": "power", "vermogen": "power", "straling": "irradiance", "voorspelling": "forecast",
    "voorspeld": "forecast", "bewolkt": "cloudy", "bewolking": "cloud cover", "helder": "clear", "werkelijk": "actual",
    "gemeten": "measured", "zonder": "without", "met": "with", "alleen": "only", "bovengrens": "upper limit",
    "weer": "weather", "vooruit": "ahead", "seizoen": "season", "maand": "month", "droog": "dry", "moesson": "monsoon",
    "omvormer": "inverter", "omvormers": "inverters", "fout": "error", "dekking": "coverage", "breedte": "width",
    "keuze": "choice", "gekozen": "chosen", "kwartier": "quarter-hour", "kwartieren": "quarter-hours",
    "aantal": "number", "residu": "residual", "residuen": "residuals", "tijd": "time", "huis": "house",
    "huizen": "houses", "temperatuur": "temperature", "lucht": "air", "paneel": "panel", "gefit": "fitted",
    "standaard": "default", "verschuiving": "shift", "correlatie": "correlation", "jaar": "year", "week": "week",
    "weken": "weeks", "kalibratie": "calibration", "mengen": "blending", "gewicht": "weight", "satelliet": "satellite",
    "systemen": "systems", "systeem": "system", "richting": "orientation", "dakhoek": "tilt", "opbrengst": "yield",
    "ontbrekend": "missing", "schaduw": "shading", "vouw": "fold", "testdag": "test day", "testdagen": "test days",
    "bevroren": "frozen", "laatste": "last", "beste": "best", "de": "the", "het": "the", "van": "of", "per": "per",
    "en": "and", "op": "on", "uit": "from", "naar": "to", "voor": "for", "geen": "no", "graden": "degrees",
    "gekozen": "chosen", "zomer": "summer", "herfst": "autumn", "winter": "winter", "voorjaar": "spring",
    "Dakhoek": "Tilt", "Grootte": "Size", "Systeem": "System", "Wind": "Wind", "geschatte": "estimated",
    "locatie": "location", "punt": "point", "weerstation": "weather station", "maandsom": "monthly sum",
    "jan": "Jan", "feb": "Feb", "mrt": "Mar", "apr": "Apr", "mei": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "okt": "Oct", "nov": "Nov", "dec": "Dec", "plant": "plant", "model": "model",
    "validatie": "validation", "binnen": "within", "eindmodel": "final model", "Fout": "Error",
    "Leercurve": "Learning curve", "fase": "phase", "Fase": "Phase",
    "instraling": "irradiance", "moduletemp": "module temp", "luchttemp": "air temp", "celtemp": "cell temp",
    "samen": "both", "beide": "both", "morgen": "tomorrow", "meer": "more", "trainingsdagen": "training days",
    "helderheid": "clearness", "luchtvochtigheid": "humidity", "neerslag": "precipitation", "gisteren": "yesterday",
    "drempel": "threshold", "geschikt": "suitable", "niet": "not", "verhouding": "ratio", "datum": "date",
    "lengtegraad": "longitude", "breedtegraad": "latitude", "achteraf": "hindsight", "stap": "step",
    "modellen": "models", "zonnestand": "sun position", "zenit": "zenith",
}
# Woorden die na het vertalen niet meer mogen voorkomen (voor het logboek); "in", "model", "per" zijn ook Engels.
_DUTCH = re.compile(r"\b(de|het|een|van|dag|dagen|uur|tegen|gemiddelde|opwek|straling|voorspel\w*|bewolk\w*|"
                    r"werkelijk|gemeten|zonder|alleen|weer|vooruit|seizoen|maand|omvormer\w*|fout|dekking|keuze|"
                    r"gekozen|kwartier\w*|aantal|residu|residuen|tijd|huis|huizen|lucht|paneel|gefit|standaard|jaar|"
                    r"validatie|binnen|eindmodel|fase|leercurve|instraling|moduletemp|luchttemp|celtemp|samen|beide|"
                    r"morgen|meer|helderheid|luchtvochtigheid|neerslag|gisteren|drempel|geschikt|niet|verhouding|"
                    r"datum|lengtegraad|breedtegraad|achteraf|stap|modellen|zonnestand|vgl|"
                    r"weken|mengen|gewicht|satelliet|systeem|richting|dakhoek|opbrengst|schaduw|vouw\w*|bevroren|"
                    r"laatste|beste|geen|en|op|uit|naar|voor|met|zomer|herfst|voorjaar|mrt|mei|okt)\b", re.I)
_PHRASES = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))
_WORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, WORDS), key=len, reverse=True)) + r")\b")


def translate(text):
    """Nederlandse grafiektekst -> Engels (zinsdelen eerst, dan losse woorden); onbekende tekst blijft staan."""
    if not text or not re.search("[A-Za-z]", text):
        return text
    out = text
    for nl, en in _PHRASES:
        if nl in out:
            out = out.replace(nl, en)
    out = _WORD_RE.sub(lambda m: WORDS[m.group(1)], out)
    log = os.environ.get("SOLAR_TEXT_LOG")
    if log and _DUTCH.search(out):
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{text!r} -> {out!r}\n")
    log_all = os.environ.get("SOLAR_TEXT_ALL")              # controle: alle vertaalde teksten
    if log_all:
        with open(log_all, "a", encoding="utf-8") as f:
            f.write(f"{out}\n")
    return out

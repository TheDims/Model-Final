# AGENTS.md – Eindmodel SolarHome Forecast (werkpipeline)

> Werkinstructie voor de AI-agent in VS Code (Claude Code, Copilot of Codex) die **samen met Dimme** het
> eindmodel van het project bouwt. Lees dit bestand helemaal voordat je iets doet. `CLAUDE.md` in deze map
> verwijst hiernaar. Bij twijfel geldt dit bestand boven `../AGENTS.md` (dat beschrijft V1–V5).

---

## 0. In het kort
- **Doel:** een eerlijk getest regressiemodel dat de zonne-opwek voorspelt, eerst op de Kaggle-data van
  Plant 1 en 2, daarna met een bestaand weermodel (ERA5) als input, zodat het voor een huis in India werkt.
- **Werkmap:** `Project/Model Final/`. Alles wat nieuw is komt hier.
  `../Model/` en `../Model V1/` t/m `../Model V5/` zijn **alleen-lezen**: referentie, ideeën en geteste functies.
- **Pipeline in 5 fases:**
  0. data inlezen, opschonen, checken
  1. regressie-benchmark op de Kaggle-data (**hoofdrun: eerste 70 % trainen, laatste 30 % testen**;
     run 1 met 30/70 is klaar en blijft als vergelijking, zie §6)
  2. de forecast perfectioneren (echt vooruit voorspellen, alleen Kaggle-data)
  3. hetzelfde met een weermodel (ERA5) als input in plaats van de plant-sensor
  4. eindmodel kiezen, opslaan en documenteren
- **Werk stap voor stap.** Na elke fase STOP je en laat je de resultaten zien (§8).

---

## 1. Projectcontext (kort)
- MADS4Engineering, ISEP 2026. Team: Woohyun Jin, Dimme Plug, Johannes Norrgård.
  Coaches: Benedita Malheiro, Manuel Silva, Pedro Guedes, Veríssimo Santos.
- Wiki-verslag: https://www.mads4e2026-wiki1.dee.isep.ipp.pt/doku.php?id=report
- Gebruiker: huiseigenaar in **India** met zonnepanelen en een smart home, vaak weinig technische kennis
  (persona's Anita, Rahul, Lakshmi, Vikram).
- User story 1: *"As a homeowner with solar panels, I want to view tomorrow's expected solar power generation
  so that I can plan my energy consumption."*
- Eisen uit de wiki die het model raken:
  - FR1 (Must): opwek van **morgen per uur** voorspellen. FR4 (Should): komende dagen. FR9 (Should): betrouwbaarheid (band).
  - O2: **meerdere modellen op dezelfde testdata vergelijken met MAE, RMSE en R²; beter dan een eenvoudige baseline.**
  - NFR6: open data en open-source software, bronnen vermelden. NFR7: SI-eenheden (kW, kWh, °C, W/m²).
  - Projectbrief: Europese wetgeving (GDPR, AI Act, ...) → het model gebruikt **geen persoonsgegevens**.
- Methodes: Design Thinking, **CRISP-DM** (data understanding → preparation → modelling → evaluation), Scrum.
- Prioriteit: **eerst een goed, eerlijk getest model.** De app (Streamlit, V5) komt pas daarna.

---

## 2. Hoe wij samenwerken
- **Dimme bouwt mee en wil het snappen.** Leg elke stap kort uit in eenvoudig Nederlands: wat je doet, waarom,
  en wat de uitkomst betekent. Geen grote blokken code zonder uitleg.
- **Plan eerst.** Begin elke fase met een kort plan (bestanden, functies, wat je gaat testen) en wacht op akkoord.
- **Klein beginnen:** eerst een simpele werkende versie + baseline, daarna pas verbeteren.
- **Eén ding tegelijk veranderen** en steeds vergelijken met de vorige stap en de baselines.
- **Vraag het** als een keuze echt van Dimme of het team is (zie §13). Doe geen aannames over de wiki of de docenten.
- **Testopzet en keuzeregel staan vast vóór je de testdata bekijkt** en worden achteraf niet veranderd.
- **Eerlijk rapporteren:** scores nooit mooier maken dan ze zijn; aannames altijd apart van feiten.
- Als iets niet werkt of een idee slechter is: gewoon zo opschrijven. Dat is ook een resultaat.

---

## 3. Omgeving
- **Python:** Anaconda, `C:\Users\dimme\anaconda3\python.exe` (Python 3.13, pandas, scikit-learn 1.7, pvlib,
  statsmodels, matplotlib). Het kale `python`-commando is Python 3.14 en is **niet** de werkomgeving.
  Sla modellen (`.joblib`) altijd met Anaconda op.
- **Shell:** Windows PowerShell 5.1 (geen `&&`, gebruik `;`). Paden bevatten spaties → altijd quoten.
- Matplotlib zonder scherm: `$env:MPLBACKEND="Agg"`.
- Scripts moeten werken met **F5 in Spyder** én in VS Code (cellen `# %%`, zie §9).
- Internet is alleen nodig om ontbrekende ERA5-data op te halen (Open-Meteo); daarna werken we uit de cache.

---

## 4. Mapstructuur (`Model Final/`)
```
Model Final/
├── AGENTS.md              ← dit bestand
├── CLAUDE.md              ← verwijst naar AGENTS.md
├── README.md              ← resultaten per fase (bijhouden!)
├── BESLISSINGEN.md        ← beslissingslog: datum, keuze, waarom, alternatief (voor het verslag)
├── requirements.txt
├── data/                  ← kopie Kaggle-csv's + ERA5-cache (uit ../Model V5/data/)
│   └── clean/             ← opgeschoonde data (gemaakt door fase 0)
├── solarcore/             ← gedeelde functies (één bron van waarheid)
│   ├── config.py          paden, split, plant-locaties, constanten
│   ├── data_prep.py       inlezen + opschonen + 15-min-rooster
│   ├── features.py        tijd/zonnestand/heldere hemel/lags
│   ├── physics.py         pvlib: POA, celtemperatuur, EFF_T, KT_POA
│   ├── weather.py         ERA5 (Open-Meteo archief) laden/cachen
│   ├── split.py           split op tijd (TRAIN_FRACTION), validatievouwen, walk-forward, leercurve
│   ├── models.py          modelladder (baselines, LR, Ridge, ...)
│   ├── evaluate.py        metrics, skill, band, residuen
│   └── plots.py           vaste stijl voor grafieken
├── fase0_data.py
├── fase1_benchmark_kaggle.py
├── fase2_forecast.py
├── fase3_weermodel.py
├── fase4_eindmodel.py
├── resultaten/split30/    ← run 1 (30/70), archief: niet meer overschrijven
├── resultaten/split70/    ← run 2 (70/30), hoofdrun: fase0/ … fase4/, heatmaps/
├── resultaten/leercurve/  ← 10/17/24/27 trainingsdagen op dezelfde testdagen
└── modellen/              ← model_final_split30.joblib, model_final_split70.joblib
```
- Functies uit `../Model V5/solarhome/` (vooral `plant_data.py`, `physics.py`, `weather.py`, `dayahead.py`,
  `venster.py`, `plant_test.py`) mag je **hergebruiken**: lees ze eerst, kopieer alleen wat nodig is,
  maak het leesbaar en zet in een commentaar waar het vandaan komt (`# overgenomen uit V5 solarhome/physics.py`).
- Geen notebooks als hoofdcode; wel `# %%`-cellen.

---

## 5. Data
### 5.1 Bronnen
- **Kaggle "Solar Power Generation Data"** (anikannal): 2 plants in India, **15 mei – 17 juni 2020 (34 dagen)**,
  per 15 min, 22 omvormers per plant.
  - `Plant_{1,2}_Generation_Data.csv`: `DATE_TIME, PLANT_ID, SOURCE_KEY (omvormer), DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD`
  - `Plant_{1,2}_Weather_Sensor_Data.csv`: `DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION`
- **ERA5** via Open-Meteo Historical API (CC BY 4.0), uurlijks, gecachet in
  `data/era5_{lat}_{lon}_2020-04-01_2020-06-18.csv`:
  `DATE_TIME, shortwave_radiation, direct_normal_irradiance, diffuse_radiation, temperature_2m,
  relative_humidity_2m, surface_pressure, wind_speed_10m, precipitation, cloud_cover, elevation`.
- **Geschatte locaties** (beste correlatie plant-sensor ↔ ERA5): Plant 1 ≈ 14,5°N 77,5°O (r = 0,89),
  Plant 2 ≈ 19,0°N 79,0°O (r = 0,92). Dit zijn **schattingen**, geen echte adressen.
- Tijd: alle tijden zijn **lokale tijd India (IST, Asia/Kolkata)**, zonder tijdzone-label. Houd dat zo.

### 5.2 Bekende datafouten en hoe we ze oplossen (zelfde regels als wiki/V5)
| Probleem | Plant | Oplossing |
|---|---|---|
| Datumformaat | 1 (generation) | `'%d-%m-%Y %H:%M'`; de andere bestanden `'%Y-%m-%d %H:%M:%S'`. Altijd expliciet `format=` meegeven. |
| DC_POWER ~10× te hoog | 1 | DC / 10. **AC_POWER is de target**, DC gebruiken we niet als input. |
| TOTAL_YIELD onbruikbaar | 2 | Niet gebruiken. DAILY_YIELD alleen ter controle. |
| Omvormer-uitval (0 kW terwijl de rest draait) | vooral 2 (~17 % van zonnige kwartieren) | Weglaten: we voorspellen **normale werking**. Storingen detecteren is een aparte functie. |
| Ontbrekende omvormers/tijdstempels | 1, 2 | **Gemiddelde per omvormer** i.p.v. som; kwartier met < 8 actieve omvormers → NaN. |
| Gaten in sensordata | 1, 2 | Volledig 15-min-rooster; gaten ≤ 1 uur interpoleren, langer → NaN. |
| IRRADIATION-eenheid | 1, 2 | Kaggle IRRADIATION is in **kW/m²**, ERA5 in **W/m²**. Omrekenen vóór je vergelijkt. |

### 5.3 Target
- `AC_KW` = gemiddeld AC-vermogen per actieve omvormer (kW), per kwartier, per plant.
- `FRAC` = `AC_KW / capaciteit`. **Capaciteit alleen uit de trainingsdagen** (bijv. 99e percentiel van `AC_KW`
  overdag in de training). Zo zijn Plant 1 en 2 en later een huis (× kWp) vergelijkbaar.
- Nacht: alleen evalueren **overdag** (cos zenit > 0,02). Wel nachtwaarden op 0 zetten in de voorspelling.

### 5.4 Wat we uit de Kaggle-notebook van Mehmet Isik wél/niet overnemen
- Wél: `check_df`-routine (shape, types, NA, kwantielen 5/50/95/99 %), uurprofiel, boxplots per omvormer,
  DC→AC-verhouding per dag (na de fix), temperatuur + straling per uur, modul- minus luchttemperatuur,
  7-daags voortschrijdend gemiddelde, correlatiematrix.
- Niet: Plant 1's ~10 % DC→AC als echt zien; opwek in "kW" plotten als het kWh is; heatmaps per weekdag of
  "jan–mrt" (34 dagen in mei–juni!); geen aandacht voor omvormer-uitval; vage conclusies.

---

## 6. Testprotocol (de kern – niet onderhandelbaar)

### 6.0 Runs
| Run | Split | Status |
|---|---|---|
| Run 1 `split30` | eerste 30 % kiezen (dag 1–10), laatste 70 % testen (dag 11–34) | klaar (28 okt). Blijft als scenario "weinig data". |
| **Run 2 `split70`** | **eerste 70 % trainen (dag 1–24, 15 mei – 7 juni), laatste 30 % testen (dag 25–34, 8–17 juni)** | **hoofdrun vanaf 29 okt** |
| Controle `leercurve` | trainen op de eerste **10 / 17 / 24 / 27** dagen, altijd testen op **dezelfde** dagen 28–34 (= de 80/20-split) | beantwoordt "is 80 % beter dan 70 %?" |

- **Waarom meer training?** Run 1 liet zien dat 10 trainingsdagen te weinig zijn voor modellen met meer inputs:
  bij e6 schommelden a en b sterk (collineariteit, r ≈ 0,97), het bevroren LR GHI voorspelde gemiddeld 4 % te laag,
  en RF/GB overfitten. In V4/V5 (14–29 trainingsdagen) was e6 wél het beste model. Op de testdagen van run 1 was
  e6 ook al beter (14,1 tegen 15,0 % nRMSE) en robuuster bij weerfouten.
- **Waarom 70/30 als hoofdrun en 80/20 alleen als controle?** Bij 80/20 blijven maar 7 testdagen × 2 plants
  = 14 plant-dagen over. De fout verschilt per dag sterk (std ~3–5 pp), dus één testuitslag op 7 dagen is erg
  onzeker. In de leercurve worden alle trainingsgroottes op **dezelfde** dagen getest; alleen zo kun je eerlijk
  zien of 27 dagen beter is dan 24.
- **Eerlijkheidsnoot:** de testdagen 25–34 zijn in run 1 al bekeken (ze zaten in dag 11–34). De test van run 2 is
  dus **niet meer blind**. Daarom leggen we de kandidaten en de keuzeregel van run 2 **vóór het draaien** vast in
  `BESLISSINGEN.md`, en noemen we dit in de README.

### 6.1 Split
- **Op tijd, in hele dagen, nooit willekeurig** (geen `shuffle`, geen `train_test_split` zonder `shuffle=False`).
- Instelling in `solarcore/config.py`: `TRAIN_FRACTION` (run 2: 0,70) en `RUN_NAME` (`"split70"`).
  `n_train = round(TRAIN_FRACTION × 34) = 24`. Bereken het in code uit de echte datums en print de grenzen.
  Met `TRAIN_FRACTION = 0.30` en `RUN_NAME = "split30"` moet run 1 exact terugkomen (regressietest).
- Dezelfde datums voor beide plants. Train standaard op **beide plants samen**; rapporteer **per plant** en samen.

### 6.2 Kiezen gebeurt alléén binnen de trainingsdagen
- Modelkeuze, features en hyperparameters kies je met **validatievouwen binnen dag 1–24** (expanding):
  train 1–14 → val 15–17; train 1–17 → val 18–20; train 1–20 → val 21–24.
  Leid de vouwen af uit `n_train` (niet hardcoden), zodat run 1 zijn eigen vouwen houdt (1–6 → 7–8; 1–8 → 9–10).
- **Keuzeregel (vooraf vast):** laagste gemiddelde nRMSE over de vouwen.
  Verschil < 0,5 procentpunt → kies de **eenvoudigste** (minste inputs; LR boven boommodellen).
  Een complex model (RF, GB, MLP) alleen als het ≥ 1 pp beter is én op **alle** vouwen beter.
- Alles wat uit data geschat wordt, komt **alleen uit de trainingsdagen**: capaciteit, dip-verhouding,
  ERA5-tijdverschuiving, heldere-hemelschaal, band, factor.
- Pas daarna kijk je naar de testdagen. **De test verandert de keuze niet.**

### 6.3 Testen (allebei rapporteren, apart)
- **A. Bevroren (hoofdbenchmark):** model getraind op dag 1–24, voorspelt dag 25–34 zonder bij te leren.
- **B. Walk-forward:** voor elke testdag een nieuw model op **alle eerdere dagen** (zoals V5).
- **Band P10–P90:** kalibreren op out-of-sample walk-forward-voorspellingen **binnen** de training
  (dag 11–24), dekking rapporteren op dag 25–34 (doel 80 %).

### 6.4 Lekken voorkomen (checklist bij elke fase)
- [ ] Capaciteit, schaal (scaler), heldere-hemelprofiel, dip-verhouding, tijdverschuiving, band en factor alleen
      uit trainingsdata.
- [ ] `sklearn.Pipeline` gebruiken zodat preprocessing alleen op de training wordt gefit.
- [ ] Lags en rollende gemiddelden gebruiken **alleen info vóór het voorspelmoment** (geen `center=True`,
      geen `shift(-1)`), en lekken niet van test naar training.
- [ ] In fase 3 en 4 **nooit** de plant-sensor (`IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`)
      als input. Een huis heeft die niet. Alleen als bovengrens/vergelijking (en om de training op te schonen).
- [ ] Geen plant-ID als feature (een huis heeft geen plant-nummer).
- [ ] Zet in elk resultaatbestand welke run, welke trainings- en welke testdagen.

### 6.5 Extra tests
- **Per plant** naast elkaar (benchmark Plant 1 ↔ Plant 2).
- **Kruis-plant ("ander dak"):** train op Plant 1, test op Plant 2 en andersom.
- **Stabiliteit:** per dag de nRMSE; mean, std en CV (= std/mean); heatmap model × dag.
- **Onzekerheid van de testuitslag:** bootstrap over testdagen (bv. 1000×) → 90 %-interval van de nRMSE en van
  het verschil tussen twee modellen. Verschil binnen het interval = niet aantoonbaar.
- **Gevoeligheid voor weerfouten** (fase 3): ruis van 10/20/30 % op de straling.
- **Leercurve** (§6.0): nRMSE, dagfout, bias en de coëfficiënten a en b tegen het aantal trainingsdagen.

---

## 7. Metrics (zelfde definities als V5)
- **nRMSE** = RMSE / capaciteit × 100 %, overdag, per kwartier → **hoofdmetric**. Ook **nMAE**.
- RMSE en MAE in kW (voorbeeldhuis 5 kWp), **R²** (wiki O2 vraagt MAE, RMSE, R²).
- **Dagfout** = gemiddelde |voorspelde − echte dagopbrengst| / gemiddelde echte dagopbrengst × 100 %.
- **Skill** = 1 − nRMSE_model / nRMSE_referentie, t.o.v. **persistence** (gisteren = morgen) en de
  **gemiddelde dag** (klimatologie uit de training).
- **Band P10–P90:** dekking = % kwartieren binnen de band (doel 80 %).
- **Geen MAPE** (ontploft bij opwek ≈ 0 in de ochtend/avond).
- **Residuen (verplicht vanaf fase 1):** residu vs tijd van de dag, vs voorspelde waarde, vs bewolking/KT,
  histogram, en autocorrelatie (ACF) – zoals in het college Evaluation Metrics, sectie 4.
  Vraag steeds: *wanneer en bij welk weer mist het model?*

Baselines die **altijd** in elke tabel staan: gemiddelde dag, persistence en (vanaf fase 2) smart persistence.

---

## 8. De fases

### Fase 0 – Data (CRISP-DM: data understanding + preparation)
- Map opzetten (§4), data kopiëren uit `../Model V5/data/`, `requirements.txt`, `README.md`, `BESLISSINGEN.md`.
- `solarcore/data_prep.py` met de regels uit §5.2; `check_df` per bestand.
- Output: `data/clean/plant_15min.csv` met per plant per kwartier: `DATE_TIME, PLANT, AC_KW, N_INV,
  IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE` + vlag `UITVAL`.
- Grafieken (`resultaten/fase0/`): uurprofiel, boxplot per omvormer, straling vs AC, correlatiematrix,
  dagopbrengst per dag per plant met de split-grens als verticale lijn.
- **STOP:** laat de datacheck en de grafieken zien.

### Fase 1 – Regressie-benchmark op de Kaggle-data (split volgens §6)
Vraag: *hoe goed verklaart regressie het vermogen uit de data van de plant zelf?*
Let op: dit is **geen echte voorspelling** (de sensor meet op hetzelfde moment). Het is de **bovengrens**
en de basis voor de rest.
- Inputs: `IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`, tijd van de dag (sin/cos), zonnestand.
- Modelladder (één stap tegelijk):
  1. baselines: gemiddelde dag, persistence
  2. LR op straling alleen
  3. LR op straling + moduletemperatuur
  4. LR natuurkundig: `IRR × (1 + γ·(T_mod − 25))` (γ ≈ −0,004 /°C, **aanname** tot getest)
  5. Ridge / polynoom graad 2
  6. ter vergelijking: Random Forest, Gradient Boosting (vaste `random_state=42`)
- Kiezen volgens §6.2, testen volgens §6.3 (A en B), plus §6.5 (per plant, kruis-plant, stabiliteit).
- Output: `resultaten/fase1/vergelijking.csv`, voorspeld-vs-werkelijk (3 voorbeelddagen: helder, half, bewolkt),
  residuplots, heatmap model × plant.
- **STOP** na de keuze (vóór de testdagen) en na de test.

### Fase 2 – Forecasting perfectioneren (alleen Kaggle-data, echt vooruit)
Vraag: *wat kunnen we voorspellen met alleen informatie die er op het voorspelmoment al is?*
- Voorspelmoment t0; horizons: **+15 min, +1 uur, +3 uur en day-ahead** (morgen per kwartier, gemaakt om 21:00).
- Inputs alleen ≤ t0: eerdere opwek, eerdere sensorwaarden, klok, zonnestand, heldere-hemelprofiel (pvlib
  Ineichen op de geschatte locatie, geschaald op de training).
- Werk met de **heldere-hemelindex** `k = AC_KW / AC_heldere_hemel` (haalt de dagcurve eruit).
- Modellen: persistence, **smart persistence** (k blijft gelijk), LR op lags van k + tijd, Ridge; RF/GB alleen ter vergelijking.
- Zelfde protocol (§6). Grafiek: **fout per horizon** (nRMSE vs horizon) voor alle modellen.
- Verwachte uitkomst (hypothese, nog niet getest): kort vooruit werkt goed, day-ahead zonder weer is nauwelijks
  beter dan de gemiddelde dag → dat is de motivatie voor fase 3.
- **STOP:** laat de horizon-grafiek en de tabel zien.

### Fase 3 – Weermodel als input (ERA5)
Vraag: *hoe goed is het model als we de plant-sensor vervangen door een bestaand weermodel?*
- ERA5 uit de cache (uurlijks) → 15 min. **Controleer eerst de tijdafspraak:** Open-Meteo-straling is het
  gemiddelde over het **voorgaande uur**. Bepaal met kruiscorrelatie (lag) tegen `IRRADIATION` hoe je moet
  verschuiven (V3–V5: `rad_shift_min`). Leg de keuze vast in `BESLISSINGEN.md`.
- Natuurkunde met pvlib (overnemen uit V5 `physics.py`/`dayahead.py`): heldere hemel (Ineichen), POA (Perez),
  celtemperatuur (Faiman), `EFF_T`, `KT_POA`.
- Feature-ladder (één stap tegelijk, telkens vergelijken):
  `GHI` → `POA` → `POA + temperatuurcorrectie (EFF_T)` → **e6** (`EFF_T`, `EFF_T·KT_POA`, LR zonder intercept;
  beste uit V4/V5) → eventueel extra (bewolking, regen, vochtigheid) alleen als het helpt.
- Vergelijk in één tabel: baselines, fase-1-bovengrens (sensor), fase-2-day-ahead (zonder weer), fase-3-modellen.
- Let op collineariteit: `EFF_T` en `EFF_T·KT_POA` hebben r ≈ 0,97. Met weinig trainingsdagen (run 1: 10) schommelen de
  coëfficiënten → toon a en b per vouw en test Ridge als alternatief.
- Gevoeligheid voor weerfouten (10/20/30 % ruis op de straling) en kruis-plant-test.
- **Eerlijk:** ERA5 is weer *achteraf*, geen voorspelling. De scores zijn een **bovengrens** voor echt gebruik.
  Echte archief-voorspellingen (Open-Meteo Historical Forecast API) bestaan pas vanaf ~2022.
- **STOP:** laat de vergelijkingstabel en de residuen zien.

### Fase 4 – Eindmodel
- Kies het eindmodel met de keuzeregel uit §6.2 (op basis van fase 3, want alleen dat werkt voor een huis).
- Train het gekozen model daarna op **alle 34 dagen** voor gebruik (de testscores blijven die van §6.3).
- Band P10–P90 per bewolkingsklasse (KT < 0,45 / 0,45–0,75 / > 0,75), gekalibreerd out-of-sample.
- Opslaan: `modellen/model_final_<RUN_NAME>.joblib` + `resultaten/model_info.json` (keuze, motivatie, scores,
  trainings-/testdagen, aannames). Naar een huis: `kW = FRAC × kWp × factor × verliesfactor` (factor uit
  de natuurkunde, **aanname** – zie V4).
- Houd het compatibel met de app van V5 (zelfde sleutels als `model_v5.joblib`: `model, features, factor,
  band_method, band, kt_bins, params, rad_shift_min, min_cosz`), zodat we het later kunnen inpluggen.
- README af + korte wikitekst (zie §10).
- **STOP:** eindtabel en keuze laten zien.

---

## 9. Code-afspraken
- Commentaar en uitleg in het **Nederlands**; namen van variabelen mogen Engels of Nederlands, maar consequent.
- Elk script begint zo (werkt met F5 in Spyder en in VS Code):
  ```python
  # %% Imports en paden
  from pathlib import Path
  try:
      ROOT = Path(__file__).resolve().parent
  except NameError:          # Spyder/interactief
      ROOT = Path.cwd()
  ```
- Constanten (split, locaties, drempels, kleuren) alleen in `solarcore/config.py`. Geen magische getallen in scripts.
- Kleine functies met een docstring van 1–3 regels. Geen dubbele code tussen fases: gedeeld → `solarcore/`.
- Vaste `random_state=42`. Resultaten als `.csv` (tabellen) en `.png` (≥ 150 dpi) in `resultaten/faseX/`.
- Grafieken: titel, asnamen **met eenheid**, legenda; kleuren blauw `#2a78d6`, oranje `#eb6834`, rest grijs.
- Elke fase schrijft `resultaten/faseX/samenvatting.json` (keuze, scores, trainings- en testdagen).
- Overschrijf geen resultaten van een andere fase.

---

## 10. Documentatie
- **README.md:** per fase wat we deden, de tabel, de keuze + motivatie, beperkingen, aannames apart van feiten.
- **BESLISSINGEN.md:** elke keuze met datum, wat, waarom, welk alternatief. Begin met:
  - split op tijd (§6) en waarom: run 1 30/70, run 2 70/30, leercurve als controle op 80/20;
  - keuzeregel (§6.2);
  - nRMSE als hoofdmetric, MAE/RMSE/R² erbij voor wiki-eis O2;
  - fase 1 gebruikt de plant-sensor alleen als bovengrens.
- **Wikiteksten** (alleen als Dimme erom vraagt): eenvoudig Engels (B1–B2), studentniveau, geen overdreven claims,
  meestal zonder getallen behalve "beter dan de baseline". **Modelkeuze en resultaten pas op de wiki na akkoord
  van Dimme** (de methode mag wel).
- Uitleg aan Dimme: informeel Nederlands, korte zinnen.

---

## 11. Wat we al weten uit V1–V5 (ter controle, niet direct vergelijkbaar: andere testopzet)
| Inzicht | Bron |
|---|---|
| LR op de plant-sensor: dagfout ~2,5 % → bovengrens, niet bruikbaar voor een huis | V1 |
| LR op weer-API-straling + zonnestand: dagfout ~7 % | V2, `Model/` |
| pvlib-natuurkunde + correctie: ~7,2 % | V3 |
| **e6** (LR zonder intercept op `EFF_T`, `EFF_T·KT_POA`): dagfout 5,6 %, nRMSE ~13,3 % | V4 |
| Baseline gemiddelde dag: dagfout ~17–18 %, nRMSE ~15–16 %; persistence nRMSE ~19 % | V4, V5 |
| RF, GB, MLP waren nooit duidelijk beter (kleine dataset); MLP instabiel | `Model/`, V4 |
| Trainingsvenster "alle eerdere dagen" is het stabielst; **het weer bepaalt de fout** (std tussen dagen ~3,4 pp ≫ verschil tussen vensters ~0,5 pp) | V5 |
| ERA5 wijkt af van de plant-sensor: per dag 12–17 %, per uur 35–43 % | wiki |
| Band P10–P90 haalt ~74–76 % op de keuzedagen (doel 80 %) | V4, V5 |
| Plant 2 is lastiger (uitval, noordelijker, mogelijk slechtere ERA5-match) – **aanname** | V5 |
| Met 10 trainingsdagen: keuze LR GHI (ERA5) 15,0 % nRMSE ≈ smart persistence; e6 op de test 14,1 % en robuuster bij weerfouten | run 1 |
| ERA5 wint op dagniveau (fout 4,9 tegen 7,4 %), maar binnen de dag mist iedereen ~13 % (timing van wolken) | run 1, `diagnose_era5.py` |
| 29 plant-brede dips bij Plant 2 weggelaten; ERA5-tijdverschuiving +15 min | run 1, BESLISSINGEN #10, #12 |
Als een nieuwe uitkomst sterk afwijkt van deze tabel: eerst zoeken naar een fout (lek, eenheid, tijdverschuiving).

---

## 12. Valkuilen
- Datumformaat Plant 1, DC-fout Plant 1, eenheid IRRADIATION (kW/m²) vs ERA5 (W/m²).
- Tijdverschuiving ERA5 (straling = gemiddelde voorgaande uur) → anders lijkt het model slechter dan het is.
- Capaciteit over alle dagen berekenen = lek. Scaler op alle data fitten = lek.
- `rolling(..., center=True)` of `shift(-1)` = toekomst in de features.
- Seizoen: alleen mei–juni (begin moesson). Geen conclusies over een heel jaar.
- Maar 34 dagen × 2 plants: verschillen < ~0,5 pp nRMSE kunnen toeval zijn.
- Pandas 3 (Python 3.14) vs pandas 2 (Anaconda) → altijd in Anaconda testen.
- `../Model V5/` scripts schrijven naar hun eigen `modellen/` en `resultaten/`: niet vanuit hier draaien.

---

## 13. Open punten – vraag Dimme als het zover is
1. **Welk eindmodel wordt het officiële?** `model_final_split70.joblib` of de run-1-versie? Pas beslissen na de
   vergelijking run 1 ↔ run 2 ↔ leercurve. `model_final.joblib` (run 1) niet overschrijven zonder akkoord.
2. Fase 1 met de plant-sensor als bovengrens: akkoord dat dit geen echte forecast is?
3. Welke day-ahead-uitgiftetijd (nu 21:00 IST)?
4. Na fase 3: echte archief-weersvoorspellingen (Open-Meteo Historical Forecast, ≥ 2022) en/of een lange
   dataset (UK PV, NREL PVDAQ) testen?
5. Wanneer koppelen we het eindmodel aan de app van V5?

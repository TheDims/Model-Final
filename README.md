# Model Final – SolarHome Forecast (eindmodel)

Eén eerlijk getest regressiemodel dat de zonne-opwek voorspelt: eerst op de Kaggle-data van Plant 1 en 2, daarna
met een weermodel (ERA5) als input, zodat het voor een huis in India werkt. Werkinstructie: `AGENTS.md`.
Alle keuzes met datum en motivatie: `BESLISSINGEN.md`.

De pipeline is twee keer gedraaid (zie **"Run 2 – 70/30"** onderaan):

| Run | Split | Resultaten | Eindmodel | Status |
|---|---|---|---|---|
| Run 1 `split30` | dag 1–10 kiezen, dag 11–34 testen | `resultaten/split30/` | `modellen/model_final_split30.joblib` (= `model_final.joblib`): LR GHI | ✅ archief |
| **Run 2 `split70`** | **dag 1–24 trainen, dag 25–34 testen** | `resultaten/split70/` | `modellen/model_final_split70.joblib`: LR POA | ✅ hoofdrun |
| Leercurve | 10 / 17 / 24 / 27 trainingsdagen, test dag 28–34 | `resultaten/leercurve/` | – (alleen meten) | ✅ |

| Fase | Script |
|---|---|
| 0 – data inlezen, opschonen, checken | `fase0_data.py` |
| 1 – regressie-benchmark op de Kaggle-data | `fase1_benchmark_kaggle.py` |
| 2 – forecast perfectioneren (alleen Kaggle-data) | `fase2_forecast.py` |
| 3 – weermodel (ERA5) als input | `fase3_weermodel.py` |
| 4 – eindmodel kiezen en opslaan | `fase4_eindmodel.py` |
| extra – leercurve, heatmaps, diagnose | `leercurve.py`, `heatmaps.py`, `diagnose_era5.py` |

**Runnen:** met de Anaconda-Python (`C:\Users\dimme\anaconda3\python.exe`), F5 in Spyder of in VS Code.
Standaard draait de hoofdrun `split70`. Een andere run: in PowerShell eerst `$env:SOLAR_RUN = "split30_check"`
(de regressietest; `split30` zelf is archief en kan niet opnieuw draaien). Welke testdagen al bekeken mogen worden,
staat per run in `solarcore/config.py` (`TEST_VRIJGEGEVEN`).

> **Let op:** de secties "Fase 0" t/m "Diagnose" hieronder beschrijven **run 1** (30/70). Run 2 staat onderaan.

---

## Fase 0 – Data (CRISP-DM: data understanding + preparation)

**Wat:** de 4 Kaggle-bestanden inlezen, controleren en opschonen tot één tabel per kwartier per plant:
`data/clean/plant_15min.csv` (6.528 rijen = 34 dagen × 96 kwartieren × 2 plants), met
`DATE_TIME, PLANT, DAY, AC_KW, N_INV, N_UITVAL, UITVAL, IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE`.
Samenvatting met alle tellingen: `resultaten/fase0/samenvatting.json`.

### Split
34 dagen → **opzet dag 1–10 (15–24 mei 2020)**, **test dag 11–34 (25 mei – 17 juni 2020)**.

### Datacheck (feiten)
| | Plant 1 | Plant 2 |
|---|---|---|
| Rijen generation / omvormers | 68.778 / 22 | 67.698 / 22 |
| Kwartieren zonder generation-data (van 3.264) | **106** | 5 |
| DC/AC vóór de fix → na de fix | 10,23 → **1,02** | 1,02 → 1,02 |
| Uitgevallen omvormer-rijen (weggelaten) | 63 | **3.365** |
| Kwartieren met minstens één uitval | 28 | **594** |
| AC_KW = NaN na opschonen | 108 | 5 |
| Plant-brede dips (vlag `DIP`, toegevoegd na fase 1) | 0 | **29** |
| Sensorgaten vóór → na interpolatie (≤ 1 uur) | 82 → 57 | 5 → 0 |
| Max. IRRADIATION | 1,22 kW/m² | 1,10 kW/m² |
| Correlatie straling ↔ AC (na opschonen) | 0,996 | 0,966 |

- **Geen dubbele rijen** en geen NA in de ruwe bestanden; de gaten zijn ontbrekende tijdstempels.
- **DC-fout Plant 1 bevestigd** (DC/AC 10,2 → na / 10: 1,02, gelijk aan Plant 2).
- **IRRADIATION is in kW/m²** (max 1,1–1,2). ERA5 is in W/m² → omrekenen in fase 3.
- **Uitval bij Plant 2: 12–13 % van de omvormer-kwartieren met echte zon** (sensor > 0,05–0,1 kW/m²; volgens onze
  mediaanregel 11–12 %). Het eerder genoemde getal van 17 % (wiki, `Model/`, AGENTS.md) geldt voor de grens
  "sensor > 0 kW/m²": dan tellen ook de schemer-kwartieren mee, waarin omvormers nog niet zijn opgestart.
  Plant 1 heeft bijna geen uitval (0,2 %).
- Plant 1 mist 106 kwartieren (~1 dag) aan generation-data en 57 kwartieren sensordata (gaten > 1 uur).

### Wat de grafieken laten zien (`resultaten/fase0/`)
| Grafiek | Wat we zien |
|---|---|
| `01_uurprofiel.png` | Beide plants hebben dezelfde dagvorm, piek rond 12:00–12:30 IST (klopt met ~78°O). Plant 2 heeft rond het middaguur een bredere spreiding (wolken en uitval). |
| `02_boxplot_per_omvormer.png` | Plant 1: alle omvormers vergelijkbaar. Plant 2: een paar omvormers leveren structureel minder (mediaan ~4.000 tegen ~6.500 kWh/dag), en er zijn veel lage uitschieters → uitval. |
| `03_straling_vs_ac.png` | Bijna een rechte lijn (r = 0,996 / 0,966). Bij Plant 2 nog punten ver onder de lijn bij veel zon: **gedeeltelijke uitval** die onze regel niet vangt (niet 0 kW, maar te laag). Aandachtspunt voor fase 1 (residuen). |
| `04_correlatiematrix.png` | Opwek hangt vooral samen met straling (0,96). Moduletemperatuur (0,89) en "module − lucht" (0,92) hangen zelf sterk samen met straling (0,92–0,95) → **collineariteit**: temperatuur voegt waarschijnlijk weinig toe als losse input. Luchttemperatuur 0,49. |
| `05_dagopbrengst_split.png` | Dagopbrengst schommelt sterk per dag (bewolkte dagen 18 en 28). De opzetperiode (dag 1–10) bevat zowel zonnige als bewolkte dagen. In de testperiode zitten de laagste dagen (Plant 2, dag 18 en 28) → de test is niet "makkelijker" dan de opzet. |
| `06_dc_ac_per_dag.png` | Na de fix DC/AC 1,020–1,024 voor beide plants: stabiel, dus de fix klopt. |
| `07_module_min_lucht.png` | Het paneel is rond het middaguur ~20 °C warmer dan de lucht; 's nachts ~1,5 °C kouder (uitstraling). Plant 1 iets warmer dan Plant 2. |

### Aannames fase 0 (apart van de feiten)
- De uitvalregel (≤ 0 kW terwijl de mediaan > 5 % van het maximum) vangt volledige uitval, niet gedeeltelijke uitval.
- Het tijdstip in de Kaggle-data is lokale tijd (IST) en het begin van het kwartier (zoals in V3–V5 getest).
- De eenheid van AC_POWER is kW per omvormer (consistent met DAILY_YIELD in kWh). Voor het model maakt dit niet uit,
  omdat we met FRAC = AC_KW / capaciteit werken.

---

## Fase 1 – Regressie-benchmark op de Kaggle-data (30/70)

**Vraag:** hoe goed verklaart regressie het vermogen uit de data van de plant zelf?
**Let op:** dit is **geen echte voorspelling**. De sensor meet op hetzelfde moment als de opwek. Dit is de
**bovengrens**: een huis heeft geen stralingssensor.

**Opzet:** target FRAC = AC_KW / capaciteit (99e percentiel overdag, alleen uit de trainingsdagen, per plant).
Alleen overdag (cos zenit > 0,02), alleen kwartieren waar alle inputs bekend zijn, en **zonder de 29 plant-brede
dips van Plant 2** (BESLISSINGEN #10): 3.403 kwartieren. Elk model wordt op precies dezelfde kwartieren beoordeeld.
Geen plant-ID als input. `random_state = 42`.

> **Eerlijk:** de plant-brede dips zijn gevonden in de residuen, dus **na** het zien van de testdata. Daarna is de
> hele fase opnieuw gedraaid (ook de keuze). De keuze bleef hetzelfde. De versie mét dips staat hieronder ook
> (`MET_DIPS = True` in het script).

### Keuze (alleen op dag 1–10, 2 validatievouwen; zonder dips)
| Model | nRMSE | vouw 1 (val. 7–8) | vouw 2 (val. 9–10) | R² | Dagfout |
|---|---|---|---|---|---|
| Baseline: gemiddelde dag | 13,2 % | 14,0 | 12,4 | 0,83 | 13,4 % |
| Baseline: persistence | 19,9 % | 26,3 | 13,5 | 0,57 | 13,6 % |
| **LR straling (gekozen)** | 6,5 % | 5,9 | 7,0 | 0,96 | 1,9 % |
| LR straling + moduletemp. | 6,3 % | 5,5 | 7,0 | 0,96 | 1,6 % |
| LR fysisch: straling × (1 + γ(T − 25)) | 6,7 % | 6,5 | 7,0 | 0,95 | 2,5 % |
| Ridge (alle sensor-inputs + tijd) | **6,2 %** | 5,6 | 6,8 | 0,96 | 2,2 % |
| Polynoom gr. 2 + Ridge | 6,4 % | 5,9 | 6,8 | 0,96 | 2,1 % |
| Random Forest | 6,6 % | 5,7 | 7,5 | 0,96 | 2,2 % |
| Gradient Boosting | 7,1 % | 6,0 | 8,2 | 0,95 | 1,9 % |

Regel: laagste nRMSE = Ridge (6,2 %), geen complex model. Binnen 0,5 pp: LR straling, LR + moduletemp., polynoom
→ de **eenvoudigste is LR straling** (1 input). Mét dips was de keuze ook LR straling (7,5 % tegen Ridge 7,1 %).

### Test op dag 11–34 (na de keuze, verandert de keuze niet)
| Model | A. Bevroren nRMSE | Plant 1 | Plant 2 | R² | Dagfout | Skill t.o.v. gem. dag | B. Walk-forward | Mét dips (A) |
|---|---|---|---|---|---|---|---|---|
| Baseline: gemiddelde dag | 16,8 % | 15,0 | 18,4 | 0,68 | 16,4 % | 0 | 15,9 % | 17,5 % |
| Baseline: persistence | 19,7 % | 19,4 | 20,0 | 0,56 | 13,9 % | −0,18 | 19,7 % | 20,3 % |
| **LR straling (gekozen)** | **5,5 %** | 4,1 | 6,6 | 0,97 | 2,9 % | 0,68 | 5,4 % | 7,9 % |
| LR straling + moduletemp. | 5,6 % | 4,3 | 6,6 | 0,96 | 3,7 % | 0,67 | 5,4 % | 8,1 % |
| LR fysisch | 5,1 % | 3,4 | 6,4 | 0,97 | 2,4 % | 0,69 | 5,1 % | 7,7 % |
| Ridge (alle) | 5,2 % | 3,7 | 6,4 | 0,97 | 2,8 % | 0,69 | 5,1 % | 7,9 % |
| Polynoom gr. 2 + Ridge | 5,2 % | 3,9 | 6,3 | 0,97 | 3,1 % | 0,69 | 5,0 % | 8,1 % |
| Random Forest | 5,7 % | 4,3 | 6,8 | 0,96 | 3,4 % | 0,66 | 5,2 % | 8,5 % |
| Gradient Boosting | 5,9 % | 3,9 | 7,4 | 0,96 | 2,4 % | 0,65 | 5,4 % | 9,0 % |

**Wat dit betekent**
- Met alleen de straling van de sensor is de fout **3× kleiner** dan met de gemiddelde dag (5,5 tegen 16,8 % nRMSE).
  Dat is de **bovengrens** voor het eindmodel.
- **De dips weglaten helpt ook Plant 1** (4,5 → 4,1 %), hoewel Plant 1 geen dips heeft: de dips in de opzetdagen
  van Plant 2 (dag 5–7) trokken het gezamenlijke model scheef.
- **Alle regressiemodellen liggen dicht bij elkaar** (5,1–5,9 %); RF en GB zijn niet beter. Bevestigt V1–V5.
- Op de testdagen zijn LR fysisch, Ridge en polynoom iets beter (5,1–5,2 %) dan de gekozen LR straling (5,5 %),
  op de validatie niet duidelijk. Verschil < 0,5 pp → de keuze blijft.
- **Walk-forward helpt nauwelijks** (5,5 → 5,4 %): met straling als input is 10 dagen al genoeg.
- **Kruis-plant** (bevroren, LR straling): P1 → P2 6,7 %, P2 → P1 4,7 %. Een model van de andere plant is maar
  iets slechter dan een model van beide plants (6,6 % en 4,1 %).
- **Stabiliteit** (48 plant-dagen, LR straling): nRMSE per dag gemiddeld 5,0 %, std 2,2, **CV 0,45**
  (mét dips: 6,7 %, std 4,1, CV 0,61). De baselines hebben een lagere CV (~0,3) maar een veel hogere fout.

### Residuen (`residuen.png`) – wanneer mist het model?
1. **Plant-brede dips bij Plant 2** (opwek < 30 % van verwacht terwijl de sensor zon meet): 29 kwartieren,
   ~1,7 % van Plant 2. Nu weggelaten. Mogelijke oorzaak (AANNAME): netstoring of afschakelen (curtailment).
   Er blijven nog **kleinere dips** over (residu +20 tot +60 %): de plant levert dan 30–80 % van verwacht.
2. **Systematische kromming:** bij weinig opwek (ochtend/avond) voorspelt LR ~3 % te veel (de intercept), in het
   middengebied ~5 % te weinig. Het verband straling → opwek is niet helemaal recht (temperatuur, zonnehoek).
   Ridge/polynoom/LR fysisch vangen dit iets beter (zie test).
3. **Autocorrelatie bij Plant 1:** de residuen hangen ~2,5 uur samen (ACF > 0 tot lag ~10). Fouten komen in
   blokken (een deel van de dag te hoog of te laag), niet als losse ruis. Bij Plant 2 korter (~1 uur).
4. Per bewolking (KT) geen duidelijk patroon.

### Aannames fase 1
- γ = −0,004 /°C in "LR fysisch" is een aanname (niet gefit).
- De dip-regel (sensor > 0,2 kW/m², opwek < 30 % van verwacht; verhouding uit dag 1–10) is een eigen keuze.
- De geschatte plantlocaties worden alleen gebruikt voor de zonnestand (COSZ) en de heldere-hemel-index (analyse).
- Grafieken: `heatmap_model_plant.png`, `heatmap_model_dag.png`, `voorbeelddagen.png`, `residuen.png`.

---

## Fase 2 – Forecasting perfectioneren (alleen Kaggle-data, echt vooruit)

**Vraag:** wat kunnen we voorspellen met alleen informatie die er op het voorspelmoment t0 al is?
Horizons: **+15 min, +1 uur, +3 uur** en **day-ahead** (morgen per kwartier, gemaakt om 21:00 de avond ervoor).
Inputs alleen ≤ t0: eerdere opwek, eerdere sensorwaarden, klok, zonnestand, heldere-hemelprofiel.

**Aanpak:** we voorspellen de **heldere-hemelindex K** = FRAC / FRAC bij heldere hemel (Ineichen × schaal per plant,
schaal uit de training) en vermenigvuldigen daarna met de heldere-hemel-opwek. Zo zit de dagcurve er al in en
leert het model alleen "hoe bewolkt". Lags alleen naar het verleden; ontbrekende lags (nacht) → gemiddelde K van
de vorige dag. Dips/uitval tellen niet mee (FRAC = NaN).

### Keuze per horizon (validatie dag 1–10, nRMSE gemiddeld over 2 vouwen)
| Model | +15 min | +1 uur | +3 uur | day-ahead |
|---|---|---|---|---|
| Baseline: gemiddelde dag | 13,2 % | 13,2 % | 13,2 % | 13,2 % |
| Baseline: persistence | 10,6 % | 20,8 % | 43,8 % | 20,9 % |
| Smart persistence (K blijft gelijk) | 8,9 % | 13,0 % | 14,2 % | **11,4 %** ✔ |
| LR op eerdere K | **8,6 %** ✔ | 12,1 % | 14,1 % | 14,9 % |
| Ridge (K + sensor + tijd) | 8,4 % | **11,1 %** ✔ | **12,6 %** ✔ | 15,7 % |
| Random Forest | 8,5 % | 11,3 % | 13,2 % | 14,8 % |
| Gradient Boosting | 9,2 % | 11,6 % | 15,6 % | 16,9 % |

✔ = gekozen (regel #2). +15 min: Ridge is het laagst, maar LR ligt binnen 0,5 pp en is eenvoudiger.

**Eerste indruk (alleen validatie, 4 dagen per plant – onzeker):**
- Kort vooruit (+15 min) helpt de geschiedenis duidelijk (8,6 tegen 13,2 % voor de gemiddelde dag).
- Hoe verder vooruit, hoe dichter bij de gemiddelde dag: bij +3 uur is het verschil nog ~0,6 pp.
- **Day-ahead zonder weer:** de regressiemodellen zijn slechter dan de gemiddelde dag (overfitting op 4–7
  trainingsdagen). Alleen smart persistence ("morgen even helder als vandaag") is iets beter.
- Daarna (akkoord Dimme) de testdagen 11–34, zie hieronder.
### Test op dag 11–34 (na de keuze; nRMSE, A = bevroren op dag 1–10, B = walk-forward)
| Model | +15 min | +1 uur | +3 uur | day-ahead |
|---|---|---|---|---|
| Baseline: gemiddelde dag | 16,8 % | 16,8 % | 16,8 % | 16,8 % (B: 15,9 %) |
| Baseline: persistence | 12,8 % | 21,2 % | 38,4 % | 19,9 % |
| Smart persistence | 11,6 % | 15,6 % | 17,8 % | **15,0 %** ✔ |
| LR op eerdere K | **10,8 %** ✔ | 13,3 % | 14,5 % | 14,7 % |
| Ridge (K + sensor + tijd) | 10,8 % | **13,6 %** ✔ | **14,9 %** ✔ | 15,0 % |
| Random Forest | 10,7 % | 13,8 % | 15,0 % | 16,1 % |
| Gradient Boosting | 11,3 % | 15,1 % | 16,2 % | 17,6 % |
| *Bovengrens fase 1 (sensor op hetzelfde moment)* | *5,5 %* | | | |

✔ = gekozen op dag 1–10. Walk-forward verandert bijna niets (max. ± 0,5 pp), behalve de gemiddelde dag (16,8 → 15,9 %).

| Gekozen model per plant (nRMSE) | +15 min | +1 uur | +3 uur | day-ahead | gem. dag |
|---|---|---|---|---|---|
| Plant 1 | 11,0 % | 13,5 % | 14,4 % | 15,0 % | 15,0 % |
| Plant 2 | 10,5 % | 13,7 % | 15,4 % | 14,9 % | 18,4 % |

Stabiliteit gekozen model (nRMSE per plant-dag, 48): +15 min 10,3 % (std 3,1; CV 0,30), +1 uur 13,1 % (CV 0,29),
+3 uur 14,3 % (CV 0,29), day-ahead 14,3 % (std 4,6; CV 0,32).

**Wat dit betekent**
- **Hypothese bevestigd.** Kort vooruit werkt de geschiedenis goed (+15 min: 10,8 tegen 16,8 %). Hoe verder vooruit,
  hoe kleiner het voordeel: +3 uur 14,9 %.
- **Day-ahead zonder weer is maar een beetje beter dan de gemiddelde dag:** 15,0 tegen 16,8 % (skill 0,11). Bij
  **Plant 1 helemaal niet** (15,0 tegen 15,0 %); bij Plant 2 wel (14,9 tegen 18,4 %). Dagfout day-ahead 13,9 % (gem. dag 16,4 %).
  In `voorbeeld_dayahead.png` zie je waarom: zonder weer voorspel je steeds een gladde "gemiddelde" curve; wolken
  van morgen kun je niet zien aankomen.
- Zelfs +15 min (10,8 %) is ruim 2× slechter dan de bovengrens met de sensor op hetzelfde moment (5,5 %).
- Op de test zijn LR en Ridge bij +1 en +3 uur ongeveer gelijk (verschil 0,2–0,4 pp, binnen de ruis).
- De testperiode is lastiger dan de validatie (bv. gemiddelde dag 16,8 tegen 13,2 %; meer bewolkte dagen, begin moesson).
- **Conclusie voor fase 3:** voor "morgen" (user story 1) is een **weersvoorspelling nodig**.

### Aannames fase 2
- Schaal heldere hemel = 90e percentiel van FRAC / CS_GHI (zon hoog) in de training; K begrensd op 0–1,5.
- Day-ahead gemaakt om 21:00: de hele dag D−1 is dan bekend (daggemiddelden van K en sensor).
- Ontbrekende lags (nacht, gaten) → gemiddelde K van de vorige dag (of van de training).
- Grafieken: `horizon_validatie.png`, `horizon_test.png`, `voorbeeld_dayahead.png`.
---

## Fase 3 – Weermodel (ERA5) als input

**Vraag:** hoe goed is het model als we de plant-sensor vervangen door een bestaand weermodel?
ERA5 = weer **achteraf** (heranalyse), dus een **bovengrens** voor gebruik met een echte weersvoorspelling.
Natuurkunde (pvlib) overgenomen uit V5: uur → kwartier, Perez-POA, Faiman-celtemperatuur, Kimber-stof, EFF_T,
KT_POA. Opstelling plants: vast, hoek = breedtegraad, zuid (AANNAME uit V3–V5). Nooit de plant-sensor als input.

### Tijdafspraak (alleen dag 1–10)
Correlatie ERA5-GHI ↔ sensor per kwartier is het hoogst bij **+15 min** (P1 r = 0,863; P2 0,939), dus de straling
van uur h hoort bij het interval met midden h + 15 min. Bij −30 min (de documentatie: "gemiddelde van het voorgaande
uur") is r veel lager (0,79 / 0,91). Zie `tijdafspraak.png` en BESLISSINGEN #12.

### Keuze (validatie binnen dag 1–10, nRMSE gemiddeld over 2 vouwen)
| Model | nRMSE | vouw 1 | vouw 2 | R² | Dagfout |
|---|---|---|---|---|---|
| Baseline: gemiddelde dag | 13,2 % | 14,0 | 12,4 | 0,83 | 13,4 % |
| Baseline: persistence | 20,9 % | 28,3 | 13,5 | 0,52 | 14,4 % |
| *Fase 2: smart persistence (geen weer)* | *11,4 %* | 12,0 | 10,8 | 0,87 | 10,4 % |
| *Fase 1: LR straling (sensor, bovengrens)* | *6,5 %* | 5,9 | 7,0 | 0,96 | 1,9 % |
| **LR GHI (ERA5) – gekozen** | **11,2 %** | 11,0 | 11,4 | 0,87 | 8,1 % |
| LR POA | 11,4 % | 11,2 | 11,6 | 0,87 | 9,2 % |
| LR EFF_T | 12,6 % | 12,5 | 12,8 | 0,84 | 11,6 % |
| e6 (V4): EFF_T × (a + b·KT_POA), zonder intercept | 14,5 % | 15,8 | 13,2 | 0,79 | 15,8 % |
| Ridge e6 (zonder intercept) | 14,5 % | 15,7 | 13,2 | 0,79 | 15,8 % |
| Ridge e6 + bewolking, regen, vochtigheid | 13,2 % | 15,4 | 11,0 | 0,82 | 10,9 % |
| Random Forest (e6 + extra) | 16,0 % | 20,3 | 11,8 | 0,73 | 14,8 % |
| Gradient Boosting (e6 + extra) | 18,9 % | 24,0 | 13,8 | 0,62 | 17,5 % |

**Eerste indruk (alleen 4 validatiedagen per plant – onzeker):**
- Met ERA5 is de fout veel groter dan met de sensor (11,2 tegen 6,5 %): **de weerdata is de bottleneck**, niet het model.
- Op de opzetdagen is ERA5 maar iets beter dan de gemiddelde dag (11,2 tegen 13,2 %) en ongeveer gelijk aan fase 2.
- **e6 (de V4-winnaar) werkt hier niet beter.** Gecontroleerd op fouten (alleen binnen dag 1–10): ook in-sample op
  dag 1–10 zijn LR GHI, LR POA en e6 gelijk (13,8 / 13,8 / 14,3 %). Met 6–8 trainingsdagen schommelen a en b sterk
  (b: −0,0012 ↔ −0,0006 per vouw; EFF_T en EFF_T·KT_POA r ≈ 0,97). ERA5 heeft per dag een flinke afwijking
  (verhouding opwek/ERA5-POA per dag 0,79–1,18) die alle modellen evenveel raakt.
- Complexe modellen (RF, GB) overfitten duidelijk met zo weinig dagen.
- De testdagen 11–34 zijn nog niet bekeken (`TOON_TEST = False`).
### Test op dag 11–34 (na de keuze; verandert de keuze niet)
| Model | A. Bevroren nRMSE | Plant 1 | Plant 2 | R² | Dagfout | Skill t.o.v. gem. dag | B. Walk-forward |
|---|---|---|---|---|---|---|---|
| Baseline: gemiddelde dag | 16,8 % | 15,0 | 18,4 | 0,68 | 16,4 % | 0 | 15,9 % |
| Baseline: persistence | 19,9 % | 19,4 | 20,4 | 0,55 | 14,0 % | −0,19 | 19,9 % |
| *Fase 2: smart persistence (geen weer)* | *15,0 %* | 15,0 | 14,9 | 0,74 | 13,9 % | 0,11 | 15,0 % |
| *Fase 1: LR straling (sensor, bovengrens)* | *5,5 %* | 4,1 | 6,6 | 0,97 | 2,9 % | 0,67 | 5,4 % |
| **LR GHI (ERA5) – gekozen** | **15,0 %** | 15,4 | 14,7 | 0,74 | 12,8 % | 0,10 | 14,7 % |
| LR POA | 14,9 % | 15,2 | 14,5 | 0,75 | 12,2 % | 0,11 | 14,5 % |
| LR EFF_T | 14,5 % | 14,4 | 14,6 | 0,76 | 10,3 % | 0,13 | 14,4 % |
| e6: EFF_T × (a + b·KT_POA) | **14,1 %** | 14,0 | 14,3 | 0,77 | **9,6 %** | **0,16** | 14,1 % |
| Ridge e6 | 14,2 % | 14,0 | 14,3 | 0,77 | 9,6 % | 0,16 | 14,1 % |
| Ridge e6 + bewolking, regen, vochtigheid | 16,1 % | 15,7 | 16,6 | 0,70 | 13,1 % | 0,04 | 14,0 % |
| Random Forest | 16,1 % | 16,2 | 16,0 | 0,70 | 13,8 % | 0,04 | 14,3 % |
| Gradient Boosting | 16,5 % | 16,7 | 16,2 | 0,69 | 13,7 % | 0,02 | 14,7 % |

**Stabiliteit** (nRMSE per plant-dag, 48): LR GHI 14,6 % (std 3,6; CV 0,25), e6 13,8 % (std 3,4; CV 0,24),
smart persistence 14,3 % (std 4,6; CV 0,32), gemiddelde dag 15,9 % (std 5,5; CV 0,34).
**Kruis-plant** (bevroren): LR GHI P1 → P2 14,9 %, P2 → P1 15,3 %; e6 15,1 / 15,0 %.

**Gevoeligheid voor weerfouten** (ruis per dag × per uur op de ERA5-straling, testdagen, 5 herhalingen):
| Fout | LR GHI (gekozen) | e6 | Ridge e6 + extra | gem. dag (geen weer) |
|---|---|---|---|---|
| 0 % | 15,0 % | 14,1 % | 16,1 % | 16,8 % |
| 10 % | 16,2 % | 14,7 % | 16,4 % | 16,8 % |
| 20 % | 18,4 % | 16,2 % | 17,0 % | 16,8 % |
| 30 % | 20,3 % | 17,6 % | 18,1 % | 16,8 % |

**Wat dit betekent**
- **Met ERA5 is day-ahead maar iets beter dan zonder weer.** Het gekozen LR GHI (15,0 %) is gelijk aan fase 2
  (smart persistence, 15,0 %) en 1,7 pp beter dan de gemiddelde dag. De sensor (5,5 %) laat zien hoeveel winst er
  zou zijn met perfect weer: **de weerdata is de bottleneck**, zoals in V1–V5.
- **Op de testdagen is e6 (V4) het beste ERA5-model:** 14,1 % nRMSE en een dagfout van 9,6 % (LR GHI 12,8 %). Op de
  validatie (met maar 6–8 trainingsdagen) was e6 slechter; de keuzeregel koos daarom LR GHI. De test verandert de
  keuze niet (protocol), maar het patroon klopt met V4/V5: **e6 heeft meer trainingsdagen nodig** (in het bevroren
  model 10 dagen; in V4/V5 14–29 dagen).
- **e6 is ook robuuster voor weerfouten:** bij 20 % fout 16,2 % tegen 18,4 % voor LR GHI. LR GHI wordt al bij ~12 %
  weerfout slechter dan de gemiddelde dag.
- Extra weervariabelen (bewolking, regen, vochtigheid) en RF/GB helpen bevroren niet (overfitting op 10 dagen); met
  walk-forward (meer dagen) komen ze wel op 14,0–14,7 %.
- **Residuen** (`residuen.png`): het model voorspelt op de testdagen gemiddeld te laag (veel residuen −10 tot −30 %),
  ook bij "100 % bewolking" volgens ERA5. ERA5 zag dus vaker wolken dan er bij de plant waren (rastercel ~25 km,
  geschatte locatie). De residuen hangen ~2–3 uur samen (ACF): fouten komen in blokken, typisch voor verkeerd
  getimede wolken.

### Aannames fase 3
- Opstelling plants: vast, hoek = breedtegraad, zuid (V3–V5). Natuurkunde-parameters uit V5 (γ −0,37 %/°C, ...).
- ERA5 als stand-in voor een weersvoorspelling; de foutgroottes in de weerfout-test zijn aannames.
- Grafieken: `tijdafspraak.png`, `vergelijking.png`, `voorbeelddagen.png`, `residuen.png`, `weerfout.png`.
---

## Fase 4 – Eindmodel

**Gekozen: LR GHI (ERA5)** – een Linear Regression op alleen de horizontale straling uit het weermodel.
Gekozen met de vaste keuzeregel op dag 1–10 (fase 3). Op de testdagen was e6 iets beter; volgens het protocol
verandert de test de keuze niet, en Dimme koos ervoor de regel te volgen (BESLISSINGEN #14).

- **Formule** (getraind op alle 34 dagen, beide plants, zonder dips):
  FRAC = 0,0459 + 0,000902 × GHI (W/m²); 's nachts (cos zenit ≤ 0,02) = 0.
- **Naar een huis:** kW = FRAC × kWp × **factor 0,664** × verliesfactor (schaduw, leeftijd), begrensd op de
  omvormer. De factor is een **AANNAME**: het 99e percentiel van de pvlib-opwek (kW/kWp) op de plantlocaties
  (Plant 1: 0,684; Plant 2: 0,643).
- **Band P10–P90** per bewolkingsklasse (ERA5-KT), × voorspelling: bewolkt 0,66–2,52; half bewolkt 0,63–1,60;
  helder 0,55–1,30. Out-of-sample gekalibreerd. Check: kalibratie op dag 11–22 → **dekking 80,5 %** op dag 23–34
  (doel 80 %). De **dagband** (som P10 tot som P90) bevat de echte dag in 100 % van de gevallen: te ruim
  (kwartier-kwantielen mag je niet optellen) → verbeterpunt.
- **Opgeslagen:** `modellen/model_final.joblib` (zelfde sleutels als `model_v5.joblib`: model, features, factor,
  band_method, band, kt_bins, params, rad_shift_min, min_cosz) en `resultaten/model_info.json`
  (keuze, motivatie, scores, trainings-/testdagen, aannames).

### Eindtabel (test dag 11–34, bevroren model op dag 1–10, huis 5 kWp)
| Model | Input | nRMSE | nMAE | RMSE | MAE | R² | Dagfout | Skill t.o.v. gem. dag |
|---|---|---|---|---|---|---|---|---|
| Baseline: gemiddelde dag | geen | 16,8 % | 11,9 % | 0,84 kW | 0,60 kW | 0,68 | 16,4 % | 0 |
| Baseline: persistence | geen | 19,9 % | 13,3 % | 0,99 kW | 0,66 kW | 0,55 | 14,0 % | −0,19 |
| Fase 2: smart persistence (geen weer) | eigen opwek | 15,0 % | 10,7 % | 0,75 kW | 0,53 kW | 0,74 | 13,9 % | 0,11 |
| **Fase 3: LR GHI (ERA5) – EINDMODEL** | ERA5-weer | **15,0 %** | 11,1 % | 0,75 kW | 0,55 kW | 0,74 | 12,8 % | 0,10 |
| Fase 3: e6 (alternatief) | ERA5 + pvlib | 14,1 % | 10,4 % | 0,71 kW | 0,52 kW | 0,77 | 9,6 % | 0,16 |
| Fase 1: LR straling (bovengrens) | plant-sensor | 5,5 % | 3,5 % | 0,27 kW | 0,18 kW | 0,97 | 2,9 % | 0,67 |

Per plant (eindmodel): Plant 1 15,4 %, Plant 2 14,7 % nRMSE. Walk-forward (elke dag opnieuw op alle eerdere
dagen): 14,7 %.

### Conclusie
- Het eindmodel is **beter dan de baseline** (gemiddelde dag): 15,0 tegen 16,8 % nRMSE, dagfout 12,8 tegen 16,4 %.
- De winst is bescheiden: met deze ERA5-data is het eindmodel even goed als "morgen even helder als vandaag" (fase 2).
- **De weerdata is de bottleneck**, niet het model: met de plant-sensor (perfect weer) is de fout 3× kleiner.
- **Aanbeveling:** zodra er meer data is (meer dagen, of een huis met eigen meetdata), **e6** opnieuw testen:
  dat gebruikt de straling op het schuine paneel (dakhoek + richting) en was op de test en bij weerfouten beter.

### Beperkingen
- 34 dagen in mei–juni (begin moesson), 2 plants, 10 opzetdagen. Geen conclusies over een heel jaar.
- ERA5 is weer **achteraf**; een echte voorspelling heeft meer fout (bij ~12 % weerfout is LR GHI al slechter dan de
  gemiddelde dag, fase 3). Testen met archief-voorspellingen (Open-Meteo Historical Forecast, vanaf ~2022) kan pas
  met recente opwekdata.
- **LR GHI kijkt niet naar de dakhoek en richting** van een huis: die tellen alleen mee via kWp en verliezen.
  Voor een plat of zuidgericht dak is dat geen probleem; voor een oost/west-dak wel (e6 zou dat meenemen).
- Plantlocaties en opstelling zijn geschat; de factor 0,664 is niet gecontroleerd met data van een echt huis.

### Aannames fase 4
- Factor 0,664 (zie boven); band gekalibreerd op 24 out-of-sample dagen (2 × 24 plant-dagen).
- Grafieken: `eindvergelijking.png`, `band_voorbeeld.png`.
---

## Heatmaps voor het verslag (`heatmaps.py` → `resultaten/heatmaps/`)
Zelfde opzet als fase 3/4 (bevroren model, getraind op dag 1–10, getest op dag 11–34). Geen nieuwe keuzes.

| Bestand | Wat | Wat valt op |
|---|---|---|
| `01_correlatie_alle_features.png` | Correlatie (Pearson) van alle 20 features met elkaar en met de opwek, overdag, beide plants | Sensor-straling 0,98 met de opwek; ERA5-straling, POA, EFF_T en pvlib-opwek ~0,88 (onderling 0,97–1,00: bijna hetzelfde). ERA5-bewolking (−0,17), -helderheid (0,15), wind en neerslag zeggen per kwartier weinig. Luchtvochtigheid −0,45 (vooral via temperatuur, −0,94). |
| `02_fout_eindmodel_dag_plant.png` | nRMSE van het eindmodel per testdag × Plant 1 / Plant 2 / samen | De fout wisselt sterk per dag (8–26 %). Samen is dag 18 het moeilijkst (Plant 2: 26 %); bij Plant 1 is dag 34 de slechtste (22 %). Dag 14–15 zijn voor beide plants het makkelijkst (8–9 %). |
| `03_fout_model_dag_plant1.png`, `..._plant2.png` | nRMSE per model × testdag, per plant | Alle ERA5-modellen gaan samen op en neer: het **weer van de dag** bepaalt de fout. De sensor (bovengrens) blijft bijna overal laag. |
| `04_fout_model_plant.png` | nRMSE per model × plant over de hele testperiode | Eindmodel 15,4 % (P1) en 14,7 % (P2); gemiddelde dag 15,0 / 18,4 %; sensor 4,1 / 6,6 %. |

De tabellen staan ernaast als `.csv` (`fout_per_dag_per_plant.csv` met de nRMSE per plant-dag voor alle modellen).
---

## Diagnose: waarom scoort ERA5 even goed als "zonder weer" (beide ~15 % nRMSE)? (`diagnose_era5.py`)
Alleen uitleg, geen nieuwe keuze. Test dag 11–34, bevroren op dag 1–10. Geen fout gevonden.

**1. ERA5 weet wél veel beter hoe zonnig de dag wordt.** Correlatie van de echte daghelderheid met ERA5 van vandaag
0,85 (P1 0,70; P2 0,91), met de gemeten helderheid van gisteren (wat smart persistence gebruikt) maar 0,54.

**2. Maar per kwartier telt vooral iets wat niemand kan voorspellen: wanneer precies een wolk voorbij komt.**
| Model | nRMSE | bias | fout dagniveau | fout binnen de dag | aandeel binnen de dag |
|---|---|---|---|---|---|
| gemiddelde dag | 16,8 % | +3,4 % | 8,5 % | 14,1 % | 70 % |
| smart persistence (geen weer) | 15,0 % | +0,5 % | 7,4 % | 13,0 % | 75 % |
| **LR GHI (ERA5) – eindmodel** | 15,0 % | **−4,1 %** | **4,9 %** | 13,6 % | 82 % |
| e6 (ERA5) | 14,2 % | −2,0 % | 4,8 % | 13,2 % | 87 % |
| sensor (bovengrens) | 5,5 % | −0,9 % | 1,3 % | 5,2 % | 92 % |

- **Dagniveau:** ERA5 is duidelijk beter (4,9 tegen 7,4 %).
- **Binnen de dag:** alle voorspellers missen ~13–14 %. ERA5 is hier zelfs iets slechter dan een gladde curve: als
  ERA5 een wolk op het verkeerde uur zet, krijg je de fout twee keer (dip voorspeld waar zon was, en zon waar de dip
  was). Een gladde curve "gokt" het midden en wordt daarvoor minder gestraft.
- **Vaste afwijking:** het bevroren LR GHI voorspelt op de testdagen gemiddeld 4,1 % te laag (getraind op 10 dagen,
  niet bijgeleerd). Smart persistence kalibreert zichzelf elke dag met de gemeten opwek van gisteren (bias 0,5 %).
- Netto heffen die effecten elkaar per kwartier op → beide ~15 %.

**3. Op grovere tijdschaal wint ERA5 wel:**
| Model | per kwartier | per uur | per dag |
|---|---|---|---|
| gemiddelde dag | 16,8 % | 14,2 % | 9,2 % |
| smart persistence (geen weer) | 15,0 % | 12,3 % | 7,5 % |
| **LR GHI (ERA5) – eindmodel** | 15,0 % | 12,3 % | **6,4 %** |
| e6 (ERA5) | 14,2 % | 11,3 % | 5,2 % |
| sensor (bovengrens) | 5,5 % | 4,1 % | 1,6 % |

Voor user story 1 ("hoeveel opwek morgen?") telt vooral het dagtotaal: daar is ERA5 beter (ook de dagfout: 12,8 tegen
13,9 %). Ter informatie (verandert de keuze niet): zonder de vaste afwijking zou LR GHI 14,5 % halen, en het
gemiddelde van LR GHI en smart persistence 13,7 % → de twee bevatten **verschillende** informatie.

---

# Run 2 – 70/30 (hoofdrun, 29 okt 2026)

**Waarom opnieuw?** Run 1 liet zien dat 10 trainingsdagen te weinig zijn voor modellen met meer inputs (e6: a en b
schommelden; LR GHI −4 % bias; RF/GB overfitten). Run 2 traint op **dag 1–24 (15 mei – 7 juni)** en test op
**dag 25–34 (8–17 juni)**. Kiezen gebeurt op 3 validatievouwen binnen de training (1–14 → 15–17; 1–17 → 18–20;
1–20 → 21–24), met dezelfde keuzeregel als run 1 (BESLISSINGEN #16–#17). De kandidaten van fase 3 zijn vóór het
draaien vastgelegd (#18), met **e5** en **e6** uit V4 erbij.

> **Eerlijkheidsnoot (#19):** dag 25–34 zaten in de testdagen van run 1 en zijn dus al bekeken. De testuitslag van
> run 2 is **niet blind**. Daarom zijn kandidaten en keuzeregel vooraf vastgelegd en verandert de test de keuze niet.

**Code:** de split is instelbaar (`solarcore/config.py`: `RUNS`, `RUN_NAME`, `TRAIN_FRACTION`); de vouwen worden
afgeleid uit het aantal trainingsdagen (`split.fold_days`). **Regressietest:** met `split30_check` komen de schone
data en alle fase-1-tabellen van run 1 exact terug (grootste verschil 7·10⁻¹⁵).

## Per fase (run 2)
| Fase | Keuze (op de vouwen) | Testuitslag dag 25–34 (bevroren) |
|---|---|---|
| 0 – data | Dips met de verhouding uit dag 1–24: **dezelfde 29 kwartieren** als run 1 | – |
| 1 – sensor (bovengrens) | LR straling (5,0 %; polynoom 4,6 % laagste, 5 modellen binnen 0,5 pp) | 5,2 % nRMSE (gem. dag 16,6 %) |
| 2 – zonder weer | LR op eerdere K bij **alle** horizons, ook day-ahead (15,8 % tegen smart persistence 16,5 %) | niet bekeken (niet nodig voor de eindkeuze) |
| 3 – ERA5 | Tijdafspraak opnieuw **+15 min**. **LR POA** (15,56 %): 6 modellen binnen 0,5 pp (15,5–15,7 %), e5 laagste (15,53 %) | zie tabel |
| 4 – eindmodel | LR POA, band uit walk-forward dag 11–24 | band-dekking **82,3 %** (doel 80 %) |

### Fase 3 – test dag 25–34 (bevroren op dag 1–24; 90 %-interval = bootstrap over de 10 testdagen)
| Model | nRMSE | 90 %-interval | Dagfout | Plant 1 | Plant 2 | Walk-forward |
|---|---|---|---|---|---|---|
| Baseline: gemiddelde dag | 16,6 % | 14,7–18,3 | 21,7 % | 14,1 | 18,7 | 16,1 % |
| Fase 2: smart persistence (geen weer) | 14,2 % | 12,3–16,2 | 15,9 % | 14,5 | 13,9 | 14,2 % |
| LR GHI | 14,7 % | 13,4–15,9 | 14,7 % | 15,7 | 13,7 | 14,4 % |
| **LR POA – eindmodel** | **14,3 %** | 13,1–15,3 | 13,3 % | 15,2 | 13,4 | 14,1 % |
| LR EFF_T | 13,5 % | 12,3–14,5 | 8,5 % | 14,1 | 12,8 | 13,4 % |
| e5 | 12,2 % | 11,2–13,2 | 8,5 % | 12,2 | 12,2 | 12,0 % |
| e6 | 12,7 % | 11,7–13,6 | 6,4 % | 13,0 | 12,3 | 12,6 % |
| Random Forest (e6 + extra) | 12,6 % | 11,5–13,7 | 7,7 % | 13,2 | 11,9 | 12,5 % |
| Fase 1: sensor (bovengrens) | 5,2 % | 4,6–5,8 | 3,6 % | 3,8 | 6,3 | 5,2 % |

Bootstrap-verschillen (90 %-interval): LR POA − gemiddelde dag **−2,3 pp (−3,7…−0,9, aantoonbaar)**;
LR POA − LR GHI −0,4 pp (−0,6…−0,2, aantoonbaar maar klein); LR POA − smart persistence +0,1 pp (−1,9…+2,0,
**niet aantoonbaar**). Weerfout 20 %: LR POA 16,9 %, e6 14,7 %. Residuen: LR POA voorspelt begin juni vaak te laag,
vooral als ERA5 "100 % bewolkt" geeft (begin moesson).

### Eindmodel run 2
FRAC = 0,0576 + 0,000941 × POA (W/m²), 's nachts 0; getraind op dag 1–34. Naar een huis: kW = FRAC × kWp × 0,664
(factor, AANNAME) × verliesfactor. POA gebruikt de **dakhoek en richting** van een huis (GHI in run 1 niet).
Band per bewolkingsklasse (× voorspelling): bewolkt 0,23–3,03, half bewolkt 0,50–1,61, helder 0,56–1,32. De dagband
(som P10 … som P90) bevat de echte dag nog steeds altijd (100 %): te ruim → verbeterpunt.
Opgeslagen: `modellen/model_final_split70.joblib`, `resultaten/split70/model_info.json`. `model_final.joblib`
(run 1) is niet overschreven. Heatmaps: `resultaten/split70/heatmaps/`.

### Diagnose run 2 (`resultaten/split70/diagnose/`)
ERA5 kent de helderheid van de dag nog beter dan in run 1 (r = 0,93, tegen 0,57 voor "gisteren"). Per dag is de fout
van LR POA 6,0 % (smart persistence 7,7 %, e6 3,4 %), maar per kwartier zit 83 % van de fout binnen de dag
(wolken op het verkeerde kwartier) en heeft LR POA een bias van −5 %. Dezelfde verklaring als in run 1.

## Leercurve (`leercurve.py`, `resultaten/leercurve/`)
Bevroren, altijd getest op **dag 28–34**; alle fouten op dezelfde schaal (capaciteit uit 27 dagen).

| Model | 10 dagen | 17 | 24 | 27 |
|---|---|---|---|---|
| gemiddelde dag | 19,2 % | 19,2 % | 17,8 % | 17,8 % |
| smart persistence | 14,1 % | 14,1 % | 14,1 % | 14,1 % |
| LR GHI | 16,3 % | 16,1 % | 15,8 % | 15,7 % |
| LR POA | 15,8 % | 15,5 % | 15,2 % | 15,1 % |
| e5 | 14,2 % | 13,2 % | 12,4 % | 12,3 % |
| e6 | 13,3 % | 13,5 % | 13,1 % | 13,2 % |

- **80 % (27 dagen) is niet aantoonbaar beter dan 70 % (24 dagen):** verschil ≤ 0,12 pp, de 90 %-intervallen
  (~±1–2,5 pp) overlappen volledig. → **Advies: 70/30 blijft de hoofdsplit** (#23).
- **10 → 24 dagen helpt wel**, vooral e5 (−1,75 pp, 4 inputs), de gemiddelde dag (−1,4) en LR (−0,6). e6 is met 10
  dagen al bijna op zijn best (−0,2); a en b van e6 bij 24/27 dagen: 0,00171/−0,00055 en 0,00170/−0,00054.
- De ERA5-tijdafspraak en de dips komen uit run 2 (dag 1–24); run 1 (dag 1–10) gaf precies dezelfde, dus dat maakt
  voor de kleine trainingsgroottes geen verschil.

## Vergelijking run 1 ↔ run 2 ↔ leercurve
| | Run 1 (10 trainingsdagen) | Run 2 (24) | Leercurve (27) |
|---|---|---|---|
| testdagen | 11–34 (24 dagen) | 25–34 (10 dagen) | 28–34 (7 dagen) |
| gekozen model | LR GHI (ERA5) | LR POA (ERA5) | – (geen keuze; LR POA ter vergelijking) |
| nRMSE / dagfout gekozen model | 15,0 % / 12,8 % | 14,3 % / 13,3 % | 15,1 % / 15,0 % (LR POA) |
| e6 | 14,1 % / 9,6 % | 12,7 % / 6,4 % | 13,2 % / 6,5 % |
| e5 | – (geen kandidaat) | 12,2 % / 8,5 % | 12,3 % / 8,2 % |
| LR GHI | 15,0 % / 12,8 % | 14,7 % / 14,7 % | 15,7 % / 17,2 % |
| smart persistence (geen weer) | 15,0 % / 13,9 % | 14,2 % / 15,9 % | 14,1 % / 16,4 % |
| gemiddelde dag | 16,8 % / 16,4 % | 16,6 % / 21,7 % | 17,8 % / 26,8 % |
| spreiding a en b (e6) | a 0,00167–0,00219; b −0,00055…−0,00118 (vouwen) | a 0,00156–0,00173; b −0,00036…−0,00055 (vouwen) | a 0,00170; b −0,00054 (stabiel vanaf 24 dagen) |

*Run 1, run 2 en de leercurve hebben **andere testdagen**: de getallen tussen de kolommen zijn niet direct
vergelijkbaar (de testperiode wordt steeds lastiger: begin moesson). Alleen de leercurve vergelijkt
trainingsgroottes eerlijk, op dezelfde dagen.*

**Conclusie run 2 (feiten):**
- Het eindmodel LR POA is **aantoonbaar beter dan de gemiddelde dag**, maar niet aantoonbaar beter dan smart
  persistence zonder weer.
- Met meer trainingsdata zijn **e5 en e6 de beste ERA5-modellen** (op de testdagen ~1,5–2 pp beter en een veel
  kleinere dagfout). Op de validatievouwen lagen alle ERA5-modellen binnen 0,5 pp, dus koos de regel de
  eenvoudigste. De keuze is volgens het protocol niet veranderd.
- **Open (Dimme, AGENTS.md §13):** welk model wordt het officiële eindmodel?

**Beperkingen run 2:** maar 10 testdagen (20 plant-dagen); de test is niet blind (#19); ERA5 is weer achteraf
(bovengrens voor een echte voorspelling); alleen mei–juni (begin moesson); 2 plants.

## Wikitekst (Development / Backend, eenvoudig Engels – concept, pas op de wiki na akkoord van Dimme)
In our first test we used only the first ten days to train the model, and we saw that this was not enough data for
models with more inputs. So we ran the whole pipeline again with the first part of the data for training and the
last part for testing, always split by time and never at random. We chose the model only on days inside the
training period, using a rule that we wrote down before we looked at the test days. The chosen model, a simple
linear regression on the sunlight that falls on the panel, was better than the baseline. We also used a learning
curve to check whether even more training days would help, and the gain was too small to measure. Models with a few
more physics-based inputs did better on the test days, so we want to compare them again when we have more data.

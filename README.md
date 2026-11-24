# Model Final – SolarHome Forecast (final model)

One honestly tested regression model that forecasts solar power: first on the Kaggle data of Plant 1 and 2, then
with a weather model (ERA5) as input, so that it works for a house in India. Working instructions: `AGENTS.md`.
All choices with date and motivation: `BESLISSINGEN.md` (decision log, in Dutch).

The pipeline was run twice (see **"Run 2 – 70/30"** below):

| Run | Split | Results | Final model | Status |
|---|---|---|---|---|
| Run 1 `split30` | choose on days 1–10, test on days 11–34 | `resultaten/split30/` | `modellen/model_final_split30.joblib` (= `model_final.joblib`): LR GHI | ✅ archive |
| **Run 2 `split70`** | **train on days 1–24, test on days 25–34** | `resultaten/split70/` | `modellen/model_final_split70.joblib`: LR POA | ✅ main run |
| Learning curve | 10 / 17 / 24 / 27 training days, test days 28–34 | `resultaten/leercurve/` | – (measurement only) | ✅ |
| **Final model v2** | no training period needed; tested on UK PV (Jul 2024 – Mar 2025) | `resultaten/taak5/ukpv/eindmodel_v2/` | `modellen/model_final_v2.joblib`: pvlib × factor (4 weeks) + blending + band | ✅ **official final model for the app** (decision #60) |

| Phase | Script |
|---|---|
| 0 – read, clean and check the data | `fase0_data.py` |
| 1 – regression benchmark on the Kaggle data | `fase1_benchmark_kaggle.py` |
| 2 – improve the forecast (Kaggle data only) | `fase2_forecast.py` |
| 3 – weather model (ERA5) as input | `fase3_weermodel.py` |
| 4 – choose and save the final model | `fase4_eindmodel.py` |
| extra – learning curve, heatmaps, diagnosis | `leercurve.py`, `heatmaps.py`, `diagnose_era5.py` |

| Task (external validation and improvement) | Scripts (in this order) | Results |
|---|---|---|
| 5a – HKUST Hong Kong 2021–2023 | `taak5a_hkust_stations.py` → `_data.py` → `_tests.py` → `_kalibratie.py` (→ `_satelliet.py`) | `resultaten/taak5/hkust/` |
| 5b – UK PV 2024–2025: day-ahead and horizon | `taak5b_ukpv_systemen.py` → `_weer.py` → `_tests.py` → `_horizon_mix.py`, `_multimodel.py`, `_band.py` | `resultaten/taak5/ukpv/` |
| 6 – Faiman fit | `taak6_faiman.py` | `resultaten/taak6/` |
| Final model v2 | `eindmodel_v2.py` (after the 5b scripts) | `resultaten/taak5/ukpv/eindmodel_v2/`, `modellen/model_final_v2.joblib` |

| `solarcore/` | What |
|---|---|
| `config.py` | all settings (paths, splits, thresholds, colours) |
| `data_prep.py`, `features.py`, `split.py`, `dataset.py` | Kaggle: reading, cleaning, features, split in time |
| `physics.py`, `weather.py` | pvlib physics, ERA5 (Open-Meteo) |
| `models.py`, `evaluate.py`, `forecast_features.py` | model ladders, metrics / selection rule / bootstrap / band, features for forecasting ahead |
| `hkust.py`, `ukpv.py`, `forecast_api.py` | external datasets and archived weather forecasts / satellite |
| `model_v2.py` | final model v2: calibration, blending, band (also for the app) |
| `plots.py` | fixed chart style |

**Running:** with the Anaconda Python (Python 3.13, see `requirements.txt`), F5 in Spyder or in VS Code.
Data that is not in the repository (Kaggle, HKUST, UK PV) and the accounts you need: see `data/README.md`.
By default the main run `split70` is used. For another run, first set in PowerShell `$env:SOLAR_RUN = "split30_check"`
(the regression test; `split30` itself is an archive and cannot be run again). Which test days may already be
viewed is set per run in `solarcore/config.py` (`TEST_VRIJGEGEVEN`).

## Running the app

The Streamlit app in `app/` is the user interface for **final model v2** (decision #60). It shows the solar power
forecast for a house in India (today + 15 days), with a likely range (P10–P90), tips and panel health alerts.

**Start:** double-click `start_app.bat`, or in a terminal in this folder:
```
pip install -r requirements.txt          # once (streamlit, plotly, pvlib, ...)
python -m streamlit run app/app.py
```
The app opens at http://localhost:8501. It needs internet for the live weather forecast (Open-Meteo, CC BY 4.0).
It loads `modellen/model_final_v2.joblib`; if that file is missing, it shows the plain physics forecast with a warning.

**How the forecast is made:**
- pvlib physics with the house's size (kWp), roof tilt and direction, on the Open-Meteo weather forecast
  (radiation shifted −30 min, as in the model);
- × level factor *f* from **the user's own daily production** (entered on *My Usage*) over the 4 weeks before
  this week, updated every week. Without data, *f* = 1 and the app says the forecast is not yet tuned;
- blended with an average day (from the same 4 weeks), more for days further ahead;
- from day 5 on (today = day 1) the day is marked as a **rough estimate** (skill ≈ 0 from there in the UK test);
- band P10–P90: variant B from the UK test (79 % coverage there).

**Privacy (NFR5, GDPR):** the user's profile and entered production are stored only on the computer, in
`app/gebruiker/` (ignored by git). The repository only contains the empty example `app/gebruiker/mijn_panelen.example.json`.

**Tests:** `python app/test_app.py` runs the app without a browser (Streamlit AppTest) for the four personas
(Anita, Rahul, Lakshmi, Vikram), a new house without data and a house with 6 weeks of data, on every page, plus the
forecast page in all 6 languages (47 checks, all pass). It uses a temporary user folder, not your own data.

**Same result as the model:** with the same calibration, the app computes exactly the same forecast as
`eindmodel_v2.py` (difference 0.0 kW on the UK test data). The app calibrates from daily totals (users only enter kWh
per day) instead of quarter-hours; on the UK test this changes nRMSE from 14.32 to 14.33 % and the daily error from
25.2 to 25.3 %.

**Limits:** the band is fixed from the UK test (not yet per house); "this year" on *My Usage* is physics without the
factor; the app works only for locations in India; there is no FastAPI service (not found in the project).

> **Note:** the sections "Phase 0" to "Diagnosis" below describe **run 1** (30/70). Run 2 follows after them.

---

## Phase 0 – Data (CRISP-DM: data understanding + preparation)

**What:** read, check and clean the 4 Kaggle files into one table per quarter-hour per plant:
`data/clean/plant_15min.csv` (6,528 rows = 34 days × 96 quarter-hours × 2 plants), with
`DATE_TIME, PLANT, DAY, AC_KW, N_INV, N_UITVAL, UITVAL, IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE`.
Summary with all counts: `resultaten/fase0/samenvatting.json`.

### Split
34 days → **setup days 1–10 (15–24 May 2020)**, **test days 11–34 (25 May – 17 June 2020)**.

### Data check (facts)
| | Plant 1 | Plant 2 |
|---|---|---|
| Generation rows / inverters | 68,778 / 22 | 67,698 / 22 |
| Quarter-hours without generation data (of 3,264) | **106** | 5 |
| DC/AC before the fix → after the fix | 10.23 → **1.02** | 1.02 → 1.02 |
| Failed inverter rows (removed) | 63 | **3,365** |
| Quarter-hours with at least one failure | 28 | **594** |
| AC_KW = NaN after cleaning | 108 | 5 |
| Plant-wide dips (flag `DIP`, added after phase 1) | 0 | **29** |
| Sensor gaps before → after interpolation (≤ 1 hour) | 82 → 57 | 5 → 0 |
| Max. IRRADIATION | 1.22 kW/m² | 1.10 kW/m² |
| Correlation irradiance ↔ AC (after cleaning) | 0.996 | 0.966 |

- **No duplicate rows** and no NA in the raw files; the gaps are missing timestamps.
- **DC error of Plant 1 confirmed** (DC/AC 10.2 → after / 10: 1.02, the same as Plant 2).
- **IRRADIATION is in kW/m²** (max 1.1–1.2). ERA5 is in W/m² → converted in phase 3.
- **Inverter failures at Plant 2: 12–13 % of the inverter quarter-hours with real sun** (sensor > 0.05–0.1 kW/m²;
  11–12 % with our median rule). The 17 % mentioned earlier (wiki, `Model/`, AGENTS.md) applies to the threshold
  "sensor > 0 kW/m²": then twilight quarter-hours also count, when inverters have not started yet.
  Plant 1 has almost no failures (0.2 %).
- Plant 1 misses 106 quarter-hours (~1 day) of generation data and 57 quarter-hours of sensor data (gaps > 1 hour).

### What the charts show (`resultaten/fase0/`)
| Chart | What we see |
|---|---|
| `01_uurprofiel.png` | Both plants have the same daily shape, peak around 12:00–12:30 IST (fits ~78°E). Plant 2 has a wider spread around noon (clouds and failures). |
| `02_boxplot_per_omvormer.png` | Plant 1: all inverters are similar. Plant 2: a few inverters produce structurally less (median ~4,000 vs ~6,500 kWh/day), and there are many low outliers → failures. |
| `03_straling_vs_ac.png` | Almost a straight line (r = 0.996 / 0.966). At Plant 2 there are still points far below the line at high irradiance: **partial failures** that our rule does not catch (not 0 kW, but too low). Point of attention for phase 1 (residuals). |
| `04_correlatiematrix.png` | Power depends mostly on irradiance (0.96). Module temperature (0.89) and "module − air" (0.92) are themselves strongly correlated with irradiance (0.92–0.95) → **collinearity**: temperature probably adds little as a separate input. Air temperature 0.49. |
| `05_dagopbrengst_split.png` | Daily yield varies strongly per day (cloudy days 18 and 28). The setup period (days 1–10) contains both sunny and cloudy days. The test period contains the lowest days (Plant 2, days 18 and 28) → the test is not "easier" than the setup. |
| `06_dc_ac_per_dag.png` | After the fix DC/AC is 1.020–1.024 for both plants: stable, so the fix is correct. |
| `07_module_min_lucht.png` | Around noon the panel is ~20 °C warmer than the air; at night ~1.5 °C colder (radiative cooling). Plant 1 is slightly warmer than Plant 2. |

### Assumptions phase 0 (separate from the facts)
- The failure rule (≤ 0 kW while the median > 5 % of the maximum) catches complete failures, not partial ones.
- The time in the Kaggle data is local time (IST) and the start of the quarter-hour (as tested in V3–V5).
- The unit of AC_POWER is kW per inverter (consistent with DAILY_YIELD in kWh). This does not matter for the model,
  because we work with FRAC = AC_KW / capacity.

---

## Phase 1 – Regression benchmark on the Kaggle data (30/70)

**Question:** how well does regression explain the power from the plant's own data?
**Note:** this is **not a real forecast**. The sensor measures at the same moment as the power. This is the
**upper limit**: a house has no irradiance sensor.

**Setup:** target FRAC = AC_KW / capacity (99th percentile during the day, from the training days only, per plant).
Daytime only (cos zenith > 0.02), only quarter-hours where all inputs are known, and **without the 29 plant-wide
dips of Plant 2** (decision #10): 3,403 quarter-hours. Every model is judged on exactly the same quarter-hours.
No plant ID as input. `random_state = 42`.

> **Honesty note:** the plant-wide dips were found in the residuals, so **after** looking at the test data. The
> whole phase was then run again (including the choice). The choice stayed the same. The version with dips is shown
> below as well (`MET_DIPS = True` in the script).

### Choice (on days 1–10 only, 2 validation folds; without dips)
| Model | nRMSE | fold 1 (val. 7–8) | fold 2 (val. 9–10) | R² | Daily error |
|---|---|---|---|---|---|
| Baseline: mean day | 13.2 % | 14.0 | 12.4 | 0.83 | 13.4 % |
| Baseline: persistence | 19.9 % | 26.3 | 13.5 | 0.57 | 13.6 % |
| **LR irradiance (chosen)** | 6.5 % | 5.9 | 7.0 | 0.96 | 1.9 % |
| LR irradiance + module temp. | 6.3 % | 5.5 | 7.0 | 0.96 | 1.6 % |
| LR physical: irradiance × (1 + γ(T − 25)) | 6.7 % | 6.5 | 7.0 | 0.95 | 2.5 % |
| Ridge (all sensor inputs + time) | **6.2 %** | 5.6 | 6.8 | 0.96 | 2.2 % |
| Polynomial deg. 2 + Ridge | 6.4 % | 5.9 | 6.8 | 0.96 | 2.1 % |
| Random Forest | 6.6 % | 5.7 | 7.5 | 0.96 | 2.2 % |
| Gradient Boosting | 7.1 % | 6.0 | 8.2 | 0.95 | 1.9 % |

Rule: lowest nRMSE = Ridge (6.2 %), not a complex model. Within 0.5 pp: LR irradiance, LR + module temp., polynomial
→ the **simplest is LR irradiance** (1 input). With dips the choice was also LR irradiance (7.5 % vs Ridge 7.1 %).

### Test on days 11–34 (after the choice; does not change the choice)
| Model | A. Frozen nRMSE | Plant 1 | Plant 2 | R² | Daily error | Skill vs mean day | B. Walk-forward | With dips (A) |
|---|---|---|---|---|---|---|---|---|
| Baseline: mean day | 16.8 % | 15.0 | 18.4 | 0.68 | 16.4 % | 0 | 15.9 % | 17.5 % |
| Baseline: persistence | 19.7 % | 19.4 | 20.0 | 0.56 | 13.9 % | −0.18 | 19.7 % | 20.3 % |
| **LR irradiance (chosen)** | **5.5 %** | 4.1 | 6.6 | 0.97 | 2.9 % | 0.68 | 5.4 % | 7.9 % |
| LR irradiance + module temp. | 5.6 % | 4.3 | 6.6 | 0.96 | 3.7 % | 0.67 | 5.4 % | 8.1 % |
| LR physical | 5.1 % | 3.4 | 6.4 | 0.97 | 2.4 % | 0.69 | 5.1 % | 7.7 % |
| Ridge (all) | 5.2 % | 3.7 | 6.4 | 0.97 | 2.8 % | 0.69 | 5.1 % | 7.9 % |
| Polynomial deg. 2 + Ridge | 5.2 % | 3.9 | 6.3 | 0.97 | 3.1 % | 0.69 | 5.0 % | 8.1 % |
| Random Forest | 5.7 % | 4.3 | 6.8 | 0.96 | 3.4 % | 0.66 | 5.2 % | 8.5 % |
| Gradient Boosting | 5.9 % | 3.9 | 7.4 | 0.96 | 2.4 % | 0.65 | 5.4 % | 9.0 % |

**What this means**
- With only the sensor irradiance the error is **3× smaller** than with the mean day (5.5 vs 16.8 % nRMSE).
  That is the **upper limit** for the final model.
- **Removing the dips also helps Plant 1** (4.5 → 4.1 %), although Plant 1 has no dips: the dips in the setup days
  of Plant 2 (days 5–7) pulled the joint model off.
- **All regression models are close together** (5.1–5.9 %); RF and GB are not better. Confirms V1–V5.
- On the test days LR physical, Ridge and polynomial are slightly better (5.1–5.2 %) than the chosen LR irradiance
  (5.5 %), not clearly so on the validation. Difference < 0.5 pp → the choice stays.
- **Walk-forward hardly helps** (5.5 → 5.4 %): with irradiance as input, 10 days are already enough.
- **Cross-plant** (frozen, LR irradiance): P1 → P2 6.7 %, P2 → P1 4.7 %. A model from the other plant is only
  slightly worse than a model from both plants (6.6 % and 4.1 %).
- **Stability** (48 plant-days, LR irradiance): nRMSE per day on average 5.0 %, std 2.2, **CV 0.45**
  (with dips: 6.7 %, std 4.1, CV 0.61). The baselines have a lower CV (~0.3) but a much higher error.

### Residuals (`residuen.png`) – when does the model miss?
1. **Plant-wide dips at Plant 2** (power < 30 % of expected while the sensor measures sun): 29 quarter-hours,
   ~1.7 % of Plant 2. Now removed. Possible cause (ASSUMPTION): grid fault or curtailment.
   **Smaller dips** remain (residual +20 to +60 %): the plant then delivers 30–80 % of expected.
2. **Systematic curvature:** at low power (morning/evening) LR forecasts ~3 % too much (the intercept), in the
   middle range ~5 % too little. The relation irradiance → power is not completely straight (temperature, sun angle).
   Ridge / polynomial / LR physical capture this slightly better (see test).
3. **Autocorrelation at Plant 1:** the residuals are correlated over ~2.5 hours (ACF > 0 up to lag ~10). Errors come
   in blocks (part of the day too high or too low), not as random noise. At Plant 2 shorter (~1 hour).
4. No clear pattern per cloudiness (KT).

### Assumptions phase 1
- γ = −0.004 /°C in "LR physical" is an assumption (not fitted).
- The dip rule (sensor > 0.2 kW/m², power < 30 % of expected; ratio from days 1–10) is our own choice.
- The estimated plant locations are only used for the sun position (COSZ) and the clear-sky index (analysis).
- Charts: `heatmap_model_plant.png`, `heatmap_model_dag.png`, `voorbeelddagen.png`, `residuen.png`.

---

## Phase 2 – Improving the forecast (Kaggle data only, truly ahead)

**Question:** what can we forecast with only the information that is available at the forecast moment t0?
Horizons: **+15 min, +1 hour, +3 hours** and **day-ahead** (tomorrow per quarter-hour, made at 21:00 the evening before).
Inputs only ≤ t0: earlier power, earlier sensor values, clock, sun position, clear-sky profile.

**Approach:** we forecast the **clear-sky index K** = FRAC / FRAC under clear sky (Ineichen × scale per plant, scale
from the training) and then multiply by the clear-sky power. This way the daily curve is already included and the
model only learns "how cloudy". Lags only look back; missing lags (night) → mean K of the previous day.
Dips/failures do not count (FRAC = NaN).

### Choice per horizon (validation days 1–10, nRMSE averaged over 2 folds)
| Model | +15 min | +1 hour | +3 hours | day-ahead |
|---|---|---|---|---|
| Baseline: mean day | 13.2 % | 13.2 % | 13.2 % | 13.2 % |
| Baseline: persistence | 10.6 % | 20.8 % | 43.8 % | 20.9 % |
| Smart persistence (K stays the same) | 8.9 % | 13.0 % | 14.2 % | **11.4 %** ✔ |
| LR on earlier K | **8.6 %** ✔ | 12.1 % | 14.1 % | 14.9 % |
| Ridge (K + sensor + time) | 8.4 % | **11.1 %** ✔ | **12.6 %** ✔ | 15.7 % |
| Random Forest | 8.5 % | 11.3 % | 13.2 % | 14.8 % |
| Gradient Boosting | 9.2 % | 11.6 % | 15.6 % | 16.9 % |

✔ = chosen (rule #2). +15 min: Ridge is the lowest, but LR is within 0.5 pp and simpler.

**First impression (validation only, 4 days per plant – uncertain):**
- Short ahead (+15 min) the history clearly helps (8.6 vs 13.2 % for the mean day).
- The further ahead, the closer to the mean day: at +3 hours the difference is only ~0.6 pp.
- **Day-ahead without weather:** the regression models are worse than the mean day (overfitting on 4–7 training
  days). Only smart persistence ("tomorrow as clear as today") is slightly better.
- After that (Dimme's approval) the test days 11–34, see below.

### Test on days 11–34 (after the choice; nRMSE, A = frozen on days 1–10, B = walk-forward)
| Model | +15 min | +1 hour | +3 hours | day-ahead |
|---|---|---|---|---|
| Baseline: mean day | 16.8 % | 16.8 % | 16.8 % | 16.8 % (B: 15.9 %) |
| Baseline: persistence | 12.8 % | 21.2 % | 38.4 % | 19.9 % |
| Smart persistence | 11.6 % | 15.6 % | 17.8 % | **15.0 %** ✔ |
| LR on earlier K | **10.8 %** ✔ | 13.3 % | 14.5 % | 14.7 % |
| Ridge (K + sensor + time) | 10.8 % | **13.6 %** ✔ | **14.9 %** ✔ | 15.0 % |
| Random Forest | 10.7 % | 13.8 % | 15.0 % | 16.1 % |
| Gradient Boosting | 11.3 % | 15.1 % | 16.2 % | 17.6 % |
| *Upper limit phase 1 (sensor at the same moment)* | *5.5 %* | | | |

✔ = chosen on days 1–10. Walk-forward changes almost nothing (max. ± 0.5 pp), except the mean day (16.8 → 15.9 %).

| Chosen model per plant (nRMSE) | +15 min | +1 hour | +3 hours | day-ahead | mean day |
|---|---|---|---|---|---|
| Plant 1 | 11.0 % | 13.5 % | 14.4 % | 15.0 % | 15.0 % |
| Plant 2 | 10.5 % | 13.7 % | 15.4 % | 14.9 % | 18.4 % |

Stability of the chosen model (nRMSE per plant-day, 48): +15 min 10.3 % (std 3.1; CV 0.30), +1 hour 13.1 % (CV 0.29),
+3 hours 14.3 % (CV 0.29), day-ahead 14.3 % (std 4.6; CV 0.32).

**What this means**
- **Hypothesis confirmed.** Short ahead the history works well (+15 min: 10.8 vs 16.8 %). The further ahead, the
  smaller the advantage: +3 hours 14.9 %.
- **Day-ahead without weather is only a little better than the mean day:** 15.0 vs 16.8 % (skill 0.11). At
  **Plant 1 not at all** (15.0 vs 15.0 %); at Plant 2 it is (14.9 vs 18.4 %). Daily error day-ahead 13.9 % (mean day 16.4 %).
  `voorbeeld_dayahead.png` shows why: without weather you always forecast a smooth "average" curve; you cannot see
  tomorrow's clouds coming.
- Even +15 min (10.8 %) is more than 2× worse than the upper limit with the sensor at the same moment (5.5 %).
- On the test, LR and Ridge are about equal at +1 and +3 hours (difference 0.2–0.4 pp, within the noise).
- The test period is harder than the validation (e.g. mean day 16.8 vs 13.2 %; more cloudy days, start of the monsoon).
- **Conclusion for phase 3:** for "tomorrow" (user story 1) a **weather forecast is needed**.

### Assumptions phase 2
- Clear-sky scale = 90th percentile of FRAC / CS_GHI (high sun) in the training; K limited to 0–1.5.
- Day-ahead made at 21:00: the whole day D−1 is then known (daily means of K and sensor).
- Missing lags (night, gaps) → mean K of the previous day (or of the training).
- Charts: `horizon_validatie.png`, `horizon_test.png`, `voorbeeld_dayahead.png`.

---

## Phase 3 – Weather model (ERA5) as input

**Question:** how good is the model if we replace the plant sensor by an existing weather model?
ERA5 = weather **in hindsight** (reanalysis), so an **upper limit** for use with a real weather forecast.
Physics (pvlib) taken from V5: hour → quarter-hour, Perez POA, Faiman cell temperature, Kimber soiling, EFF_T,
KT_POA. Plant setup: fixed, tilt = latitude, facing south (ASSUMPTION from V3–V5). Never the plant sensor as input.

### Time alignment (days 1–10 only)
The correlation ERA5 GHI ↔ sensor per quarter-hour is highest at **+15 min** (P1 r = 0.863; P2 0.939), so the
irradiance of hour h belongs to the interval with centre h + 15 min. At −30 min (the documentation: "mean of the
preceding hour") r is much lower (0.79 / 0.91). See `tijdafspraak.png` and decision #12.

### Choice (validation within days 1–10, nRMSE averaged over 2 folds)
| Model | nRMSE | fold 1 | fold 2 | R² | Daily error |
|---|---|---|---|---|---|
| Baseline: mean day | 13.2 % | 14.0 | 12.4 | 0.83 | 13.4 % |
| Baseline: persistence | 20.9 % | 28.3 | 13.5 | 0.52 | 14.4 % |
| *Phase 2: smart persistence (no weather)* | *11.4 %* | 12.0 | 10.8 | 0.87 | 10.4 % |
| *Phase 1: LR irradiance (sensor, upper limit)* | *6.5 %* | 5.9 | 7.0 | 0.96 | 1.9 % |
| **LR GHI (ERA5) – chosen** | **11.2 %** | 11.0 | 11.4 | 0.87 | 8.1 % |
| LR POA | 11.4 % | 11.2 | 11.6 | 0.87 | 9.2 % |
| LR EFF_T | 12.6 % | 12.5 | 12.8 | 0.84 | 11.6 % |
| e6 (V4): EFF_T × (a + b·KT_POA), no intercept | 14.5 % | 15.8 | 13.2 | 0.79 | 15.8 % |
| Ridge e6 (no intercept) | 14.5 % | 15.7 | 13.2 | 0.79 | 15.8 % |
| Ridge e6 + cloud cover, rain, humidity | 13.2 % | 15.4 | 11.0 | 0.82 | 10.9 % |
| Random Forest (e6 + extra) | 16.0 % | 20.3 | 11.8 | 0.73 | 14.8 % |
| Gradient Boosting (e6 + extra) | 18.9 % | 24.0 | 13.8 | 0.62 | 17.5 % |

**First impression (only 4 validation days per plant – uncertain):**
- With ERA5 the error is much larger than with the sensor (11.2 vs 6.5 %): **the weather data is the bottleneck**,
  not the model.
- On the setup days ERA5 is only slightly better than the mean day (11.2 vs 13.2 %) and about equal to phase 2.
- **e6 (the V4 winner) does not work better here.** Checked for errors (within days 1–10 only): also in-sample on
  days 1–10, LR GHI, LR POA and e6 are equal (13.8 / 13.8 / 14.3 %). With 6–8 training days a and b vary strongly
  (b: −0.0012 ↔ −0.0006 per fold; EFF_T and EFF_T·KT_POA r ≈ 0.97). ERA5 has a considerable deviation per day
  (ratio power / ERA5 POA per day 0.79–1.18) that affects all models equally.
- Complex models (RF, GB) clearly overfit with so few days.
- The test days 11–34 had not been viewed yet (`TOON_TEST = False`).

### Test on days 11–34 (after the choice; does not change the choice)
| Model | A. Frozen nRMSE | Plant 1 | Plant 2 | R² | Daily error | Skill vs mean day | B. Walk-forward |
|---|---|---|---|---|---|---|---|
| Baseline: mean day | 16.8 % | 15.0 | 18.4 | 0.68 | 16.4 % | 0 | 15.9 % |
| Baseline: persistence | 19.9 % | 19.4 | 20.4 | 0.55 | 14.0 % | −0.19 | 19.9 % |
| *Phase 2: smart persistence (no weather)* | *15.0 %* | 15.0 | 14.9 | 0.74 | 13.9 % | 0.11 | 15.0 % |
| *Phase 1: LR irradiance (sensor, upper limit)* | *5.5 %* | 4.1 | 6.6 | 0.97 | 2.9 % | 0.67 | 5.4 % |
| **LR GHI (ERA5) – chosen** | **15.0 %** | 15.4 | 14.7 | 0.74 | 12.8 % | 0.10 | 14.7 % |
| LR POA | 14.9 % | 15.2 | 14.5 | 0.75 | 12.2 % | 0.11 | 14.5 % |
| LR EFF_T | 14.5 % | 14.4 | 14.6 | 0.76 | 10.3 % | 0.13 | 14.4 % |
| e6: EFF_T × (a + b·KT_POA) | **14.1 %** | 14.0 | 14.3 | 0.77 | **9.6 %** | **0.16** | 14.1 % |
| Ridge e6 | 14.2 % | 14.0 | 14.3 | 0.77 | 9.6 % | 0.16 | 14.1 % |
| Ridge e6 + cloud cover, rain, humidity | 16.1 % | 15.7 | 16.6 | 0.70 | 13.1 % | 0.04 | 14.0 % |
| Random Forest | 16.1 % | 16.2 | 16.0 | 0.70 | 13.8 % | 0.04 | 14.3 % |
| Gradient Boosting | 16.5 % | 16.7 | 16.2 | 0.69 | 13.7 % | 0.02 | 14.7 % |

**Stability** (nRMSE per plant-day, 48): LR GHI 14.6 % (std 3.6; CV 0.25), e6 13.8 % (std 3.4; CV 0.24),
smart persistence 14.3 % (std 4.6; CV 0.32), mean day 15.9 % (std 5.5; CV 0.34).
**Cross-plant** (frozen): LR GHI P1 → P2 14.9 %, P2 → P1 15.3 %; e6 15.1 / 15.0 %.

**Sensitivity to weather errors** (noise per day × per hour on the ERA5 irradiance, test days, 5 repetitions):
| Error | LR GHI (chosen) | e6 | Ridge e6 + extra | mean day (no weather) |
|---|---|---|---|---|
| 0 % | 15.0 % | 14.1 % | 16.1 % | 16.8 % |
| 10 % | 16.2 % | 14.7 % | 16.4 % | 16.8 % |
| 20 % | 18.4 % | 16.2 % | 17.0 % | 16.8 % |
| 30 % | 20.3 % | 17.6 % | 18.1 % | 16.8 % |

**What this means**
- **With ERA5, day-ahead is only slightly better than without weather.** The chosen LR GHI (15.0 %) equals phase 2
  (smart persistence, 15.0 %) and is 1.7 pp better than the mean day. The sensor (5.5 %) shows how much gain perfect
  weather would give: **the weather data is the bottleneck**, as in V1–V5.
- **On the test days e6 (V4) is the best ERA5 model:** 14.1 % nRMSE and a daily error of 9.6 % (LR GHI 12.8 %). On
  the validation (with only 6–8 training days) e6 was worse; the selection rule therefore chose LR GHI. The test
  does not change the choice (protocol), but the pattern fits V4/V5: **e6 needs more training days** (10 days in the
  frozen model; 14–29 days in V4/V5).
- **e6 is also more robust to weather errors:** at 20 % error 16.2 % vs 18.4 % for LR GHI. LR GHI is already worse
  than the mean day at ~12 % weather error.
- Extra weather variables (cloud cover, rain, humidity) and RF/GB do not help when frozen (overfitting on 10 days);
  with walk-forward (more days) they reach 14.0–14.7 %.
- **Residuals** (`residuen.png`): on the test days the model forecasts too low on average (many residuals −10 to
  −30 %), also at "100 % cloud cover" according to ERA5. So ERA5 saw clouds more often than there were at the plant
  (grid cell ~25 km, estimated location). The residuals are correlated over ~2–3 hours (ACF): errors come in blocks,
  typical for clouds at the wrong time.

### Assumptions phase 3
- Plant setup: fixed, tilt = latitude, facing south (V3–V5). Physics parameters from V5 (γ −0.37 %/°C, ...).
- ERA5 as a stand-in for a weather forecast; the error sizes in the weather-error test are assumptions.
- Charts: `tijdafspraak.png`, `vergelijking.png`, `voorbeelddagen.png`, `residuen.png`, `weerfout.png`.

---

## Phase 4 – Final model

**Chosen: LR GHI (ERA5)** – a linear regression on only the horizontal irradiance from the weather model.
Chosen with the fixed selection rule on days 1–10 (phase 3). On the test days e6 was slightly better; according to
the protocol the test does not change the choice, and Dimme chose to follow the rule (decision #14).

- **Formula** (trained on all 34 days, both plants, without dips):
  FRAC = 0.0459 + 0.000902 × GHI (W/m²); at night (cos zenith ≤ 0.02) = 0.
- **For a house:** kW = FRAC × kWp × **factor 0.664** × loss factor (shading, age), limited by the inverter.
  The factor is an **ASSUMPTION**: the 99th percentile of the pvlib power (kW/kWp) at the plant locations
  (Plant 1: 0.684; Plant 2: 0.643).
- **Band P10–P90** per cloudiness class (ERA5 KT), × forecast: cloudy 0.66–2.52; partly cloudy 0.63–1.60;
  clear 0.55–1.30. Calibrated out-of-sample. Check: calibration on days 11–22 → **coverage 80.5 %** on days 23–34
  (target 80 %). The **daily band** (sum of P10 to sum of P90) contains the real day in 100 % of the cases: too wide
  (quarter-hour quantiles must not be added up) → point for improvement.
- **Saved:** `modellen/model_final.joblib` (same keys as `model_v5.joblib`: model, features, factor,
  band_method, band, kt_bins, params, rad_shift_min, min_cosz) and `resultaten/model_info.json`
  (choice, motivation, scores, training/test days, assumptions).

### Final table (test days 11–34, frozen model on days 1–10, house 5 kWp)
| Model | Input | nRMSE | nMAE | RMSE | MAE | R² | Daily error | Skill vs mean day |
|---|---|---|---|---|---|---|---|---|
| Baseline: mean day | none | 16.8 % | 11.9 % | 0.84 kW | 0.60 kW | 0.68 | 16.4 % | 0 |
| Baseline: persistence | none | 19.9 % | 13.3 % | 0.99 kW | 0.66 kW | 0.55 | 14.0 % | −0.19 |
| Phase 2: smart persistence (no weather) | own power | 15.0 % | 10.7 % | 0.75 kW | 0.53 kW | 0.74 | 13.9 % | 0.11 |
| **Phase 3: LR GHI (ERA5) – FINAL MODEL** | ERA5 weather | **15.0 %** | 11.1 % | 0.75 kW | 0.55 kW | 0.74 | 12.8 % | 0.10 |
| Phase 3: e6 (alternative) | ERA5 + pvlib | 14.1 % | 10.4 % | 0.71 kW | 0.52 kW | 0.77 | 9.6 % | 0.16 |
| Phase 1: LR irradiance (upper limit) | plant sensor | 5.5 % | 3.5 % | 0.27 kW | 0.18 kW | 0.97 | 2.9 % | 0.67 |

Per plant (final model): Plant 1 15.4 %, Plant 2 14.7 % nRMSE. Walk-forward (every day again on all earlier days):
14.7 %.

### Conclusion
- The final model is **better than the baseline** (mean day): 15.0 vs 16.8 % nRMSE, daily error 12.8 vs 16.4 %.
- The gain is modest: with this ERA5 data the final model is as good as "tomorrow as clear as today" (phase 2).
- **The weather data is the bottleneck**, not the model: with the plant sensor (perfect weather) the error is 3× smaller.
- **Recommendation:** as soon as there is more data (more days, or a house with its own measurements), test **e6**
  again: it uses the irradiance on the tilted panel (roof tilt + orientation) and was better on the test and with
  weather errors.

### Limitations
- 34 days in May–June (start of the monsoon), 2 plants, 10 setup days. No conclusions about a whole year.
- ERA5 is weather **in hindsight**; a real forecast has more error (at ~12 % weather error LR GHI is already worse
  than the mean day, phase 3). Testing with archived forecasts (Open-Meteo Historical Forecast, from ~2022) is only
  possible with recent power data.
- **LR GHI does not look at the tilt and orientation** of a house's roof: these only count through kWp and losses.
  For a flat or south-facing roof that is no problem; for an east/west roof it is (e6 would include it).
- Plant locations and setup are estimated; the factor 0.664 has not been checked with data of a real house.

### Assumptions phase 4
- Factor 0.664 (see above); band calibrated on 24 out-of-sample days (2 × 24 plant-days).
- Charts: `eindvergelijking.png`, `band_voorbeeld.png`.

---

## Heatmaps for the report (`heatmaps.py` → `resultaten/heatmaps/`)
Same setup as phase 3/4 (frozen model, trained on days 1–10, tested on days 11–34). No new choices.

| File | What | What stands out |
|---|---|---|
| `01_correlatie_alle_features.png` | Correlation (Pearson) of all 20 features with each other and with the power, daytime, both plants | Sensor irradiance 0.98 with the power; ERA5 irradiance, POA, EFF_T and pvlib power ~0.88 (0.97–1.00 among themselves: almost the same). ERA5 cloud cover (−0.17), clearness (0.15), wind and rain say little per quarter-hour. Humidity −0.45 (mostly through temperature, −0.94). |
| `02_fout_eindmodel_dag_plant.png` | nRMSE of the final model per test day × Plant 1 / Plant 2 / both | The error varies strongly per day (8–26 %). Together, day 18 is the hardest (Plant 2: 26 %); for Plant 1 day 34 is the worst (22 %). Days 14–15 are the easiest for both plants (8–9 %). |
| `03_fout_model_dag_plant1.png`, `..._plant2.png` | nRMSE per model × test day, per plant | All ERA5 models go up and down together: the **weather of the day** determines the error. The sensor (upper limit) stays low almost everywhere. |
| `04_fout_model_plant.png` | nRMSE per model × plant over the whole test period | Final model 15.4 % (P1) and 14.7 % (P2); mean day 15.0 / 18.4 %; sensor 4.1 / 6.6 %. |

The tables are next to them as `.csv` (`fout_per_dag_per_plant.csv` with the nRMSE per plant-day for all models).
English versions of two charts: `EN_correlation_features.png`, `EN_error_final_model_per_day.png`.

---

## Diagnosis: why does ERA5 score as well as "without weather" (both ~15 % nRMSE)? (`diagnose_era5.py`)
Explanation only, no new choice. Test days 11–34, frozen on days 1–10. No error found.

**1. ERA5 does know much better how sunny the day will be.** Correlation of the real daily clearness with ERA5 of
the same day 0.85 (P1 0.70; P2 0.91), with the measured clearness of yesterday (what smart persistence uses) only 0.54.

**2. But per quarter-hour what counts most is something nobody can forecast: exactly when a cloud passes.**
| Model | nRMSE | bias | error at day level | error within the day | share within the day |
|---|---|---|---|---|---|
| mean day | 16.8 % | +3.4 % | 8.5 % | 14.1 % | 70 % |
| smart persistence (no weather) | 15.0 % | +0.5 % | 7.4 % | 13.0 % | 75 % |
| **LR GHI (ERA5) – final model** | 15.0 % | **−4.1 %** | **4.9 %** | 13.6 % | 82 % |
| e6 (ERA5) | 14.2 % | −2.0 % | 4.8 % | 13.2 % | 87 % |
| sensor (upper limit) | 5.5 % | −0.9 % | 1.3 % | 5.2 % | 92 % |

- **Day level:** ERA5 is clearly better (4.9 vs 7.4 %).
- **Within the day:** all forecasters miss ~13–14 %. ERA5 is even slightly worse here than a smooth curve: if ERA5
  puts a cloud at the wrong hour, you get the error twice (dip forecast where there was sun, and sun where the dip
  was). A smooth curve "guesses" the middle and is punished less for it.
- **Fixed offset:** the frozen LR GHI forecasts 4.1 % too low on average on the test days (trained on 10 days, not
  updated). Smart persistence recalibrates itself every day with yesterday's measured power (bias 0.5 %).
- Net, these effects cancel out per quarter-hour → both ~15 %.

**3. On a coarser time scale ERA5 does win:**
| Model | per quarter-hour | per hour | per day |
|---|---|---|---|
| mean day | 16.8 % | 14.2 % | 9.2 % |
| smart persistence (no weather) | 15.0 % | 12.3 % | 7.5 % |
| **LR GHI (ERA5) – final model** | 15.0 % | 12.3 % | **6.4 %** |
| e6 (ERA5) | 14.2 % | 11.3 % | 5.2 % |
| sensor (upper limit) | 5.5 % | 4.1 % | 1.6 % |

For user story 1 ("how much power tomorrow?") the daily total counts most: there ERA5 is better (also the daily
error: 12.8 vs 13.9 %). For information (does not change the choice): without the fixed offset LR GHI would reach
14.5 %, and the average of LR GHI and smart persistence 13.7 % → the two contain **different** information.

---

# Run 2 – 70/30 (main run, 29 Oct 2026)

**Why again?** Run 1 showed that 10 training days are too few for models with more inputs (e6: a and b varied;
LR GHI −4 % bias; RF/GB overfit). Run 2 trains on **days 1–24 (15 May – 7 June)** and tests on
**days 25–34 (8–17 June)**. The choice is made on 3 validation folds within the training (1–14 → 15–17; 1–17 → 18–20;
1–20 → 21–24), with the same selection rule as run 1 (decisions #16–#17). The candidates of phase 3 were fixed
before running (#18), with **e5** and **e6** from V4 added.

> **Honesty note (#19):** days 25–34 were part of the test days of run 1 and have therefore already been viewed.
> The test result of run 2 is **not blind**. That is why the candidates and the selection rule were fixed in advance
> and the test does not change the choice.

**Code:** the split is configurable (`solarcore/config.py`: `RUNS`, `RUN_NAME`, `TRAIN_FRACTION`); the folds are
derived from the number of training days (`split.fold_days`). **Regression test:** with `split30_check` the clean
data and all phase-1 tables of run 1 come back exactly (largest difference 7·10⁻¹⁵).

## Per phase (run 2)
| Phase | Choice (on the folds) | Test result days 25–34 (frozen) |
|---|---|---|
| 0 – data | Dips with the ratio from days 1–24: **the same 29 quarter-hours** as run 1 | – |
| 1 – sensor (upper limit) | LR irradiance (5.0 %; polynomial 4.6 % lowest, 5 models within 0.5 pp) | 5.2 % nRMSE (mean day 16.6 %) |
| 2 – without weather | LR on earlier K at **all** horizons, also day-ahead (15.8 % vs smart persistence 16.5 %) | not viewed (not needed for the final choice) |
| 3 – ERA5 | Time alignment again **+15 min**. **LR POA** (15.56 %): 6 models within 0.5 pp (15.5–15.7 %), e5 lowest (15.53 %) | see table |
| 4 – final model | LR POA, band from walk-forward days 11–24 | band coverage **82.3 %** (target 80 %) |

### Phase 3 – test days 25–34 (frozen on days 1–24; 90 % interval = bootstrap over the 10 test days)
| Model | nRMSE | 90 % interval | Daily error | Plant 1 | Plant 2 | Walk-forward |
|---|---|---|---|---|---|---|
| Baseline: mean day | 16.6 % | 14.7–18.3 | 21.7 % | 14.1 | 18.7 | 16.1 % |
| Phase 2: smart persistence (no weather) | 14.2 % | 12.3–16.2 | 15.9 % | 14.5 | 13.9 | 14.2 % |
| LR GHI | 14.7 % | 13.4–15.9 | 14.7 % | 15.7 | 13.7 | 14.4 % |
| **LR POA – final model** | **14.3 %** | 13.1–15.3 | 13.3 % | 15.2 | 13.4 | 14.1 % |
| LR EFF_T | 13.5 % | 12.3–14.5 | 8.5 % | 14.1 | 12.8 | 13.4 % |
| e5 | 12.2 % | 11.2–13.2 | 8.5 % | 12.2 | 12.2 | 12.0 % |
| e6 | 12.7 % | 11.7–13.6 | 6.4 % | 13.0 | 12.3 | 12.6 % |
| Random Forest (e6 + extra) | 12.6 % | 11.5–13.7 | 7.7 % | 13.2 | 11.9 | 12.5 % |
| Phase 1: sensor (upper limit) | 5.2 % | 4.6–5.8 | 3.6 % | 3.8 | 6.3 | 5.2 % |

Bootstrap differences (90 % interval): LR POA − mean day **−2.3 pp (−3.7…−0.9, significant)**;
LR POA − LR GHI −0.4 pp (−0.6…−0.2, significant but small); LR POA − smart persistence +0.1 pp (−1.9…+2.0,
**not significant**). Weather error 20 %: LR POA 16.9 %, e6 14.7 %. Residuals: LR POA often forecasts too low in
early June, especially when ERA5 gives "100 % cloud cover" (start of the monsoon).

### Final model run 2
FRAC = 0.0576 + 0.000941 × POA (W/m²), 0 at night; trained on days 1–34. For a house: kW = FRAC × kWp × 0.664
(factor, ASSUMPTION) × loss factor. POA uses the **roof tilt and orientation** of a house (GHI in run 1 does not).
Band per cloudiness class (× forecast): cloudy 0.23–3.03, partly cloudy 0.50–1.61, clear 0.56–1.32. The daily band
(sum P10 … sum P90) still always contains the real day (100 %): too wide → point for improvement.
Saved: `modellen/model_final_split70.joblib`, `resultaten/split70/model_info.json`. `model_final.joblib`
(run 1) was not overwritten. Heatmaps: `resultaten/split70/heatmaps/`.

### Diagnosis run 2 (`resultaten/split70/diagnose/`)
ERA5 knows the clearness of the day even better than in run 1 (r = 0.93, vs 0.57 for "yesterday"). Per day the error
of LR POA is 6.0 % (smart persistence 7.7 %, e6 3.4 %), but per quarter-hour 83 % of the error is within the day
(clouds at the wrong quarter-hour) and LR POA has a bias of −5 %. The same explanation as in run 1.

## Learning curve (`leercurve.py`, `resultaten/leercurve/`)
Frozen, always tested on **days 28–34**; all errors on the same scale (capacity from 27 days).

| Model | 10 days | 17 | 24 | 27 |
|---|---|---|---|---|
| mean day | 19.2 % | 19.2 % | 17.8 % | 17.8 % |
| smart persistence | 14.1 % | 14.1 % | 14.1 % | 14.1 % |
| LR GHI | 16.3 % | 16.1 % | 15.8 % | 15.7 % |
| LR POA | 15.8 % | 15.5 % | 15.2 % | 15.1 % |
| e5 | 14.2 % | 13.2 % | 12.4 % | 12.3 % |
| e6 | 13.3 % | 13.5 % | 13.1 % | 13.2 % |

- **80 % (27 days) is not significantly better than 70 % (24 days):** difference ≤ 0.12 pp, the 90 % intervals
  (~±1–2.5 pp) overlap completely. → **Advice: 70/30 stays the main split** (#23).
- **10 → 24 days does help**, mostly e5 (−1.75 pp, 4 inputs), the mean day (−1.4) and LR (−0.6). e6 is already almost
  at its best with 10 days (−0.2); a and b of e6 at 24/27 days: 0.00171/−0.00055 and 0.00170/−0.00054.
- The ERA5 time alignment and the dips come from run 2 (days 1–24); run 1 (days 1–10) gave exactly the same, so this
  makes no difference for the small training sizes.

## Comparison run 1 ↔ run 2 ↔ learning curve
| | Run 1 (10 training days) | Run 2 (24) | Learning curve (27) |
|---|---|---|---|
| test days | 11–34 (24 days) | 25–34 (10 days) | 28–34 (7 days) |
| chosen model | LR GHI (ERA5) | LR POA (ERA5) | – (no choice; LR POA for comparison) |
| nRMSE / daily error chosen model | 15.0 % / 12.8 % | 14.3 % / 13.3 % | 15.1 % / 15.0 % (LR POA) |
| e6 | 14.1 % / 9.6 % | 12.7 % / 6.4 % | 13.2 % / 6.5 % |
| e5 | – (no candidate) | 12.2 % / 8.5 % | 12.3 % / 8.2 % |
| LR GHI | 15.0 % / 12.8 % | 14.7 % / 14.7 % | 15.7 % / 17.2 % |
| smart persistence (no weather) | 15.0 % / 13.9 % | 14.2 % / 15.9 % | 14.1 % / 16.4 % |
| mean day | 16.8 % / 16.4 % | 16.6 % / 21.7 % | 17.8 % / 26.8 % |
| spread of a and b (e6) | a 0.00167–0.00219; b −0.00055…−0.00118 (folds) | a 0.00156–0.00173; b −0.00036…−0.00055 (folds) | a 0.00170; b −0.00054 (stable from 24 days) |

*Run 1, run 2 and the learning curve have **different test days**: the numbers between the columns are not directly
comparable (the test period gets harder each time: start of the monsoon). Only the learning curve compares training
sizes fairly, on the same days.*

**Conclusion run 2 (facts):**
- The final model LR POA is **significantly better than the mean day**, but not significantly better than smart
  persistence without weather.
- With more training data **e5 and e6 are the best ERA5 models** (~1.5–2 pp better on the test days and a much
  smaller daily error). On the validation folds all ERA5 models were within 0.5 pp, so the rule chose the simplest.
  According to the protocol the choice was not changed.
- **Decided (16 Nov 2026, decision #60):** LR POA (run 2) is the tested core model in the report; final model v2
  (`modellen/model_final_v2.joblib`, see "Improving the model") is the official model for the app.

**Limitations run 2:** only 10 test days (20 plant-days); the test is not blind (#19); ERA5 is weather in hindsight
(upper limit for a real forecast); only May–June (start of the monsoon); 2 plants.

## Wiki text (Development / Backend, simple English – draft, only on the wiki after Dimme's approval)
In our first test we used only the first ten days to train the model, and we saw that this was not enough data for
models with more inputs. So we ran the whole pipeline again with the first part of the data for training and the
last part for testing, always split by time and never at random. We chose the model only on days inside the
training period, using a rule that we wrote down before we looked at the test days. The chosen model, a simple
linear regression on the sunlight that falls on the panel, was better than the baseline. We also used a learning
curve to check whether even more training days would help, and the gain was too small to measure. Models with a few
more physics-based inputs did better on the test days, so we want to compare them again when we have more data.

---

# Task 5a – External validation: HKUST rooftop PV (Hong Kong, 2021–2023)

**Source:** Lin et al. (2025), *Scientific Data*, doi:10.1038/s41597-025-04397-y; data Dryad doi:10.5061/dryad.m37pvmd99
(version 8 Sep 2024, licence CC0). 60 rooftop systems on the HKUST campus, PV every 5 min (inverter) or every hour
(stations without optimizer), weather station every 1 min. Download it yourself to `data/extern/hkust/raw/` (not in git).
ERA5 via Open-Meteo (CC BY 4.0), 22.3363 N 114.2634 E.

## Step 1 – Choosing stations (`taak5a_hkust_stations.py` → `resultaten/taak5/hkust/stap1_stations/`)
**Facts**
- Brick metadata read with rdflib + SPARQL. One typo in the .ttl (Zone_D) is repaired while reading.
- `ratedPowerOutput` = **AC power of the inverter**; the panel power (DC, modules × W) is 5–30 % higher.
  Evidence: `P_max/kWp` ≈ 1.00 at almost all stations (clipping).
- Times are **HKT** (centre of the irradiance on clear days 12:13, calculated solar noon 12:22). Power and weather
  station are aligned (best shift 0 min per quarter).
- 23 stations without optimizer only have **hourly data** → not used (our protocol is 15 min).
- PV files stop every day at 17:55; after 18:00 at most 0.8 % of the monthly irradiance falls (June).
- **The irradiance sensor of the weather station is unreliable** (`03_sensorcheck.png`): Mar–Apr 2021 ≈ 0; May 2021 –
  14 Feb 2023 ~20 % too low (P99 ≈ 0.80 × clear sky); from 15 Feb 2023 impossibly high (P99 1.2–1.5 × clear sky,
  peaks up to 3,600 W/m²). All stations jump on the same day; the PV power itself stays stable.

**Choice (decision #25):** SQ_Apartment37_48 (16 kW AC), LSK_South (25), SQ1 (28), SQ4 (33), SQ567 (48); all flat.
**Honesty note:** the first selection rule let 0 stations through; after that 2 measures were changed (shading
index over hours 8–16, correlation per quarter). The old columns are still in `stations_inventaris.csv`.

## Step 2 – Cleaning to 15 min (`taak5a_hkust_data.py` → `data/extern/hkust/clean/hkust_15min.csv`)
- PV: sum of the inverters, mean per quarter-hour only if all 3 five-minute values exist. Weather: ≥ 10 of 15 minutes.
  Label = start of the quarter-hour (as in Kaggle).
- Flags: `DAG` (cos zenith > 0.02), `SENSOR_OK` (#26), `STORING` (fault; SQ1: 2021-05, 2023-05), `UITVAL` (failure:
  P < 1 % of kWp AC while ERA5 GHI > 300 W/m²). Usable during the day: 87–93 % per station.
- **Wind and temperature vs ERA5** (`02_wind_temp_vs_era5.png`): temperature r ≈ 0.96 (sensor 1–2 °C warmer in
  summer). Wind r ≈ 0.63–0.70 per hour; sensor on average 10–20 % lower than ERA5 and decreasing further from mid-2022
  (2023: 2.3 vs 2.9 m/s). **Assumption:** possibly sheltering or ageing of the anemometer; to be used with care in
  the Faiman fit (Task 6).

## Step 3 – Tests (`taak5a_hkust_tests.py` → `resultaten/taak5/hkust/stap3_tests/`)
Setup fixed in advance (decision #28). Flat roofs → POA ≈ GHI. Capacity = 99th percentile from the training.

**ERA5 time shift (2021 only):** −30 min (r ≈ 0.80–0.82). That is the official Open-Meteo convention (mean over the
preceding hour); for Kaggle +15 fitted better. **Assumption:** the difference comes from the plant sensor and the
estimated location in Kaggle; HKUST has exact coordinates.

**(1) Upper limit with the station's own sensor (train 2021, test 2022):**

| Model | nRMSE % | R² | daily error % |
|---|---|---|---|
| Baseline mean day (per month) | 22.7 | 0.37 | 45.5 |
| Baseline persistence | 23.1 | 0.34 | 33.2 |
| Smart persistence | 19.3 | 0.54 | 31.3 |
| LR GHI (ERA5) | 15.4 | 0.71 | 20.7 |
| **Sensor: LR irradiance** | **8.6** | 0.91 | 9.5 |
| Sensor: Ridge (irradiance, temp., time) | 8.5 | 0.91 | 10.3 |

Same pattern as Kaggle (sensor ~5 %, ERA5 ~14–15 %): with its own sensor the error halves; the difference is the
error of the weather model (ERA5 sees the clouds at the wrong moment / in the wrong place).

**(3) Retrain choice on 2022 (2 folds):** **LR GHI (ERA5)**, 15.38 %. Seven models are within 0.5 pp
(LR GHI/POA/EFF_T, e5, e6, Ridge e6 (+ extra)); Random Forest (16.2 %) and Gradient Boosting (15.6 %) are not better.
As with Kaggle: with ERA5 as input the model form matters little; the weather determines the error.

**Test 2023 (after approval; choice stays LR GHI).** nRMSE %, 90 % interval = bootstrap over 365 test days.

| Model | A: 2021 + 2022 | A': 2021 only | B: walk-forward | dry | monsoon |
|---|---|---|---|---|---|
| Baseline mean day (per month) | 21.8 | 21.4 | 22.0 | 21.0 | 22.9 |
| Baseline persistence | 22.6 | 22.4 | 22.8 | 19.5 | 26.3 |
| Smart persistence | 18.5 | 18.3 | 18.6 | 16.4 | 21.0 |
| **LR GHI (ERA5), chosen** | **16.6** (16.0–17.3) | 16.8 | 16.6 | 14.5 | 19.1 |
| e6 | 16.7 | 17.1 | 16.7 | 14.5 | 19.2 |
| Ridge e6 + extra weather | 16.6 | 17.1 | 16.6 | 14.5 | 19.1 |
| Gradient Boosting | 16.6 | 17.5 | 16.5 | 14.3 | 19.3 |
| Random Forest | 17.0 | 17.7 | 17.0 | 14.7 | 19.8 |
| Zero-shot 2a: Kaggle final model (LR POA) | 26.6 | – | – | 25.2 | 28.4 |
| Zero-shot 2a: Kaggle e6 | 31.5 | – | – | 30.4 | 32.9 |
| Zero-shot 2b: pvlib physics | 18.7 | – | – | 17.3 | 20.6 |

- LR GHI is significantly better than the mean day (−5.2 pp), smart persistence (−1.8 pp), pvlib zero-shot (−2.1 pp)
  and Kaggle zero-shot (−10 pp). Among the ERA5 models: < 0.5 pp, not relevant.
- Per station 16.0–17.1 %; per month 11.5 % (Dec) to 21.4 % (Aug). Monsoon ~4.5 pp worse than the dry season.
- Residuals (`04_residuen.png`): largest errors around noon and in partly cloudy weather (KT 0.4–0.9); strong
  autocorrelation (ERA5 puts clouds hours off); median slightly positive in the dry season (forecast too high).

**Diagnosis in hindsight (after seeing 2023, no new model; `diagnose_*.csv`):**
- **Zero-shot = a level problem, not a shape problem.** Kaggle forecasts 1.4–2.0× too much, pvlib 1.2–1.4×. With one
  scale factor per station everyone is at 15.6–18 % (the same as the retrain). Causes: (a) Kaggle FRAC × kWp AC
  assumes that the system reaches its AC power; at SQ567 and SQ_Apartment the real capacity is 0.80 and 0.74 × AC
  (DC/AC ≤ 1); (b) the Kaggle intercept (0.058) gives 12 % power in low light (really 3 %); (c) the HKUST roofs
  deliver less per unit of ERA5 irradiance than pvlib expects (**assumption:** shading, dirt on flat panels, or ERA5
  too high near the coast).
- **The yield per unit of ERA5 irradiance decreases every year** (all 5 stations, also within the same season; e.g.
  SQ4 0.74 → 0.71 → 0.68). Because of this the retrain also forecasts 7–22 % too much in 2023. According to the
  datasheet: 2.5 % in year 1 and 0.6 %/year after that – this explains only part. **Assumption:** soiling of flat
  panels or drift in ERA5.

**What does this mean for a house in India?** The retrain score (16.6 %) is very similar to Kaggle (14–15 %): the
method also works in another climate and over a whole year. Zero-shot (without own data) is **not** good enough with
the Kaggle model; the physical pvlib model is much better (18.7 %), but still above smart persistence. So a house
**needs its own measured data to calibrate the level** (and recalibrate regularly after that, because of the decrease).

## Step 4 – Calibration test "new house" (`taak5a_hkust_kalibratie.py` → `resultaten/taak5/hkust/stap4_kalibratie/`)
**TEST IN HINDSIGHT** (2023 already viewed in step 3; setup fixed in advance in decision #31, chosen by the agent at
Dimme's request). 30 "new houses" (5 stations × installation on the 1st of Jan…Jun 2023). One level factor
f = Σ measured / Σ forecast from the first N weeks; evaluation always in weeks 13–26 after installation.

| Mean nRMSE % (30 houses) | N = 0 | 1 wk | 2 wk | 4 wk | 8 wk | 13 wk |
|---|---|---|---|---|---|---|
| pvlib × f | **19.8** | 22.0 | 20.2 | 19.4 | 19.4 | 18.9 |
| Kaggle final model × f | 26.8 | 21.8 | 20.4 | 19.7 | 19.7 | 19.2 |
| LR GHI on the first N weeks only | – | 20.8 | 19.7 | 18.9 | 18.8 | 18.4 |

References on the same quarter-hours: pvlib with oracle f 17.8 · LR GHI with 2 years of history 18.2 · smart
persistence 20.2 · mean day 22.7 · persistence 25.2.

**Facts**
- According to the rule fixed in advance (within 0.5 pp of the oracle) **no N is enough**, not even 13 weeks.
- **Calibrating on 1–2 weeks is on average worse than not calibrating** (pvlib). From 4 weeks it helps (−0.4 pp).
- The Kaggle model does need calibration (26.8 → 19.7 % after 4 weeks), but stays just behind pvlib.
- With 4+ weeks of own data an LR on only those weeks is as good as or slightly better than pvlib × f.
- Cause: f depends on the season of the calibration. Installation 1 April → f from week 1 ≈ 0.33–0.40 (all
  stations); installation 1 January → f ≈ 1.1–1.2. **Assumption:** in spring (fog, drizzle) ERA5 sees too much sun;
  in winter too little.

**What does this mean for the app?** Without own data: pvlib with the system data (not the Kaggle model). After that
**calibrate continuously** (e.g. update the factor or the model on the last weeks every week) instead of once.
This was not tested at this point; it was a proposal for a next test (fixed in advance; see "Follow-up tests").

---

# Task 5b – Day-ahead and horizon: Open Climate Fix UK PV (2024–2025)

**Source:** huggingface.co/datasets/openclimatefix/uk_pv (CC BY 4.0, doi:10.57967/hf/0878), data from Sheffield Solar.
Gated: free account, accept the conditions, and `HF_TOKEN` as an environment variable (never in the code). Only the
chosen systems are streamed with polars. **Climate different from India: this tests the forecasting, not the location.**

## Step 1 – Choosing systems (`taak5b_ukpv_systemen.py` → `resultaten/taak5/ukpv/stap1_systemen/`)
**Facts**
- Times in UTC; a row with time t = energy of (t − 30 min, t]. Power = `generation_Wh` × 2. We stay at 30 min (not
  15 min: that would invent data). The data runs until 2 April 2025 → period **Jan 2024 – Mar 2025**.
- In Apr, May, Jul, Aug, Dec 2024 and Jan–Feb 2025, 1–4 whole days are missing for almost all systems at the same
  time (outage in the data stream). Completeness is therefore measured against the dataset maximum per month
  (decision #34).
- Filter (#33–#35): 23,493 systems after metadata (1.5–10 kWp, tilt 10–60°, east–south–west, no bad_data) →
  462 complete → pool of 150 (30 per latitude band) → 99 through the quality and plausibility check → **40 chosen**
  (8 per band, max. 1 per location, seed 42).
- Chosen: 1.5–4 kWp, tilt 20–45°, orientation 90–270° (also east and west), yield 565–985 kWh/kWp per year.
- After the first choice 5 systems stood out with an illogical yield (358–488 kWh/kWp or P_max 1.34 × kWp):
  plausibility requirement added before any model (#35).

## Step 2 – Fetching the weather (`taak5b_ukpv_weer.py` → `data/extern/uk_pv/weer/`, `resultaten/taak5/ukpv/stap2_weer/`)
Open-Meteo (CC BY 4.0), everything in UTC, cached (decision #37).
- **Day-ahead:** ECMWF IFS 0.25° (Previous Runs), forecast 24/48 hours before the time. Archive only from
  **6 March 2024**; 100 % filled after that. (Checked: `ecmwf_ifs` 9 km is empty in Previous Runs; UK Met Office only
  from Aug 2024; ICON/GFS from 19 Jan 2024.)
- **Horizon:** ECMWF IFS 9 km (Single Runs), 96 runs (00 UTC, every 4th day, 14 Mar 2024 – 29 Mar 2025). Days 1–10
  ~100 % filled; days 11–15 only 68 % (runs from 2024 only go 10 days ahead).
- **ERA5** (weather in hindsight) as the upper limit.
- **First check** (correlation power ↔ GHI per hour, median over 40 systems): ERA5 0.825; forecast day 1 0.812;
  day 2 0.792. So in the UK yesterday's forecast is almost as good as the weather in hindsight.

## Step 3 – Tests (`taak5b_ukpv_tests.py` → `resultaten/taak5/ukpv/stap3_tests/`)
Setup fixed in advance (decision #38). Training 6 Mar – 30 Jun 2024 (forecasts only from 6 Mar), test
1 Jul 2024 – 31 Mar 2025.

**Irradiance time shift:** −30 min (ERA5, Jan–Jun 2024), the same as HKUST and the official Open-Meteo convention.

**Choice on 2 folds (input = ECMWF forecast from 1 day earlier):** **LR POA**, 16.6 % nRMSE.

| Model (folds May, June 2024) | nRMSE % | R² | daily error % |
|---|---|---|---|
| Baseline mean day (K × clear sky) | 21.3 | 0.48 | 32.3 |
| Baseline persistence | 26.0 | 0.24 | 37.1 |
| Smart persistence | 23.4 | 0.38 | 36.7 |
| Zero-shot pvlib (kWp, tilt, orientation) | 19.2 | 0.58 | 26.0 |
| **LR POA (forecast)** | **16.6** | 0.69 | 19.1 |
| e6 / Ridge e6 + extra / Gradient Boosting | 16.9 / 16.8 / 16.8 | | |
| LR POA trained on ERA5, applied to the forecast | 16.7 | | |

With a **real** day-ahead forecast the error in the UK (16.6 %) is about the same as what we reached in India and
Hong Kong with ERA5 (weather in hindsight). Training on ERA5 or on the forecast makes almost no difference.

**Test Jul 2024 – Mar 2025 (after approval; choice stays LR POA).** 40 houses, 273 test days, 90 % interval = bootstrap.

| Model | nRMSE % | R² | daily error % |
|---|---|---|---|
| Baseline mean day (K × clear sky) | 20.3 | 0.38 | 47.1 |
| Baseline persistence | 24.2 | 0.12 | 52.0 |
| Smart persistence | 21.8 | 0.28 | 50.3 |
| Zero-shot pvlib (kWp, tilt, orientation) | 17.5 | 0.54 | 37.0 |
| **LR POA, forecast day 1 (chosen)** | **14.6** (14.3–14.9) | 0.68 | 27.2 |
| LR POA, forecast day 2 | 15.5 | 0.64 | 30.4 |
| Upper limit: LR POA with ERA5 (perfect weather) | 14.3 | 0.69 | 25.9 |
| e6 / Ridge e6 + extra / Gradient Boosting | 14.7 / 14.7 / 14.9 | | |

- Walk-forward per month: LR POA 14.8 % (frozen 14.6): extra training after June does not help.
- Per season: winter 12.2 · autumn 13.5 · summer 15.4 · spring (March) 17.3 %. Per system 10.8–22.8 % (median 14.0).
- All differences with the baselines and with day 2 are significant. Difference with perfect weather: **only 0.2 pp**.

**Horizon (Task 7):** chosen model, frozen, with ECMWF IFS Single Runs (00 UTC, ~68 runs per day in the test;
days 11–15 ~62 runs). Day 1 = 0–24 hours after the run.

| Days ahead | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| nRMSE % model | 14.4 | 15.6 | 17.1 | 18.3 | 18.8 | 20.4 | 22.7 | 23.4 | 24.5 |
| nRMSE % mean day | 19.4 | 20.7 | 20.3 | 21.0 | 19.4 | 20.7 | 20.3 | 20.7 | 20.5 |
| skill vs mean day | 0.26 | 0.24 | 0.16 | 0.13 | 0.03 | 0.01 | −0.12 | −0.13 | −0.20 |

**What does this mean?**
- With a **real** weather forecast from yesterday the error (14.6 %) is almost as small as with perfect weather
  (14.3 %). So in the UK tomorrow's weather model is hardly worse than the weather in hindsight; the rest of the error
  comes from clouds that ERA5 does not get exactly right either (the same pattern as Kaggle and HKUST).
- The forecast clearly helps up to **day 4**, hardly on days 5–6, and from **day 7 a mean day is better**. An app
  (FR4: coming days) can therefore forecast days 1–4 with the weather and show the mean day after that (or a blend;
  not tested at this point – see "Follow-up tests").
- UK climate ≠ India: this tests the **forecasting**, not the location.

## Follow-up tests (IN HINDSIGHT; setup fixed before running in decisions #41–#43)

**HKUST – continuous calibration** (`taak5a_hkust_kalibratie.py`, part 4 → `stap4_kalibratie/doorlopend_*.csv`).
The same 30 new houses, weeks 13–26. Every week a new factor from the last N weeks of own data.

| Method | mean nRMSE % | P10–P90 |
|---|---|---|
| pvlib without calibration | 19.8 | 16.6–22.4 |
| pvlib × fixed f from 13 weeks | 18.9 | 14.1–21.7 |
| **pvlib × f, every week from the last 4 weeks** | **18.1** | 15.1–19.5 |
| pvlib × f, last 2 / 8 weeks / everything since installation | 18.4 / 18.3 / 18.5 | |
| LR GHI with 2 years of history | 18.2 | |
| oracle f | 17.8 | |

→ A new house with pvlib and a factor that is updated every week on the last 4 weeks is, after a few weeks, as good
as a model with 2 years of history. The seasonal effect and the yearly decrease are followed automatically.

**UK – blending the horizon with the mean day** (`taak5b_ukpv_horizon_mix.py` → `resultaten/taak5/ukpv/stap4_horizon_mengen/`).
Forecast = w_d × model + (1 − w_d) × mean day; w_d per day ahead from the training period (Mar–Jun 2024).

| Days ahead | 1 | 2 | 3 | 4 | 5 | 7 | 10 | 15* |
|---|---|---|---|---|---|---|---|---|
| model weight w_d | 0.79 | 0.70 | 0.67 | 0.70 | 0.46 | 0.40 | 0.33 | 0.33* |
| model only | 14.4 | 15.6 | 17.1 | 18.3 | 18.8 | 22.7 | 23.4 | 24.5 |
| mean day only | 19.4 | 20.7 | 20.3 | 21.0 | 19.4 | 20.3 | 20.7 | 20.5 |
| **blending** | **14.0** | **15.3** | **16.4** | **17.4** | **17.2** | **19.9** | **20.3** | **20.4** |

\* days 11–15: no training runs that far ahead → weight of day 10 carried forward.
Mean over days 1–10: blending 18.0 % · fixed rule (model up to day 4) 18.8 · model only 19.8 · mean day 20.3.
→ Blending is the best on **every** day, also on day 1. For the app (FR4) this is the approach for "coming days".

---

# Task 6 – Faiman fit (`taak6_faiman.py` → `resultaten/taak6/`)
Setup fixed in advance (decision #44). Faiman: T_cell = T_air + irradiance / (u0 + u1 × wind). Default (assumption
from V4/V5): u0 = 25, u1 = 6.84. Question: does the model get better with fitted values, and with measured instead of
ERA5 wind?

| Dataset | What | Default | Fitted |
|---|---|---|---|
| Kaggle (days 25–34) | RMSE module temperature | 4.5 °C (bias −2.1) | **2.5 °C** (bias +0.9); u0 37.7, u1 0 |
| Kaggle | nRMSE power LR EFF_T / e6 | 13.46 / 12.66 % | 13.52 / 12.67 % |
| HKUST 2023, measured wind | nRMSE LR EFF_T / e6 | 16.56 / 16.47 % | 16.56 / 16.47 % |
| HKUST 2023, ERA5 wind | nRMSE LR EFF_T / e6 | 16.70 / 16.60 % | 16.63 / 16.53 % |

**Facts**
- On Kaggle the fitted Faiman forecasts the module temperature much better (error almost halved). ERA5 wind does not
  help here (u1 → 0): the wind on the ERA5 grid says little about the wind at the panel.
- On the **power** this has no effect (< 0.1 pp). A few °C difference changes the power by only ~1 %.
- HKUST (no module temperature, indirectly through the power): the optimum is at the edge of the grid and the grid is
  flat → the data cannot determine u0/u1 well.
- Measured wind (weather station) vs ERA5 wind: 0.14 pp difference → not relevant. (Moreover a doubtful anemometer,
  see Task 5a step 2.)

**Choice:** the default values stay (#45). The model error is in the irradiance (clouds), not in the temperature.

---

# Clouds from satellite (Dimme's question: "can a cloud model help?")

**Research (short):** running a cloud simulation ourselves (e.g. WRF-Solar) is not realistic for this project and an
app (computing cluster, and it needs ECMWF/GFS itself) and helps little for *tomorrow*: clouds above one roof cannot
be forecast exactly a day ahead. According to the literature, satellite images (extrapolating cloud motion) are better
than weather models up to ~5–6 hours ahead. Open-Meteo offers free satellite irradiance: Himawari-9 (India, Asia;
10 min, ~5 km, archive from 2015) and EUMETSAT IODC (India, from 2025).

## Step 5 HKUST – satellite as "weather in hindsight" (`taak5a_hkust_satelliet.py` → `resultaten/taak5/hkust/stap5_satelliet/`)
Setup fixed in advance (decision #46). Same model (LR GHI), same quarter-hours; only the source of the irradiance
differs. Satellite time shift +10 min (chosen on 2021).

| | ERA5 (31 km, 1 hour) | **Satellite Himawari (5 km, 10 min)** | station's own irradiance sensor |
|---|---|---|---|
| Correlation with the sensor (per quarter-hour) | 0.84 | **0.93** | – |
| Correlation with the power | 0.82 | **0.92** | – |
| nRMSE validation 2022 (train 2021) | 15.4 % | **10.1 %** | 8.6 % |
| nRMSE test 2023 (train 2021 + 2022) | 16.6 % | **10.9 %** (−5.7 pp, significant) | sensor broken |
| Dry / monsoon 2023 | 14.5 / 19.1 % | **8.3 / 13.8 %** | |

**What does this mean?** The satellite sees the clouds almost as the roof sees them: the error drops from 16.6 to
10.9 %, close to an own irradiance meter (8.6 %). So the largest error source was the coarse resolution of ERA5.
Satellite and ERA5 together do not help extra.
**Limitation:** the satellite measures the weather of *now* (30 min delay); it is not a forecast. Useful for:
(1) the coming hours (extrapolating clouds = nowcast), (2) calibrating a house without its own meter, (3) checking
whether a house works normally (detecting faults). For **tomorrow** the weather forecast is still needed.

**Status (16 Nov 2026):** the satellite / cloud track was stopped at Dimme's request (decision #50). The check in
India was not done (Open-Meteo only has satellite data for India from 2025). The HKUST result stays as a finding:
satellite is much closer to the roof than ERA5, but it is not a forecast for tomorrow.

---

# Improving the model (16 Nov 2026; decisions #51–#58)

**1. Regression test Kaggle (#55):** after the changes for Task 5 (time-zone parameter, moved band functions) the
control run (`SOLAR_RUN=split30_check`, phases 2–4) gives exactly the same numbers as the archive of run 1.

**2. Band P10–P90 with real forecasts (UK, `taak5b_ukpv_band.py` → `resultaten/taak5/ukpv/stap6_band/`)**

| Band | Test coverage (target 80 %) | Winter / summer | Width (% cap.) |
|---|---|---|---|
| Phase-4 method (factor per cloudiness class, calibrated on May–June) | 63.4 % | 51.7 / 71.5 | 30.6 |
| A: factor, every week on the last 4 weeks | 72.8 % | 77.6 / 71.0 | 32.8 |
| **B: additive per forecast class, every week on the last 4 weeks** | **79.4 %** | **80.2 / 79.5** | **29.8** |

Why the old band failed: in low sun "factor × forecast" is almost 0 wide (coverage 24 %), and the band was only
calibrated on May–June (in autumn/winter a house often delivers almost nothing while sun is forecast). With more days
ahead the old band gets even worse (44–64 %). **Variant B goes into final model v2.**

**3. Final model v2 – "new house" (`eindmodel_v2.py`, `solarcore/model_v2.py` → `modellen/model_final_v2.joblib`)**
pvlib (kWp, tilt, orientation) × factor from the last 4 weeks, blended with the mean day (also from the last
4 weeks), band variant B. **No training period needed**: after 4 weeks of own measured data the model is complete.

| UK test Jul 2024 – Mar 2025 (40 houses) | nRMSE % | R² | daily error % |
|---|---|---|---|
| LR POA (step 3; needs Mar–Jun 2024 as training) | 14.58 | 0.68 | 27.2 |
| **v2: pvlib × f, blended (w = 0.87)** | **14.32** | 0.69 | 25.2 |
| v2: pvlib × f, without blending | 14.33 | 0.69 | 24.0 |
| v2: mean day only | 19.33 | 0.44 | 44.7 |

- v2 is significantly slightly better than LR POA (−0.25 pp; 90 %: −0.35…−0.15) and needs no training.
- Band v2: 79.3 % coverage (all seasons 78.6–79.9 %).
- Horizon (blended per day): day 1 14.0 · day 2 15.1 · day 4 17.3 · day 7 19.7 · day 10 20.0 · day 15 20.8 %.
- HKUST (Hong Kong) showed the same: continuous calibration 18.1 % ≈ 2 years of history (#43).
- **Assumptions:** pvlib parameters (PARAMS, DC/AC 1.1), blending weights from the UK training period, band in
  fraction of kWp.
- Compatible with the app: same keys as `model_v5.joblib` (`model, features, factor, band_method, band, kt_bins,
  params, rad_shift_min, min_cosz`) + a `v2` block with the new components.

**4. Several weather models for tomorrow (UK, `taak5b_ukpv_multimodel.py` → `resultaten/taak5/ukpv/stap5_multimodel/`, #52/#58)**

| Test Jul 2024 – Mar 2025 (day 1) | nRMSE % | vs ECMWF |
|---|---|---|
| **ECMWF IFS only (current)** | **14.58** | – |
| DWD ICON only | 15.59 | +1.0 pp |
| NOAA GFS only | 17.32 | +2.8 pp |
| Mean of the 3 (chosen with rule #17) | 14.83 | +0.25 pp (worse) |
| LR on the 3 models | 14.38 | −0.19 pp |

→ No relevant gain (≤ 0.2 pp, and the model chosen by the rule is even worse on the test). ECMWF alone stays; that
is also what final model v2 uses.

## Summary "improving the model"
| What | Result | Consequence |
|---|---|---|
| Regression test Kaggle | identical to run 1 | Kaggle results remain valid |
| Band with real forecasts | old band 63 %, new band (B) 79 % | band B in v2 |
| Final model v2 | 14.3 % without training (LR POA 14.6 % with training) | official final model for the app (#60) |
| Several weather models | ≤ 0.2 pp | not worth it |

## TASK 8 – Feedback pitch week 9 (`taak8_feedback_pitch9.py` → `resultaten/taak8/uk`, `/hkust`, decisions #67–#70)

Teachers' tips after pitch week 9 (plus the Orbit idea from the interim feedback), each tested against final model v2 on
**two datasets**: UK PV (40 homes, real ECMWF forecasts, train Mar–Jun 2024, test Jul 2024 – Mar 2025) and HKUST
(5 stations, ERA5, train 2021–2022 with all seasons, test 2023). Rule fixed before running: adopt only if ≥ 0.3 pp
better than v2 **and** the 90 % bootstrap interval of the difference is below 0.

| Tip | Best variant | UK vs v2 (pp) | HKUST vs v2 (pp) | Verdict |
|---|---|---|---|---|
| – | **v2 (reference)** | 14.32 % | 16.58 % | std over days 3.9 / 6.1 pp |
| Stream learning | update f every day, exponential forgetting (half-life 14 d) | **−0.12** | **−0.55** | only tip that wins on both; proposed |
| Decision tree on the residual of v2 | depth 8 / 3 | +0.81 | −0.31 | only helps with a full year of training; no cold start |
| Decision tree instead of LR | depth 5 | +0.13 | −0.27 (not significant) | not adopted |
| Tree as switch between techniques | v2 or the home's average day | 0.00 | +0.01 | almost always picks v2; blending already does this |
| Calibration window | 1 / 2 / 8 weeks / all data | +0.30 / +0.17 / −0.01 / +0.03 | +0.63 / +0.31 / −0.07 / −0.08 | 4–8 weeks is fine; shorter is worse |
| Mean/std of past clearness as features | all previous days | −0.10 | −0.17 | too small, needs training |
| Bayesian calibration (core of Orbit, Kalman filter) | q chosen on validation | +0.29 | −0.41 (not significant) | no gain over exponential forgetting; Orbit itself needs Stan |

**For the report:** the error comes from the weather (clouds), not from the model form. More complex models (trees)
only help when a full year of history is available, which a new home does not have. The one change that helps on both
datasets is updating the calibration every day with exponential forgetting (stream learning). Final choice: team.

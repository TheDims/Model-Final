# SolarHome Forecast – final model

Day-ahead solar power forecasting for homeowners in India with rooftop solar panels.
Student project of the **Minor in Applied Data Science for Engineering (MADS4Engineering)**, ISEP, Porto, 2026.

**Team:** Woohyun Jin, Dimme Plug, Johannes Norrgård
**Project report (wiki):** <https://www.mads4e2026-wiki1.dee.isep.ipp.pt/doku.php?id=report>

> The detailed documentation (`README.md`, `BESLISSINGEN.md`, `AGENTS.md`) is written in **Dutch**.
> This page is a short English overview.

---

## What it does
1. Learns how solar power depends on the weather, using 34 days of data from two solar plants in India (Kaggle).
2. Replaces the plant's own sensor with a weather model (**ERA5**), because a household only has a weather service.
3. Turns this into a forecast for a house: `kW = FRAC × kWp × factor × losses`, with a likely range (P10–P90).

The pipeline follows CRISP-DM in five phases:

| Phase | Script | Question |
|---|---|---|
| 0 – data | `fase0_data.py` | Read, check and clean the Kaggle data |
| 1 – benchmark | `fase1_benchmark_kaggle.py` | How well does regression explain power from the plant's own sensor? (upper limit) |
| 2 – forecasting | `fase2_forecast.py` | What can we predict with only past data (+15 min … day-ahead)? |
| 3 – weather model | `fase3_weermodel.py` | How good is the forecast with ERA5 weather instead of the sensor? |
| 4 – final model | `fase4_eindmodel.py` | Choose, calibrate the band, save the model |
| extra | `leercurve.py`, `heatmaps.py`, `diagnose_era5.py` | Learning curve, figures for the report, error analysis |

## How we test
- Split **in time, by whole days, never at random**. Everything estimated from data (capacity, scaling, band) comes
  only from the training days.
- The model is chosen **before** we look at the test days, with a fixed rule: lowest nRMSE on validation folds;
  if the difference is below 0.5 percentage points, the simplest model wins.
- Main metric: **nRMSE** (RMSE / capacity, daytime only). Also MAE, RMSE, R², daily energy error and skill score.
- Always compared with simple baselines: *mean day* and *persistence*.

Two runs:

| Run | Training / test | Final model | Results |
|---|---|---|---|
| Run 1 `split30` | day 1–10 / day 11–34 | Linear regression on ERA5 horizontal irradiance (GHI) | `resultaten/split30/` |
| **Run 2 `split70`** (main) | **day 1–24 / day 25–34** | **Linear regression on ERA5 irradiance on the panel (POA)** | `resultaten/split70/` |
| Learning curve | 10 / 17 / 24 / 27 days, same test days 28–34 | – | `resultaten/leercurve/` |

Main result of run 2 (test days 25–34, frozen model): the final model has an nRMSE of about **14 %**, against
**17 %** for the mean-day baseline. With the plant's own sensor (not available to a house) the error would be
about 5 %: the **quality of the weather data** is the main limitation, not the model.
All numbers, choices and limitations: see `README.md` and `BESLISSINGEN.md`.

## Repository structure
```
├── fase0_data.py … fase4_eindmodel.py   pipeline (run in this order)
├── leercurve.py, heatmaps.py, diagnose_era5.py
├── solarcore/          shared functions: config, data preparation, physics (pvlib), models, evaluation, plots
├── data/               ERA5 cache (Kaggle files: download yourself, see data/README.md)
├── modellen/           saved models (.joblib)
├── resultaten/         tables (.csv), figures (.png) and model_info.json per run
├── README.md           full results per phase (Dutch)
├── BESLISSINGEN.md     decision log with date, reason and alternative (Dutch)
└── AGENTS.md           working rules for AI coding agents and team members (Dutch)
```

## Getting started
1. Python 3.13 (we use Anaconda). Install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the Kaggle data into `data/` (see `data/README.md`).
3. Run the scripts in order: `fase0_data.py` → `fase1_benchmark_kaggle.py` → `fase2_forecast.py` →
   `fase3_weermodel.py` → `leercurve.py` → `fase4_eindmodel.py` (then optionally `heatmaps.py`, `diagnose_era5.py`).
   Every script also runs with F5 in Spyder (`# %%` cells).
4. The run is set with the environment variable `SOLAR_RUN` (default `split70`). Settings are in `solarcore/config.py`.

## Limitations
- Only 34 days (May–June 2020, start of the monsoon) from two large plants, not from houses.
- Plant locations are estimated (best match with ERA5), not known.
- ERA5 is *past* weather, not a forecast: with a real weather forecast the error will be larger.
- The factor from plant to house (kW per kWp) is an assumption that has not been checked with real home data.

## Data and software
- Kaggle "Solar Power Generation Data" (Ani Kannal) – not included, see `data/README.md`.
- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ERA5 from the Copernicus Climate Change Service.
- [pvlib](https://pvlib-python.readthedocs.io/), scikit-learn, pandas, NumPy, matplotlib, statsmodels.

# Data

## Kaggle – Solar Power Generation Data (not included)
The raw Kaggle files are **not** in this repository, because we could not confirm the licence of the dataset.
Download them yourself and put the four CSV files in this `data/` folder (same names):

- Dataset: <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data> (Ani Kannal)
- Files: `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`,
  `Plant_2_Generation_Data.csv`, `Plant_2_Weather_Sensor_Data.csv`
- Content: 2 solar plants in India, 15 May – 17 June 2020 (34 days), every 15 minutes, 22 inverters per plant.

Then run `fase0_data.py`. It cleans the data and writes `data/clean/<run>/plant_15min.csv`
(this folder is also not in the repository; it is always rebuilt from the raw files).

## ERA5 weather (included)
`era5_14.5_77.5_2020-04-01_2020-06-18.csv` and `era5_19.0_79.0_2020-04-01_2020-06-18.csv`:
hourly ERA5 reanalysis for the **estimated** plant locations, downloaded with the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), based on ERA5 from the
Copernicus Climate Change Service (C3S).

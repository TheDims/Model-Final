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

## External validation datasets (TASK 5, not included)
These folders are in `.gitignore` (`data/extern/`). Download them yourself; the scripts cache everything else.

- **HKUST rooftop PV, Hong Kong 2021–2023** (CC0): <https://doi.org/10.5061/dryad.m37pvmd99> (version 8 Sep 2024).
  Download `Dataset.zip` and unzip it so that `data/extern/hkust/raw/Dataset/` contains `Metadata/` and
  `Time series dataset/`. Then run `taak5a_hkust_stations.py` → `taak5a_hkust_data.py` → `taak5a_hkust_tests.py`.
- **Open Climate Fix UK PV** (CC BY 4.0): <https://huggingface.co/datasets/openclimatefix/uk_pv> is gated.
  Make a free Hugging Face account, click "Agree and access repository", create a **Read** token and set it as the
  environment variable `HF_TOKEN` (never put it in the code). Then run `taak5b_ukpv_systemen.py` →
  `taak5b_ukpv_weer.py` → `taak5b_ukpv_tests.py`.
- **Weather / satellite (Open-Meteo, CC BY 4.0):** ERA5, archived forecasts and satellite radiation are downloaded
  automatically the first time (internet needed) and cached in `data/extern/`.
- **NSRDB Meteosat IODC (India satellite, optional):** free API key from <https://developer.nlr.gov/signup/>,
  set as environment variable `NLR_API_KEY`.

Every person uses their **own** accounts and keys.

# -*- coding: utf-8 -*-
"""
Kaggle-data inlezen, controleren en opschonen (fase 0)
======================================================
Regels (AGENTS.md §5.2, zelfde als wiki en V5):
  - datumformaat expliciet per bestand (Plant 1 generation = dag-maand-jaar);
  - Plant 1: DC_POWER / 10 (alleen voor de controle DC/AC, DC is nooit een input);
  - Plant 2: TOTAL_YIELD niet gebruiken;
  - omvormer-uitval (0 kW terwijl de rest draait) weglaten -> we voorspellen normale werking;
  - gemiddelde per actieve omvormer i.p.v. de som; < 8 actieve omvormers -> AC_KW = NaN;
  - volledig 15-min-rooster; sensorgaten <= 1 uur interpoleren; AC_KW NOOIT interpoleren (geen opwek verzinnen).
"""
import numpy as np
import pandas as pd

from .config import (DATA, DATE_FORMATS, PERIOD, FREQ, DC_FIX_PLANT1, MIN_INVERTERS,
                     OUTAGE_MEDIAN_FRAC, MAX_GAP_QUARTERS, SENSOR_COLS)


# ---------- inlezen ----------
def read_generation(p):
    """Generation-csv van plant p, met de juiste datums en de DC-fix voor Plant 1."""
    gen = pd.read_csv(DATA / f"Plant_{p}_Generation_Data.csv")
    gen["DATE_TIME"] = pd.to_datetime(gen.DATE_TIME, format=DATE_FORMATS[("gen", p)])
    if p == 1:
        gen["DC_POWER"] = gen.DC_POWER / DC_FIX_PLANT1
    return gen


def read_sensor(p):
    """Weather-sensor-csv van plant p (IRRADIATION in kW/m², temperaturen in °C)."""
    s = pd.read_csv(DATA / f"Plant_{p}_Weather_Sensor_Data.csv")
    s["DATE_TIME"] = pd.to_datetime(s.DATE_TIME, format=DATE_FORMATS[("sensor", p)])
    return s


def check_df(df, name, quantiles=(0.05, 0.50, 0.95, 0.99)):
    """Snelle datacheck (idee: notebook Mehmet Isik): vorm, types, NA, dubbele rijen, kwantielen."""
    num = df.select_dtypes("number")
    info = {"bestand": name, "rijen": len(df), "kolommen": df.shape[1],
            "types": df.dtypes.astype(str).to_dict(), "NA": df.isna().sum().to_dict(),
            "dubbele_rijen": int(df.duplicated().sum()),
            "van_tot": (str(df.DATE_TIME.min()), str(df.DATE_TIME.max())),
            "kwantielen": num.quantile(list(quantiles)).round(3).to_dict()}
    print(f"\n--- check_df: {name} ---")
    print(f"vorm {df.shape}, {info['dubbele_rijen']} dubbele rijen, periode {info['van_tot'][0]} t/m {info['van_tot'][1]}")
    print("NA per kolom:", {k: v for k, v in info["NA"].items() if v} or "geen")
    print(num.quantile(list(quantiles)).round(3).to_string())
    return info


# ---------- opschonen ----------
def mark_outages(gen):
    """Kolom UITVAL per omvormer-rij (overgenomen uit V5 solarhome/plant_data.py):
    AC <= 0 terwijl de mediaan van alle omvormers op dat moment > 5 % van het plant-maximum is."""
    med = gen.groupby("DATE_TIME").AC_POWER.transform("median")
    return gen.assign(UITVAL=(gen.AC_POWER <= 0) & (med > OUTAGE_MEDIAN_FRAC * gen.AC_POWER.max()),
                      ZONNIG=med > OUTAGE_MEDIAN_FRAC * gen.AC_POWER.max())


def plant_15min(p):
    """Eén plant -> tabel per kwartier: AC_KW (gemiddeld per actieve omvormer), N_INV, N_UITVAL,
    UITVAL, sensorwaarden. Plus een dict met tellingen voor de samenvatting."""
    gen = mark_outages(read_generation(p))
    ok = gen[~gen.UITVAL]
    g = ok.groupby("DATE_TIME").agg(AC_KW=("AC_POWER", "mean"), N_INV=("AC_POWER", "size"))
    g["N_UITVAL"] = gen.groupby("DATE_TIME").UITVAL.sum()
    g.loc[g.N_INV < MIN_INVERTERS, "AC_KW"] = np.nan
    grid = pd.date_range(*PERIOD, freq=FREQ, name="DATE_TIME")
    d = g.reindex(grid)
    d["N_UITVAL"] = d.N_UITVAL.fillna(0).astype(int)
    d["UITVAL"] = d.N_UITVAL > 0

    sens = read_sensor(p).set_index("DATE_TIME")[SENSOR_COLS]
    sens = sens[~sens.index.duplicated()].reindex(grid)
    missing_before = sens.isna().sum()
    # overgenomen uit V1 model_v1.py: korte gaten (<= 1 uur) opvullen, alleen binnen de reeks
    sens = sens.interpolate(limit=MAX_GAP_QUARTERS, limit_area="inside")
    d = d.join(sens).reset_index()
    d.insert(1, "PLANT", p)

    stats = {"rijen_generation": len(gen), "omvormers": int(gen.SOURCE_KEY.nunique()),
             "tijdstempels_generation": int(gen.DATE_TIME.nunique()),
             "uitval_rijen": int(gen.UITVAL.sum()),
             "uitval_%_van_zonnige_omvormer_kwartieren": float(gen.UITVAL.sum() / gen.ZONNIG.sum() * 100),
             "kwartieren_rooster": len(grid),
             "kwartieren_zonder_generation": int(g.reindex(grid).N_INV.isna().sum()),
             "kwartieren_minder_dan_8_omvormers": int((g.N_INV < MIN_INVERTERS).sum()),
             "kwartieren_met_uitval": int(d.UITVAL.sum()),
             "sensor_gaten_voor": missing_before.astype(int).to_dict(),
             "sensor_gaten_na_interpolatie": d[SENSOR_COLS].isna().sum().astype(int).to_dict(),
             "AC_KW_NaN": int(d.AC_KW.isna().sum()),
             "IRRADIATION_max_kW_m2": float(d.IRRADIATION.max())}
    return d, gen, stats


def flag_dips(clean, setup_days):
    """Vlag DIP: plant-brede dip (BESLISSINGEN #10). De sensor meet zon (> 0,2 kW/m²), maar de hele plant
    levert < 30 % van de verwachte opwek. Verwacht = straling x mediane verhouding AC_KW/IRRADIATION van die
    plant, ALLEEN uit de opzetdagen (geen info uit de testdagen). Niet uit het weer te voorspellen ->
    weglaten bij trainen en beoordelen, net als omvormer-uitval. Geeft (tabel, verhouding per plant)."""
    from .config import DIP_MIN_IRR, DIP_MAX_RATIO
    sun = clean[clean.DAY.isin(setup_days) & (clean.IRRADIATION > DIP_MIN_IRR) & clean.AC_KW.notna()]
    k = (sun.AC_KW / sun.IRRADIATION).groupby(sun.PLANT).median()
    expected = clean.IRRADIATION * clean.PLANT.map(k)
    dip = (clean.IRRADIATION > DIP_MIN_IRR) & (clean.AC_KW < DIP_MAX_RATIO * expected)
    return clean.assign(DIP=dip.fillna(False)), k


def dc_ac_per_day(gen):
    """DC/AC-verhouding per dag (na de DC-fix), alleen omvormer-rijen zonder uitval en met opwek."""
    g = gen[~gen.UITVAL & (gen.AC_POWER > 0)]
    s = g.groupby(g.DATE_TIME.dt.normalize())[["DC_POWER", "AC_POWER"]].sum()
    return s.DC_POWER / s.AC_POWER


def inverter_daily_kwh(gen):
    """Opwek per omvormer per dag (kWh) uit AC_POWER (kW, per kwartier), met uitval erin (ruwe staat)."""
    e = gen.groupby(["SOURCE_KEY", gen.DATE_TIME.dt.normalize()]).AC_POWER.sum() * 0.25
    return e.rename("kWh").reset_index()

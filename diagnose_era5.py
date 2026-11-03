# -*- coding: utf-8 -*-
"""
DIAGNOSE  -  Waarom scoort het gekozen ERA5-model ongeveer even goed als smart persistence zonder weer?
=================================================================================================
Alleen uitleg, GEEN nieuwe keuze (de keuze in fase 3/4 staat vast). Zelfde opzet: bevroren op de trainingsdagen,
getest op de testdagen van de run.

  1. Fout opsplitsen: bias (vaste afwijking) + dagniveau (hoe zonnig is de dag in totaal) + binnen de dag
     (wolken op het verkeerde kwartier). MSE = bias^2 + var(dagafwijking) + binnen-de-dag-deel.
  2. Weet ERA5 hoe zonnig vandaag wordt? Correlatie van de echte daghelderheid met (a) ERA5 van vandaag en
     (b) de gemeten helderheid van gisteren (wat smart persistence gebruikt).
  3. Hoeveel helpt bias-correctie of het combineren van beide (alleen ter informatie).
Runnen: F5 in Spyder of in VS Code (Anaconda-Python). Eerst fase 0-4.
"""

# %% Imports en paden
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parent
except NameError:          # Spyder/interactief
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from solarcore.config import RESULTS   # noqa: E402
from solarcore.dataset import weather_frame   # noqa: E402
from solarcore.features import capacity, add_target   # noqa: E402
from solarcore.forecast_features import add_clear, add_daily, dayahead   # noqa: E402
from solarcore.models import LADDER_FASE3, LADDER_FASE1   # noqa: E402

warnings.filterwarnings("ignore")
OUT = RESULTS / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
f3 = json.loads((RESULTS / "fase3" / "samenvatting.json").read_text(encoding="utf-8"))
RAD_SHIFT, N_SETUP = int(f3["tijdafspraak_min"]), int(f3["opzet_dagen"].split("-")[1])
data = weather_frame(RAD_SHIFT)
SETUP, TEST = list(range(1, N_SETUP + 1)), list(range(N_SETUP + 1, int(data.DAY.max()) + 1))

# %% 1. Voorspellingen (bevroren, zelfde als fase 3)
d = add_target(data, capacity(data[data.DAY.isin(SETUP)]))
d.loc[d.DIP, "FRAC"] = np.nan
tr, te = d[d.DAY.isin(SETUP) & d.OK], d[d.DAY.isin(TEST) & d.OK].copy()
k = dayahead(add_daily(add_clear(d, d[d.DAY.isin(SETUP)]))).loc[te.index]
prof = tr.groupby(["PLANT", "SLOT"]).FRAC.mean()
P = {"gemiddelde dag": np.nan_to_num(prof.reindex(list(zip(te.PLANT, te.SLOT))).values),
     "smart persistence (geen weer)": (k.KD_1.fillna(k.KD.mean()) * k.FRAC_CS).values}
EIND = f"{f3['keuze'].split(' (')[0]} (ERA5) - eindmodel"
for name, lab in [(f3["keuze"], EIND), ("e6: EFF_T x (a + b KT_POA), zonder intercept", "e6 (ERA5)")]:
    cols, make, _ = LADDER_FASE3[name]
    P[lab] = make().fit(tr[cols], tr.FRAC).predict(te[cols])
cols, make, _ = LADDER_FASE1["LR straling"]
P["sensor (bovengrens)"] = make().fit(tr[cols], tr.FRAC).predict(te[cols])
P = {n: np.clip(v, 0, None) for n, v in P.items()}

# %% 2. Fout opsplitsen: bias + dagniveau + binnen de dag
rows = []
for n, y in P.items():
    r = pd.Series(y - te.FRAC.values, index=te.index)
    day_mean = r.groupby([te.PLANT, te.DAY]).transform("mean")          # gemiddelde afwijking per plant-dag
    bias = r.mean()
    dayvar = ((day_mean - bias) ** 2).mean()
    within = ((r - day_mean) ** 2).mean()
    rows.append({"model": n, "nRMSE_%": np.sqrt((r ** 2).mean()) * 100, "bias_%": bias * 100,
                 "deel_bias_%": bias ** 2 / (r ** 2).mean() * 100, "deel_dagniveau_%": dayvar / (r ** 2).mean() * 100,
                 "deel_binnen_dag_%": within / (r ** 2).mean() * 100,
                 "fout_dagniveau_%": np.sqrt(dayvar) * 100, "fout_binnen_dag_%": np.sqrt(within) * 100})
dec = pd.DataFrame(rows).set_index("model")
print(f"=== 1. Waar zit de fout? (aandeel van de kwadratische fout; test dag {TEST[0]}-{TEST[-1]}) ===")
print(dec.round(2).to_string())
dec.round(3).to_csv(OUT / "fout_opgesplitst.csv")

# %% 3. Weet ERA5 hoe zonnig vandaag wordt?
day = te.groupby(["PLANT", "DAY"]).agg(echt=("FRAC", "sum"), cs=("CS_GHI", "sum"), ghi=("GHI_W", "sum"),
                                       cs_w=("CS_GHI_W", "sum"))
day["K_echt"] = day.echt / day.cs                       # gemeten helderheid van de dag (relatief)
day["KT_ERA5"] = day.ghi / day.cs_w                     # helderheid volgens ERA5
day["K_gisteren"] = k.groupby([te.PLANT, te.DAY]).KD_1.first()
corr = {p: {"r(echt, ERA5 vandaag)": day.loc[p][["K_echt", "KT_ERA5"]].corr().iloc[0, 1],
            "r(echt, gemeten gisteren)": day.loc[p][["K_echt", "K_gisteren"]].corr().iloc[0, 1]} for p in (1, 2)}
corr["samen"] = {"r(echt, ERA5 vandaag)": day[["K_echt", "KT_ERA5"]].corr().iloc[0, 1],
                 "r(echt, gemeten gisteren)": day[["K_echt", "K_gisteren"]].corr().iloc[0, 1]}
print(f"\n=== 2. Hoe goed kent men de helderheid van vandaag? ({len(TEST)} testdagen per plant) ===")
print(pd.DataFrame(corr).round(3).to_string())
print(f"Spreiding daghelderheid: echt (CV) {day.K_echt.std() / day.K_echt.mean():.2f}, "
      f"ERA5 (CV) {day.KT_ERA5.std() / day.KT_ERA5.mean():.2f}")
day.round(4).to_csv(OUT / "daghelderheid.csv")

# %% 4. Ter informatie: bias-correctie en combineren (GEEN nieuwe keuze)
y = te.FRAC.values
lr = P[EIND]
kort = f3["keuze"].split(" (")[0]
info = {f"{kort} zoals gekozen": lr,
        f"{kort} zonder vaste afwijking (achteraf, alleen ter info)": lr - (lr - y).mean(),
        f"gemiddelde van {kort} en smart persistence": (lr + P["smart persistence (geen weer)"]) / 2}
print("\n=== 3. Ter informatie (verandert de keuze niet) ===")
for n, v in info.items():
    print(f"  {n:58s} nRMSE {np.sqrt(((np.clip(v, 0, None) - y) ** 2).mean()) * 100:.2f} %")

# %% 5. Hetzelfde, maar per uur en per dag gemiddeld (minder last van wolken op het verkeerde kwartier)
hour = te.DATE_TIME.dt.floor("h")
rows = []
for n, v in P.items():
    s = pd.DataFrame({"y": y, "f": v, "p": te.PLANT.values, "h": hour.values, "d": te.DAY.values})
    hr = s.groupby(["p", "h"])[["y", "f"]].mean()
    dy = s.groupby(["p", "d"])[["y", "f"]].mean()
    rows.append({"model": n, "nRMSE per kwartier_%": np.sqrt(((s.f - s.y) ** 2).mean()) * 100,
                 "nRMSE per uur_%": np.sqrt(((hr.f - hr.y) ** 2).mean()) * 100,
                 "nRMSE per dag_%": np.sqrt(((dy.f - dy.y) ** 2).mean()) * 100})
res = pd.DataFrame(rows).set_index("model")
print(f"\n=== 4. Fout per kwartier, per uur en per dag (gemiddeld vermogen; test dag {TEST[0]}-{TEST[-1]}) ===")
print(res.round(2).to_string())
res.round(3).to_csv(OUT / "fout_per_tijdschaal.csv")

# -*- coding: utf-8 -*-
"""
Modelladder (fase 1): van eenvoudig naar complex, één stap tegelijk
===================================================================
Elk model: inputs, een fabriek die een nieuw (ongetraind) sklearn-model maakt, en de 'eenvoud'
voor de keuzeregel: (complex? 0/1, aantal ruwe inputs, aantal features).
Preprocessing (scaler, polynoom) zit IN de Pipeline, zodat die alleen op de training wordt gefit.
"""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import RANDOM_STATE

ALL_SENSOR = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HOUR_SIN", "HOUR_COS", "COSZ"]
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]

LADDER_FASE1 = {
    # naam: (inputs, fabriek, (complex, ruwe inputs, features))
    "LR straling": (["IRRADIATION"], lambda: make_pipeline(StandardScaler(), LinearRegression()), (0, 1, 1)),
    "LR straling + moduletemp.": (["IRRADIATION", "MODULE_TEMPERATURE"],
                                  lambda: make_pipeline(StandardScaler(), LinearRegression()), (0, 2, 2)),
    "LR fysisch: straling x (1 + γ(T-25))": (["IRR_T"], lambda: make_pipeline(StandardScaler(), LinearRegression()),
                                             (0, 2, 1)),
    "Ridge (alle sensor-inputs + tijd)": (ALL_SENSOR, lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)),
                                          (0, 6, 6)),
    "Polynoom gr. 2 + Ridge (straling, moduletemp.)": (
        ["IRRADIATION", "MODULE_TEMPERATURE"],
        lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)),
        (0, 2, 5)),
    "Random Forest (alle)": (ALL_SENSOR, lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                                       max_features=0.6, n_jobs=-1,
                                                                       random_state=RANDOM_STATE), (1, 6, 6)),
    "Gradient Boosting (alle)": (ALL_SENSOR, lambda: HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
        random_state=RANDOM_STATE), (1, 6, 6)),
}
BASELINES = ["Baseline: gemiddelde dag", "Baseline: persistence (gisteren)"]

# ---------- fase 2: echt vooruit (target = heldere-hemelindex K, daarna x FRAC_CS) ----------
INTRA_K = ["K_L0", "K_L1", "K_L2", "K_L4", "KD_1"]                  # eerdere K's en gisteren
INTRA_ALL = INTRA_K + ["KS_L0", "HOUR_SIN", "HOUR_COS", "COSZ"]      # + sensor op t0 + klok/zon van het doel
DA_K = ["KD_1", "KD_2"]                                             # helderheid vandaag en gisteren
DA_ALL = DA_K + ["KSD_1", "HOUR_SIN", "HOUR_COS", "COSZ"]


def _lr():
    return make_pipeline(StandardScaler(), LinearRegression())


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def _rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.6, n_jobs=-1,
                                 random_state=RANDOM_STATE)


def _gb():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=RANDOM_STATE)


def ladder_fase2(dayahead):
    """Modellen per horizon: naam -> (inputs, fabriek, eenvoud). Baselines zitten in het script."""
    k, allc = (DA_K, DA_ALL) if dayahead else (INTRA_K, INTRA_ALL)
    return {"LR op eerdere K": (k, _lr, (0, len(k), len(k))),
            "Ridge (K + sensor + tijd)": (allc, _ridge, (0, len(allc), len(allc))),
            "Random Forest": (allc, _rf, (1, len(allc), len(allc))),
            "Gradient Boosting": (allc, _gb, (1, len(allc), len(allc)))}


# ---------- fase 3: weermodel (ERA5) als input, feature-ladder ----------
E6 = ["EFF_T", "EFF_T_KT"]
E6_EXTRA = E6 + ["CLOUD", "PRECIP", "RH"]


def _ridge0():
    """Ridge zonder intercept (geen centreren, wel schalen), voor e6 met collineaire inputs."""
    return make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False))


LADDER_FASE3 = {
    "LR GHI (ERA5)": (["GHI_W"], _lr, (0, 1, 1)),
    "LR POA (straling op het paneel)": (["POA"], _lr, (0, 1, 1)),
    "LR EFF_T (POA + reflectie, spectrum, stof, temp.)": (["EFF_T"], _lr, (0, 1, 1)),
    # e5 (uit V4, BESLISSINGEN #18): LR met intercept op EFF_T, KT_POA, CS_POA en de zonnestand
    "e5: EFF_T, KT_POA, CS_POA, COSZ (LR met intercept)": (["EFF_T", "KT_POA", "CS_POA", "COSZ"], _lr, (0, 4, 4)),
    "e6: EFF_T x (a + b KT_POA), zonder intercept": (E6, lambda: LinearRegression(fit_intercept=False), (0, 2, 2)),
    "Ridge e6 (zonder intercept)": (E6, _ridge0, (0, 2, 2)),
    "Ridge e6 + bewolking, regen, vochtigheid": (E6_EXTRA, _ridge, (0, 5, 5)),
    "Random Forest (e6 + extra)": (E6_EXTRA, _rf, (1, 5, 5)),
    "Gradient Boosting (e6 + extra)": (E6_EXTRA, _gb, (1, 5, 5)),
}

BASELINES_FASE2 = {"Baseline: gemiddelde dag": (0, 0, 1), "Baseline: persistence": (0, 1, 1),
                   "Smart persistence (K blijft gelijk)": (0, 1, 1)}

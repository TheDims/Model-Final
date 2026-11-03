# -*- coding: utf-8 -*-
"""
Split op tijd, in hele dagen, nooit willekeurig (AGENTS.md §6)
==============================================================
  day_split(dates)      trainingsdagen (eerste TRAIN_FRACTION) en testdagen, uit de echte datums
  val_folds(df, n)      validatievouwen binnen de trainingsdagen, afgeleid uit n_train (BESLISSINGEN #17)
  add_day_index(df)     kolom DAY = dagnummer 1..34
"""
import pandas as pd

from .config import TRAIN_FRACTION, VAL_SHARE, VAL_FOLDS_SMALL, VAL_FOLDS_LARGE, LARGE_FROM


def day_split(dates):
    """Datums -> dict met de trainingsdagen (eerste TRAIN_FRACTION) en de testdagen, in hele dagen.
    De sleutels 'n_opzet'/'opzet' heten zo sinds run 1 (= trainingsdagen)."""
    days = sorted(pd.DatetimeIndex(dates).normalize().unique())
    n_train = round(TRAIN_FRACTION * len(days))
    return {"n_dagen": len(days), "n_opzet": n_train, "n_test": len(days) - n_train,
            "opzet": days[:n_train], "test": days[n_train:],
            "opzet_van_tot": (days[0].date(), days[n_train - 1].date()),
            "test_van_tot": (days[n_train].date(), days[-1].date()),
            "train_dagnrs": list(range(1, n_train + 1)), "test_dagnrs": list(range(n_train + 1, len(days) + 1))}


def fold_days(n_train):
    """[(laatste trainingsdag, [validatiedagen]), ...] (expanding). Validatie = de laatste round(0,4 x n_train)
    trainingsdagen, verdeeld in 2 (n_train < 15) of 3 aaneengesloten vouwen; de rest gaat naar de laatste vouw.
    Run 1 (10): 1-6 -> 7-8, 1-8 -> 9-10. Run 2 (24): 1-14 -> 15-17, 1-17 -> 18-20, 1-20 -> 21-24."""
    n_val = round(VAL_SHARE * n_train)
    k = VAL_FOLDS_SMALL if n_train < LARGE_FROM else VAL_FOLDS_LARGE
    size = n_val // k
    start, out = n_train - n_val + 1, []
    for i in range(k):
        stop = n_train if i == k - 1 else start + size - 1
        out.append((start - 1, list(range(start, stop + 1))))
        start = stop + 1
    return out


def val_folds(df, n_train):
    """Validatievouwen als (naam, train-tabel, validatie-tabel)."""
    return [(f"train 1-{last} -> val {val[0]}-{val[-1]}", df[df.DAY <= last], df[df.DAY.isin(val)])
            for last, val in fold_days(n_train)]


def add_day_index(df, start):
    """Kolom DAY = dagnummer 1..34 (dag 1 = startdatum)."""
    return df.assign(DAY=(df.DATE_TIME.dt.normalize() - pd.Timestamp(start).normalize()).dt.days + 1)

# pyenmeval/model_selection.py

import pandas as pd
import numpy as np


# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
def summarize_models(df):

    summary = df.groupby(
        ["feature_class", "regularization"]
    ).agg({
        "auc_train": "mean",
        "auc_test": "mean",
        "auc_diff": "mean",
        "omission_rate": "mean",
        "aicc": "mean"
    }).reset_index()

    summary = summary.rename(columns={
        "auc_train": "auc_train_mean",
        "auc_test": "auc_test_mean",
        "auc_diff": "auc_diff_mean",
        "omission_rate": "omission_rate_mean",
        "aicc": "aicc_mean"
    })

    return summary


# ---------------------------------------------------
# DELTA AICc + WEIGHTS
# ---------------------------------------------------
def compute_delta_aicc(df):

    df = df.copy()

    # eliminar filas sin AICc válido
    df = df.dropna(subset=["aicc_mean"])

    if df.empty:
        raise ValueError("No valid AICc values to compute delta AICc.")

    min_aicc = df["aicc_mean"].min()

    df["delta_aicc"] = df["aicc_mean"] - min_aicc

    # pesos de Akaike
    weights = np.exp(-0.5 * df["delta_aicc"])

    if weights.sum() == 0:
        df["w_aicc"] = np.nan
    else:
        df["w_aicc"] = weights / weights.sum()

    return df


# ---------------------------------------------------
# RANKING
# ---------------------------------------------------
def rank_models(df):

    df = df.sort_values("delta_aicc").reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ---------------------------------------------------
# BEST MODEL
# ---------------------------------------------------
def select_best_model(df):
    """
    Selecciona el mejor modelo basado en delta AICc (mínimo).
    """

    if df.empty:
        raise ValueError("Model summary is empty.")

    best_idx = df["delta_aicc"].idxmin()
    best = df.loc[best_idx]

    return best
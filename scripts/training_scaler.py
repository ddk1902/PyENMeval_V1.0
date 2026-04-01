# =====================================
# ENTRENAMIENTO MAXNET (CON SCALER)
# =====================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from pyenmeval.maxnet_model import build_features
from sklearn.linear_model import LogisticRegression

# =====================================
# PATHS
# =====================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "Resultados"
MODELS_DIR = RESULTS_DIR / "modelos"

os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================
# DATASET 
# =====================================
df = pd.read_csv(DATA_DIR / "occurrences.csv")

# columnas esperadas:
# species | presence | var1 | var2 | ...

species_list = df["species"].unique()

print("Especies:", species_list)

# =====================================
# VARIABLES AMBIENTALES
# =====================================
feature_cols = [col for col in df.columns if col not in ["species", "presence"]]

# =====================================
# LOOP POR ESPECIE
# =====================================
for sp in species_list:

    print(f"\nProcesando: {sp}")

    sp_name = sp.replace(" ", "_")

    df_sp = df[df["species"] == sp].copy()

    X = df_sp[feature_cols].values
    y = df_sp["presence"].values

    # =========================
    # 1. ESCALADO (CRÍTICO)
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================
    # 2. FEATURES (MAXENT)
    # =========================
    feature_class = "lq"   # puedes ajustar luego (l, lq, lqh)

    X_feat = build_features(X_scaled, feature_class)

    # =========================
    # 3. MODELO (MAXENT ≈ Logistic con regularización)
    # =========================
    modelo = LogisticRegression(
        penalty="l2",
        C=1.0,              # equivalente a RM
        solver="lbfgs",
        max_iter=1000
    )

    modelo.fit(X_feat, y)

    # =========================
    # 4. WRAPPER (para compatibilidad con el pipeline)
    # =========================
    class MaxNetWrapper:
        def __init__(self, model, feature_class):
            self.model = model
            self.feature_class = feature_class

    modelo_final = MaxNetWrapper(modelo, feature_class)

    # =========================
    # 5. GUARDAR MODELO + SCALER
    # =========================
    model_path = MODELS_DIR / f"{sp_name}.joblib"
    scaler_path = MODELS_DIR / f"{sp_name}_scaler.joblib"

    joblib.dump(modelo_final, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✔ Modelo guardado: {model_path}")
    print(f"✔ Scaler guardado: {scaler_path}")

print("\n✔ ENTRENAMIENTO COMPLETADO")
import numpy as np
import pandas as pd
from pyenmeval.enmevaluate import ENMevaluate

# ----------------------------
# 1. Generar datos ficticios
# ----------------------------

# Ocurrencias: 20 puntos aleatorios
np.random.seed(42)
occ_df = pd.DataFrame({
    "x": np.random.uniform(0, 100, 20),
    "y": np.random.uniform(0, 100, 20)
})

# Background: 100 puntos aleatorios
bg_df = pd.DataFrame({
    "x": np.random.uniform(0, 100, 100),
    "y": np.random.uniform(0, 100, 100)
})

# Variables ambientales: 2 variables, valores aleatorios para cada punto (occ + bg)
env_values = np.random.rand(len(occ_df) + len(bg_df), 2)

# ----------------------------
# 2. Instanciar ENMevaluate
# ----------------------------
enmeval = ENMevaluate(
    occ_df=occ_df,
    env_values=env_values,
    bg_df=bg_df,
    k=5
)

# ----------------------------
# 3. Ejecutar k-fold
# ----------------------------
enmeval.run_kfold()

# ----------------

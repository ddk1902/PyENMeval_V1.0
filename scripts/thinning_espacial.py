# ===========================================
# BALANCEO + THINNING ESPACIAL (PYTHON)
# ===========================================

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import random
import os

# =====================================
# CONFIG
# =====================================
BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "data/processed_data/csv/Dataset_Triatominos_2__.csv"
OUTPUT_DIR = BASE_DIR / "resultados_tesis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Distancias en km
distancias_km = [1, 3, 5]

# =====================================
# 1. CARGA DE DATOS
# =====================================
df = pd.read_csv(INPUT_CSV)

df = df[["species", "lon", "lat", "year"]].dropna()

df = df[~df["species"].isin(["PREDADOR", "SD"])]

# =====================================
# 2. REGIÓN
# =====================================
df["REGION"] = np.where(df["lon"] < -58.3, "Occidental", "Oriental")

# =====================================
# 3. BALANCEO REGIONAL
# =====================================
n_min = df["REGION"].value_counts().min()

df_bal = (
    df.groupby("REGION")
    .apply(lambda x: x.sample(n_min, random_state=42))
    .reset_index(drop=True)
)

print("\nBALANCEO COMPLETADO")
print(df_bal["REGION"].value_counts())

# =====================================
# FUNCIÓN THINNING
# =====================================
def thinning_geografico(df, distancia_km):

    puntos = df[["lon", "lat"]].values.tolist()
    seleccionados = []

    for p in puntos:
        if not seleccionados:
            seleccionados.append(p)
            continue

        distancias = [
            geodesic((p[1], p[0]), (q[1], q[0])).km
            for q in seleccionados
        ]

        if all(d >= distancia_km for d in distancias):
            seleccionados.append(p)

    return pd.DataFrame(seleccionados, columns=["lon", "lat"])

# =====================================
# LOOP PRINCIPAL
# =====================================
for dist_km in distancias_km:

    print("\n====================================")
    print(f"THINNING {dist_km} km")
    print("====================================")

    resultados = []

    # grupo especie + región
    grupos = df_bal.groupby(["species", "REGION"])

    for (sp, reg), grupo in grupos:

        if len(grupo) <= 1:
            resultados.append(grupo)
            continue

        mejor = None
        max_n = 0

        # replicar lógica de spThin (reps=100)
        for _ in range(50):  # puedes subir a 100 si quieres
            shuffled = grupo.sample(frac=1)

            thinned = thinning_geografico(shuffled, dist_km)

            if len(thinned) > max_n:
                max_n = len(thinned)
                mejor = thinned

        mejor["species"] = sp
        mejor["REGION"] = reg

        resultados.append(mejor)

    df_final = pd.concat(resultados, ignore_index=True)

    # =====================================
    # MÉTRICAS
    # =====================================
    n_before = len(df_bal)
    n_after = len(df_final)

    print(f"Antes: {n_before}")
    print(f"Después: {n_after}")
    print(f"Eliminados: {n_before - n_after}")

    # Distancia promedio
    if len(df_final) > 1:
        coords = df_final[["lat", "lon"]].values

        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = geodesic(coords[i], coords[j]).km
                dists.append(d)

        print(f"Distancia media: {np.mean(dists):.2f} km")

    # =====================================
    # GUARDAR CSV
    # =====================================
    out_csv = OUTPUT_DIR / f"ocurrencias_thinned_{dist_km}km.csv"
    df_final.to_csv(out_csv, index=False)

    print(f"Guardado: {out_csv}")
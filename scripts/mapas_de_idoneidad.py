# =====================================
# MAPAS DE IDONEIDAD + IMPORTANCIA VARIABLES
# =====================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import joblib
from scipy.special import logsumexp

from pyenmeval.maxnet_model import build_features


# =====================================
# FUNCIÓN: IMPORTANCIA DE VARIABLES
# =====================================

def permutation_importance(modelo, X_valid):

    X_feat = build_features(X_valid, modelo.feature_class)

    base_logits = modelo.model.decision_function(X_feat)
    base_pred = 1 - np.exp(-np.exp(base_logits))

    importances = []

    for i in range(X_valid.shape[1]):

        X_perm = X_valid.copy()

        np.random.shuffle(X_perm[:, i])

        X_perm_feat = build_features(X_perm, modelo.feature_class)

        logits_perm = modelo.model.decision_function(X_perm_feat)

        pred_perm = 1 - np.exp(-np.exp(logits_perm))

        importance = np.mean(np.abs(base_pred - pred_perm))

        importances.append(importance)

    importances = np.array(importances)

    importances = 100 * importances / importances.sum()

    return importances
def response_curves(modelo, X_valid, var_names):

    import numpy as np
    import matplotlib.pyplot as plt

    # valores medios de todas las variables
    means = np.nanmean(X_valid, axis=0)

    n_vars = X_valid.shape[1]

    for i in range(n_vars):

        var_range = np.linspace(
            np.nanmin(X_valid[:, i]),
            np.nanmax(X_valid[:, i]),
            100
        )

        X_curve = np.tile(means, (100,1))

        X_curve[:, i] = var_range

        X_feat = build_features(X_curve, modelo.feature_class)

        logits = modelo.model.decision_function(X_feat)

        y_pred = 1 - np.exp(-np.exp(logits))

        plt.figure(figsize=(6,4))

        plt.plot(var_range, y_pred)

        plt.xlabel(var_names[i])

        plt.ylabel("Probabilidad de presencia")

        plt.title(f"Curva de respuesta - {var_names[i]}")

        plt.grid(True)

        plt.show()

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RASTER_DIR = DATA_DIR / "rasters_ascii"

RESULTS_FILE = BASE_DIR / "Resultados" / "tesis_resultados_python.csv"
MODELS_DIR = BASE_DIR / "Resultados" / "modelos"

SHAPE_PATH = DATA_DIR / "shapefiles" / "paraguay.shp"

OUTPUT_DIR = BASE_DIR / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# RESULTADOS
# =====================================

df = pd.read_csv(RESULTS_FILE)

best_per_species = (
    df.sort_values("aicc_mean")
      .groupby("species")
      .first()
      .reset_index()
)

robust_models = best_per_species[
    (best_per_species["auc_test_mean"] >= 0.70) &
    (best_per_species["auc_diff_mean"] <= 0.10)
]

species_list = robust_models["species"].unique()

print("Especies robustas:", species_list)


# =====================================
# SHAPEFILE
# =====================================

shape = gpd.read_file(SHAPE_PATH)


# =====================================
# STACK RASTERS
# =====================================

asc_files = sorted([f for f in os.listdir(RASTER_DIR) if f.endswith(".asc")])
var_names = [Path(f).stem for f in asc_files]

rasters = [rasterio.open(RASTER_DIR / f) for f in asc_files]

arrays = [r.read(1).astype("float32") for r in rasters]

stack = np.stack(arrays, axis=-1)

# eliminar nodata
stack[stack < -9000] = np.nan

meta = rasters[0].meta.copy()

n_rows, n_cols, n_vars = stack.shape

print("Variables:", n_vars)

X = stack.reshape(-1, n_vars)


# =====================================
# RASTER STATS
# =====================================

print("\n===== RASTER STATS =====")

for i in range(X.shape[1]):

    col = X[:, i]

    print(
        f"Variable {i}",
        "min:", np.nanmin(col),
        "max:", np.nanmax(col),
        "std:", np.nanstd(col)
    )


# =====================================
# MÁSCARA ESPACIAL
# =====================================

masked, _ = mask(rasters[0], shape.geometry, crop=False)

valid_mask = masked[0] != rasters[0].nodata

valid_mask = valid_mask.flatten()


# =====================================
# LOOP ESPECIES
# =====================================

for sp in tqdm(species_list):

    print("\nProcesando:", sp)

    sp_name = sp.replace(" ", "_")

    model_path = MODELS_DIR / f"{sp_name}.joblib"

    if not model_path.exists():

        print("Modelo no encontrado:", model_path)

        continue

    modelo = joblib.load(model_path)


    # =====================================
    # FEATURES
    # =====================================

    X_valid = X[valid_mask]

    # eliminar filas con NaN antes de features
    mask_env = ~np.isnan(X_valid).any(axis=1)

    X_valid = X_valid[mask_env]

    X_feat = build_features(X_valid, modelo.feature_class)


    # =====================================
    # NORMALIZACIÓN MAXENT
    # =====================================

    logits_all = modelo.model.decision_function(X_feat)

    c = np.percentile(logits_all, 75)


    # =====================================
    # PREDICCIÓN
    # =====================================

    logits = modelo.model.decision_function(X_feat)

    print("\nLogits stats:")

    print("min:", np.min(logits))
    print("max:", np.max(logits))
    print("std:", np.std(logits))


    logits_corr = logits - c

    y_pred = 1 - np.exp(-np.exp(logits_corr))


    # =====================================
    # RECONSTRUIR RASTER
    # =====================================

    valid_pixels = np.where(valid_mask)[0]

    valid_pixels = valid_pixels[mask_env]

    print("Predicciones:", len(y_pred))
    print("Pixeles válidos:", len(valid_pixels))

    full_pred = np.full(X.shape[0], np.nan)

    full_pred[valid_pixels] = y_pred

    mapa = full_pred.reshape(n_rows, n_cols)


    # =====================================
    # DEBUG MAPA
    # =====================================

    print("Min:", np.nanmin(mapa))
    print("Max:", np.nanmax(mapa))
    print("Mean:", np.nanmean(mapa))
    print("STD:", np.nanstd(mapa))


    # =====================================
    # GUARDAR RASTER
    # =====================================

    meta.update(driver="GTiff", count=1, dtype="float32")

    out_raster = OUTPUT_DIR / f"{sp_name}_idoneidad.tif"

    with rasterio.open(out_raster, "w", **meta) as dst:

        dst.write(mapa.astype("float32"), 1)


    # =====================================
    # CLASIFICACIÓN
    # =====================================

    clas = np.full_like(mapa, np.nan)

    clas[(mapa >= 0) & (mapa < 0.2)] = 1
    clas[(mapa >= 0.2) & (mapa < 0.4)] = 2
    clas[(mapa >= 0.4) & (mapa < 0.6)] = 3
    clas[(mapa >= 0.6) & (mapa < 0.8)] = 4
    clas[(mapa >= 0.8)] = 5


    # =====================================
    # VISUALIZACIÓN MAPA
    # =====================================

    colors = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]

    plt.figure(figsize=(8,6))

    plt.imshow(
        clas,
        cmap=plt.matplotlib.colors.ListedColormap(colors),
        interpolation="nearest"
    )

    plt.title(f"Idoneidad ambiental - {sp}")

    plt.axis("off")

    legend_elements = [
        Patch(facecolor=colors[0], label="0.0–0.2   Nula"),
        Patch(facecolor=colors[1], label="0.2–0.4   Baja"),
        Patch(facecolor=colors[2], label="0.4–0.6   Media"),
        Patch(facecolor=colors[3], label="0.6–0.8   Alta"),
        Patch(facecolor=colors[4], label="0.8–1.0   Muy alta"),
    ]

    plt.legend(handles=legend_elements, title="Probabilidad", loc="lower left")

    plt.show()


    # =====================================
    # IMPORTANCIA VARIABLES
    # =====================================

    importances = permutation_importance(modelo, X_valid)

    print("\nImportancia de variables (%)")

    for name, imp in zip(var_names, importances):
      print(f"{name}: {imp:.2f}%")

    plt.figure(figsize=(6,4))

    plt.bar(var_names, importances)

    plt.xticks(rotation=45)
    plt.xlabel("Variable ambiental")

    plt.ylabel("Importancia (%)")

    plt.title(f"Importancia de variables - {sp}")

    plt.show()
    # =====================================
    # CURVAS DE RESPUESTA
    # =====================================

    response_curves(modelo, X_valid, var_names)

print("\nMAPAS GENERADOS CORRECTAMENTE")
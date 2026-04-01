import pandas as pd
from pathlib import Path
from pyenmeval.spatial_utils import prepare_occurrences
from pyenmeval.utils import generate_background
from pyenmeval import ENMevaluate

import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import warnings
import joblib

warnings.filterwarnings("ignore")

# =====================================
# CONFIGURACIÓN DEL MODELO
# =====================================

def configurar_modelo(n_occs):

    if n_occs < 20:
        return "random", 3, ["L"], [1,2,3]

    elif n_occs < 30:
        return "random", 3, ["L", "LQ"], [2, 2.5, 3]

    else:
        return "block", 5, ["L","LQ","H","LQH"], [0.1,0.5,1,2]


# =====================================
# PROCESAR ESPECIE
# =====================================

def procesar_especie(sp, occ, bg_cache, ascii_dir, MODELS_DIR):

    print(f"\nProcesando especie: {sp}")

    occ_sp = occ[occ["species"] == sp]

    if len(occ_sp) < 5:
        print("Muy pocos datos, se omite")
        return None

    occ_sp = prepare_occurrences(occ_sp)
    bg_sp = bg_cache[sp]

    partition, k, feature_classes, regularization_values = configurar_modelo(len(occ_sp))

    print(f"Config → {partition}, k={k}, FC={feature_classes}, RM={regularization_values}")

    model = ENMevaluate(
        occ_df=occ_sp,
        bg_df=bg_sp,
        predictors_dir=ascii_dir,
        k=k,
        partition_method=partition,
        seed=42
    )

    ranking, best_model = model.run_tuning(
        feature_classes=feature_classes,
        regularization_values=regularization_values
    )

    if ranking is None or ranking.empty:
        print("Ranking vacío")
        return None

    best_fc = best_model["feature_class"]
    best_rm = best_model["regularization"]

    print(f"Mejor modelo → FC={best_fc}, RM={best_rm}")

    # =====================================
    # ENTRENAMIENTO FINAL
    # =====================================

    final_model = ENMevaluate(
        occ_df=occ_sp,
        bg_df=bg_sp,
        predictors_dir=ascii_dir,
        k=k,
        partition_method=partition,
        seed=42
    )

    _, best_final = final_model.run_tuning(
        feature_classes=[best_fc],
        regularization_values=[best_rm]
    )

    modelo_entrenado = best_final.get("model", None)

    if modelo_entrenado is None:
        print("No se pudo recuperar modelo")
        return None

    sp_name = sp.replace(" ", "_")

    model_path = MODELS_DIR / f"{sp_name}.joblib"
    scaler_path = MODELS_DIR / f"{sp_name}_scaler.joblib"

    joblib.dump(modelo_entrenado, model_path)

    if hasattr(modelo_entrenado, "scaler"):
        joblib.dump(modelo_entrenado.scaler, scaler_path)
    else:
        print("Scaler no encontrado en el modelo")

    print(f"Modelo guardado: {model_path}")

    ranking = ranking.copy()
    ranking["species"] = sp
    ranking.reset_index(drop=True, inplace=True)

    return ranking


# =====================================
# MAIN
# =====================================

def main():

    BASE_DIR = Path(__file__).resolve().parent

    input_dir = BASE_DIR / "data/rasters"
    reproj_dir = BASE_DIR / "data/rasters_reproj"
    aligned_dir = BASE_DIR / "data/rasters_aligned"
    ascii_dir = BASE_DIR / "data/rasters_ascii"

    for d in [reproj_dir, aligned_dir, ascii_dir]:
        os.makedirs(d, exist_ok=True)

    # =====================================
    # REPROYECCIÓN
    # =====================================

    if len(os.listdir(reproj_dir)) == 0:

        print("Reproyectando rasters...")

        for f in os.listdir(input_dir):

            if f.endswith(".tif"):

                with rasterio.open(input_dir / f) as src:

                    transform, width, height = calculate_default_transform(
                        src.crs,
                        "EPSG:4326",
                        src.width,
                        src.height,
                        *src.bounds
                    )

                    profile = src.profile.copy()
                    profile.update(
                        crs="EPSG:4326",
                        transform=transform,
                        width=width,
                        height=height
                    )

                    with rasterio.open(reproj_dir / f, "w", **profile) as dst:

                        for i in range(1, src.count + 1):

                            reproject(
                                rasterio.band(src, i),
                                rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs="EPSG:4326",
                                resampling=Resampling.bilinear
                            )

    print("Reproyección lista")

    # =====================================
    # ALINEACIÓN
    # =====================================

    ref_file = list(os.listdir(reproj_dir))[0]

    with rasterio.open(reproj_dir / ref_file) as ref:

        for f in os.listdir(reproj_dir):

            if f.endswith(".tif"):

                with rasterio.open(reproj_dir / f) as src:

                    with rasterio.open(aligned_dir / f, "w", **ref.profile) as dst:

                        for i in range(1, src.count + 1):

                            reproject(
                                rasterio.band(src, i),
                                rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref.transform,
                                dst_crs=ref.crs,
                                resampling=Resampling.bilinear
                            )

    print("Alineación lista")

    # =====================================
    # ASCII
    # =====================================

    if len(os.listdir(ascii_dir)) == 0:

        print("Convirtiendo rasters a ASCII...")

        for f in os.listdir(aligned_dir):

            if f.endswith(".tif"):

                with rasterio.open(aligned_dir / f) as src:

                    data = src.read(1).astype("float32")

                    profile = src.profile.copy()
                    profile.update(driver="AAIGrid", count=1)

                    out = ascii_dir / f.replace(".tif", ".asc")

                    with rasterio.open(out, "w", **profile) as dst:
                        dst.write(data, 1)

    print("ASCII listo")

    # =====================================
    # OCURRENCIAS
    # =====================================

    occ = pd.read_csv(BASE_DIR / "data/occurrences.csv")

    occ = occ.rename(columns={
        "lon": "longitude",
        "lat": "latitude"
    })

    occ = prepare_occurrences(occ)

    species_list = occ["species"].unique()

    # =====================================
    # BACKGROUND
    # =====================================

    bg_cache = {}

    for sp in species_list:

        occ_sp = occ[occ["species"] == sp]

        if len(occ_sp) >= 5:

            bg_cache[sp] = generate_background(
                occ_sp,
                n=5000,
                seed=42,
                buffer=5.0
            )

    print("Background generado")

    # =====================================
    # MODELOS
    # =====================================

    MODELS_DIR = BASE_DIR / "Resultados/modelos"
    os.makedirs(MODELS_DIR, exist_ok=True)

    resultados = []

    for sp in species_list:

        if sp in bg_cache:

            res = procesar_especie(
                sp,
                occ,
                bg_cache,
                ascii_dir,
                MODELS_DIR
            )

            if res is not None:
                resultados.append(res)

    if len(resultados) == 0:
        print("No hay resultados")
        return

    df_final = pd.concat(resultados, ignore_index=True)

    out = BASE_DIR / "Resultados/tesis_resultados_python.csv"

    df_final.to_csv(out, index=False)

    print("\nProceso finalizado")
    print("Resultados:", out)


if __name__ == "__main__":
    main()
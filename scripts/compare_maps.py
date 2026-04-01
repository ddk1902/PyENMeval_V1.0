import rasterio
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling

BASE_DIR = Path("data/maps")

PY_DIR = BASE_DIR / "python"
R_DIR = BASE_DIR / "R"

results = []

for py_map in PY_DIR.glob("*.tif"):

    species = py_map.stem
    r_map = R_DIR / f"{species}.tif"

    if not r_map.exists():
        print(f"No existe mapa R para {species}")
        continue

    print(f"\nComparando {species}")

    with rasterio.open(py_map) as src_py:
        py = src_py.read(1).astype(float)
        py_meta = src_py.meta

    with rasterio.open(r_map) as src_r:

        r_resampled = np.empty(py.shape, dtype="float32")

        reproject(
            source=rasterio.band(src_r, 1),
            destination=r_resampled,
            src_transform=src_r.transform,
            src_crs=src_r.crs,
            dst_transform=py_meta["transform"],
            dst_crs=py_meta["crs"],
            resampling=Resampling.bilinear
        )

    py[py < -9000] = np.nan
    r_resampled[r_resampled < -9000] = np.nan

    mask = (~np.isnan(py)) & (~np.isnan(r_resampled))

    x = py[mask]
    y = r_resampled[mask]

    corr, _ = pearsonr(x, y)
    rmse = np.sqrt(np.mean((x - y) ** 2))

    print("Correlación:", round(corr,3))
    print("RMSE:", round(rmse,3))

    results.append({
        "species": species,
        "correlation": corr,
        "rmse": rmse
    })

    plt.figure(figsize=(6,6))

    plt.scatter(x, y, s=1, alpha=0.2)

    plt.xlabel("Python model")
    plt.ylabel("MaxEnt R")

    plt.title(f"{species}\nCorrelation = {corr:.2f}")

    plt.plot([0,1],[0,1], linestyle="--")

    plt.grid(True)

    plt.show()


df = pd.DataFrame(results)

print("\nResumen:")
print(df)

df.to_csv("model_comparison.csv", index=False)
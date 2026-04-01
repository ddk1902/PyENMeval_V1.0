import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pyenmeval.enmevaluate import ENMevaluate
from pyenmeval.run_kfold import run_kfold
from pyenmeval.run_jackknife import run_jackknife
from pyenmeval.utils import generate_background

# ----------------------------
# 1. DATOS SIMULADOS DE PRESENCIA
# ----------------------------
np.random.seed(42)
n_points = 50
lon = np.random.uniform(-60, -58, n_points)
lat = np.random.uniform(-25, -23, n_points)
species = ['Triatoma infestans']*n_points
year = np.random.randint(2008, 2014, n_points)

df_pres = pd.DataFrame({'species': species, 'lon': lon, 'lat': lat, 'year': year})
gdf_pres = gpd.GeoDataFrame(df_pres, geometry=gpd.points_from_xy(df_pres.lon, df_pres.lat))
gdf_pres.set_crs(epsg=4326, inplace=True)

# ----------------------------
# 2. CREAR RASTER SIMULADO
# ----------------------------
width = height = 100
cell_size = 0.02
transform = from_origin(-60, -23, cell_size, cell_size)
raster_path = "examples/env_raster.tif"

with rasterio.open(
    raster_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=rasterio.float32,
    crs='EPSG:4326',
    transform=transform
) as dst:
    data = np.random.rand(height, width).astype('float32')
    dst.write(data, 1)

# ----------------------------
# 3. EXTRAER VALORES DE RASTER
# ----------------------------
env_values = []
with rasterio.open(raster_path) as src:
    for x, y in zip(df_pres.lon, df_pres.lat):
        row, col = src.index(x, y)
        env_values.append(src.read(1)[row, col])

env_values = np.array(env_values)

# ----------------------------
# 4. GENERAR BACKGROUND
# ----------------------------
bg_df = generate_background(df_pres, n=200, buffer=0.1)

# ----------------------------
# 5. EJECUTAR ENMevaluate
# ----------------------------
enme = ENMevaluate(
    occ_df=gdf_pres,
    env_values=env_values,
    bg_df=bg_df,
    k=5,
    thin_distance=0.05
)

# K-Fold
print("=== RUN K-FOLD ===")
kfold_results = enme.run_kfold()
print(kfold_results)

# Jackknife
print("=== RUN JACKKNIFE ===")
jack_results = enme.run_jackknife()
print(jack_results)

# Block (simulamos dos bloques)
block_labels = np.array([0 if i < n_points//2 else 1 for i in range(n_points)])
print("=== RUN BLOCK ===")
block_results = enme.run_block(block_labels)
print(block_results)

# ----------------------------
# 6. PREDICCIÓN A RASTER
# ----------------------------
output_raster = "examples/prediction_raster.tif"
enme.predict_to_raster([raster_path], output_raster)
print(f"Predicción raster guardada en: {output_raster}")

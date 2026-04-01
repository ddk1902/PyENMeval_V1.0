import geopandas as gpd
import pandas as pd
import rasterio

# ===============================
# CRS CONSTANT
# ===============================
TARGET_CRS = "EPSG:32721"  # UTM Zona 21 Sur

# ===============================
# CREATE GEODATAFRAME
# ===============================
def to_geodataframe(df, lon_col="longitude", lat_col="latitude", crs="EPSG:4326"):
    """
    Convierte DataFrame a GeoDataFrame.
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"Columns {lon_col}, {lat_col} not found in DataFrame")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs
    )

    return gdf


# ===============================
# PROJECT TO UTM
# ===============================
def project_to_utm(gdf, target_crs=TARGET_CRS):
    """
    Proyecta a UTM (EPSG:32721).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined")

    gdf_utm = gdf.to_crs(target_crs)

    # agregar coordenadas métricas
    gdf_utm["x"] = gdf_utm.geometry.x
    gdf_utm["y"] = gdf_utm.geometry.y

    return gdf_utm


# ===============================
# ALIGN TO RASTER CRS
# ===============================
def align_to_raster_crs(gdf, raster_path):
    """
    Reproyecta puntos al CRS del raster.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    return gdf.to_crs(raster_crs)


# ===============================
# FULL PIPELINE
# ===============================
def prepare_occurrences(df, lon_col="longitude", lat_col="latitude"):
    """
    Pipeline completo:
    1. DataFrame → GeoDataFrame
    2. Proyección a UTM
    """
    gdf = to_geodataframe(df, lon_col, lat_col)
    gdf = project_to_utm(gdf)

    return gdf


# ===============================
# VALIDATION
# ===============================
def check_same_crs(gdf, raster_path):
    """
    Verifica que puntos y raster tengan el mismo CRS.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if gdf.crs != raster_crs:
        print("WARNING: CRS mismatch → reprojection needed")
    else:
        print("CRS OK")
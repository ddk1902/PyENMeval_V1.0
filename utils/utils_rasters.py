import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_origin

def generar_mapa_promedio(raster_files, output_path):
    """
    Calcula el promedio de varios rasters (misma resolución y extensión).
    """
    datasets = [rasterio.open(r) for r in raster_files]
    meta = datasets[0].meta.copy()

    stack = np.stack([ds.read(1, out_shape=(ds.count, ds.height, ds.width)).squeeze() for ds in datasets])
    stack_mean = np.nanmean(stack, axis=0)

    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(stack_mean.astype(rasterio.float32), 1)

    for ds in datasets:
        ds.close()

    print(f"✅ Mapa promedio guardado en: {output_path}")
    return output_path

import os
import numpy as np
import rasterio


def sample_raster(raster_dir, coords):

    values = []

    files = sorted([
        f for f in os.listdir(raster_dir)
        if f.endswith(".asc")
    ])

    for f in files:
        with rasterio.open(os.path.join(raster_dir, f)) as src:
            vals = [v[0] for v in src.sample(coords)]
            values.append(vals)

    return np.array(values).T
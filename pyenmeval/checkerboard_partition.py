# pyenmeval/checkerboard_partition.py

import numpy as np


def checkerboard_partition(occ_df, cell_size=1.0):
    """
    Genera particiones espaciales tipo checkerboard.

    Parameters
    ----------
    occ_df : DataFrame
        Debe contener columnas longitude y latitude

    cell_size : float
        Tamaño de celda de la grilla (en unidades del CRS)

    Returns
    -------
    numpy array
        vector con fold asignado para cada punto
    """

    lon = occ_df["longitude"].values
    lat = occ_df["latitude"].values

    lon_index = np.floor(lon / cell_size)
    lat_index = np.floor(lat / cell_size)

    folds = (lon_index + lat_index) % 2

    return folds.astype(int)
def checkerboard_partition_2(occ_df, cell_size1=1.0, cell_size2=2.0):

    folds1 = checkerboard_partition(occ_df, cell_size1)
    folds2 = checkerboard_partition(occ_df, cell_size2)

    folds = (folds1 + folds2) % 2

    return folds
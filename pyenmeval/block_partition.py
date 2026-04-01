# pyenmeval/block_partition.py

import numpy as np
import pandas as pd


def block_partition(occ_df):
    """
    Divide ocurrencias en 4 bloques espaciales.

    Parameters
    ----------
    occ_df : DataFrame
        Debe contener columnas: longitude, latitude

    Returns
    -------
    numpy array
        Vector con fold asignado para cada punto
    """

    longitude = occ_df["longitude"].values
    latitude = occ_df["latitude"].values

    longitude_median = np.median(longitude)
    latitude_median = np.median(latitude)

    folds = []

    for x, y in zip(longitude, latitude):

        if x <= longitude_median and y > latitude_median:
            fold = 0  # NW

        elif x > longitude_median and y > latitude_median:
            fold = 1  # NE

        elif x <= longitude_median and y <= latitude_median:
            fold = 2  # SW

        else:
            fold = 3  # SE

        folds.append(fold)

    return np.array(folds)
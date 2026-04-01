import numpy as np
import pandas as pd


def generate_background(
    occ_df,
    n=5000,
    seed=42,
    buffer=0.1,
    x_col="longitude",
    y_col="latitude"
):
    """
    Generate random background points within a buffered bounding box.

    Parameters
    ----------
    occ_df : DataFrame
        Occurrence data with lon/lat
    n : int
        Number of background points
    seed : int
        Random seed
    buffer : float
        Expansion factor of bounding box
    """

    if seed is not None:
        np.random.seed(seed)

    if x_col not in occ_df.columns or y_col not in occ_df.columns:
        raise ValueError("occ_df must contain 'longitude' and 'latitude' columns")

    xmin, xmax = occ_df[x_col].min(), occ_df[x_col].max()
    ymin, ymax = occ_df[y_col].min(), occ_df[y_col].max()

    # aplicar buffer
    dx = (xmax - xmin) * buffer
    dy = (ymax - ymin) * buffer

    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    # generar puntos aleatorios
    bg_x = np.random.uniform(xmin, xmax, n)
    bg_y = np.random.uniform(ymin, ymax, n)

    bg = pd.DataFrame({
        "longitude": bg_x,
        "latitude": bg_y
    })

    return bg
# pyenmeval/parameter_grid.py

import pandas as pd
import itertools


def generate_parameter_grid(feature_classes, regularization_values):
    """
    Genera todas las combinaciones FC × RM.

    Parameters
    ----------
    feature_classes : list
        Ejemplo: ["L", "LQ", "H", "LQH"]

    regularization_values : list
        Ejemplo: [0.5, 1, 2, 3]

    Returns
    -------
    DataFrame
        Tabla con todas las combinaciones
    """

    combinations = list(itertools.product(feature_classes, regularization_values))

    grid = pd.DataFrame(combinations, columns=["feature_class", "regularization"])

    grid["model_id"] = range(1, len(grid) + 1)

    return grid
# pyenmeval/null_models.py

import numpy as np
import pandas as pd


def generate_null_occurrences(bg_df, n_occ, replace=False):
    """
    Genera pseudo-presencias aleatorias desde background.

    Parameters
    ----------
    bg_df : DataFrame
    n_occ : int
    replace : bool

    Returns
    -------
    DataFrame
    """

    if n_occ > len(bg_df) and not replace:
        raise ValueError("n_occ mayor que background disponible")

    sample = bg_df.sample(n=n_occ, replace=replace)

    return sample.reset_index(drop=True)


def generate_null_datasets(bg_df, n_occ, n_reps=100, replace=False):
    """
    Genera múltiples datasets nulos.

    Returns
    -------
    list[DataFrame]
    """

    datasets = []

    for _ in range(n_reps):

        null_occ = generate_null_occurrences(bg_df, n_occ, replace)

        datasets.append(null_occ)

    return datasets


def evaluate_null_models(
    bg_df,
    predictors_dir,
    maxent_jar,
    feature_classes,
    regularization_values,
    n_occ,
    n_reps=100,
    partition_method="random"
):
    """
    Ejecuta modelos nulos completos usando PyENMeval.

    Returns
    -------
    list
        métricas (ej. AUCtest) de modelos nulos
    """

    from .enmevaluate import ENMevaluate

    null_metrics = []

    null_datasets = generate_null_datasets(bg_df, n_occ, n_reps)

    for i, null_occ in enumerate(null_datasets):

        print(f"[Null Model {i+1}/{n_reps}]")

        evaluator = ENMevaluate(
            occ_df=null_occ,
            bg_df=bg_df,
            predictors_dir=predictors_dir,
            maxent_jar=maxent_jar,
            partition_method=partition_method,
            cleanup_workspace=True
        )

        ranking, best_model = evaluator.run_tuning(
            feature_classes=feature_classes,
            regularization_values=regularization_values
        )

        # usamos AUCtest como referencia (igual que ENMeval)
        null_metrics.append(best_model["auc_test"])

    return null_metrics


def compute_p_value(real_value, null_values):
    """
    Calcula p-value empírico.

    Parameters
    ----------
    real_value : float
    null_values : list

    Returns
    -------
    float
    """

    null_values = np.array(null_values)

    p = np.mean(null_values >= real_value)

    return p
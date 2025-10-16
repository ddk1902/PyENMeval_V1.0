# pyenmeval/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def auc(y_true, y_pred):
    """
    Calcula el AUC (Area Under Curve) para los datos de prueba.
    Devuelve np.nan si solo hay una clase presente.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)

def omission_rate(y_true, y_pred, threshold=0.5):
    """
    Calcula la tasa de omisión:
    Proporción de puntos de presencia mal predichos como ausencia.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    if np.sum(y_true==1) == 0:
        return np.nan
    return np.mean((y_true == 1) & (y_pred_bin == 0))

def tss(y_true, y_pred, threshold=0.5):
    """
    True Skill Statistic: TSS = Sensibilidad + Especificidad - 1
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity + specificity - 1

def predicted_presence_sum(y_pred, threshold=0.5):
    """
    Suma de puntos predichos como presencia según threshold.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    return np.sum(y_pred_bin)

def accuracy(y_true, y_pred, threshold=0.5):
    """
    Exactitud de predicción.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    return np.mean(y_true == y_pred_bin)

def kappa(y_true, y_pred, threshold=0.5):
    """
    Cohen's Kappa para la predicción binaria.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0,1]).ravel()
    total = tp + tn + fp + fn
    po = (tp + tn) / total
    pe = ((tp + fp)*(tp + fn) + (fn + tn)*(fp + tn)) / total**2
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else np.nan



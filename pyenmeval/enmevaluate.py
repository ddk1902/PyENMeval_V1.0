# pyenmeval/enmevaluate.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from .metrics import auc, omission_rate, tss, accuracy, kappa, predicted_presence_sum

class ENMevaluate:
    def __init__(self, occ_df, env_values, bg_df=None, k=5):
        """
        occ_df: DataFrame de ocurrencias (x, y)
        env_values: numpy array n_points x n_vars
        bg_df: DataFrame de puntos de fondo o array de env_values para background
        k: número de folds para validación cruzada
        """
        self.occ_df = occ_df
        self.env_values = np.array(env_values)
        self.bg_df = np.array(bg_df) if bg_df is not None else None
        self.k = k
        self.results = None

    def run_kfold(self, threshold=0.5):
        """Ejecuta k-fold cross-validation y calcula todas las métricas."""
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        fold_results = []

        # Construir X (variables) y y (clases)
        X = self.env_values
        y = np.ones(len(self.occ_df))

        if self.bg_df is not None:
            if X.shape[1] != self.bg_df.shape[1]:
                raise ValueError("Número de variables en occ_df y bg_df debe ser igual")
            X = np.vstack([X, self.bg_df])
            y = np.concatenate([y, np.zeros(len(self.bg_df))])

        # Ejecutar k-fold
        for i, (train_index, test_index) in enumerate(kf.split(self.occ_df)):
            # Entrenamiento sobre fold_train + fondo completo
            train_X = X[train_index]
            train_y = y[train_index]
            test_X = X[test_index]
            test_y = y[test_index]

            if len(np.unique(train_y)) < 2:
                # Si no hay ambas clases en entrenamiento
                fold_results.append({
                    'fold': i+1,
                    'auc': np.nan,
                    'omission_rate': np.nan,
                    'tss': np.nan,
                    'accuracy': np.nan,
                    'kappa': np.nan,
                    'predicted_presence_sum': np.nan
                })
                continue

            # Modelo proxy: Logistic Regression
            model = LogisticRegression(solver='liblinear')
            model.fit(train_X, train_y)
            y_pred = model.predict_proba(test_X)[:, 1]

            # Calcular métricas
            fold_results.append({
                'fold': i+1,
                'auc': auc(test_y, y_pred),
                'omission_rate': omission_rate(test_y, y_pred, threshold),
                'tss': tss(test_y, y_pred, threshold),
                'accuracy': accuracy(test_y, y_pred, threshold),
                'kappa': kappa(test_y, y_pred, threshold),
                'predicted_presence_sum': predicted_presence_sum(y_pred, threshold)
            })

        self.results = pd.DataFrame(fold_results)
        return self.results

    def summary(self):
        """Devuelve resumen estadístico de todas las métricas por fold."""
        if self.results is None:
            raise ValueError("run_kfold() debe ejecutarse primero")
        return self.results.describe()


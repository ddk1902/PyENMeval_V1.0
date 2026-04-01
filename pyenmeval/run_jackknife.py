def run_jackknife(self):
    """
    Implementación de validación tipo Jackknife (Leave-One-Out),
    inspirada en ENMeval (R). Cada punto de presencia se deja fuera
    una vez para evaluar el modelo entrenado con los demás.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    if not hasattr(self, "occ_df") or self.occ_df is None:
        raise ValueError("Se requiere occ_df (ocurrencias) para ejecutar jackknife.")

    occ_n = len(self.occ_df)
    if occ_n < 2:
        print("Muy pocos puntos de ocurrencia para jackknife.")
        return pd.DataFrame()

    # Matrices de entrada
    X = np.array(self.env_values)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_pres = np.ones(occ_n, dtype=int)

    if self.bg_df is None:
        raise ValueError("Se requiere bg_df (background) para entrenamiento MaxEnt.")
    bg_array = np.array(self.bg_df)
    if bg_array.ndim == 1:
        bg_array = bg_array.reshape(-1, 1)

    X_all = np.vstack([X, bg_array])
    y_all = np.concatenate([y_pres, np.zeros(len(bg_array), dtype=int)])

    bg_start = occ_n
    bg_indices = np.arange(bg_start, bg_start + len(bg_array))

    fold_results = []
    self.models = []

    print(f"Ejecutando Jackknife con {occ_n} iteraciones...")

    # Loop leave-one-out
    for i in range(occ_n):
        test_idx = np.array([i])
        train_occ_idx = np.delete(np.arange(occ_n), i)
        train_idx = np.concatenate([train_occ_idx, bg_indices]).astype(int)

        try:
            # Entrenar modelo
            model = self._train_maxent(train_idx, X_all, y_all)
            self.models.append(model)

            # Predecir
            y_true = y_all[test_idx]
            y_score = self._predict_proba_on_indices(model, test_idx, X_all)

            # Métricas
            auc_val = np.nan
            omission = np.nan
            try:
                auc_val = roc_auc_score(y_true, y_score)
            except Exception:
                pass
            omission = np.mean((y_score < 0.5).astype(int))

            fold_results.append({
                "fold": i + 1,
                "auc": auc_val,
                "omission_rate": omission
            })
            print(f"Jackknife {i+1}/{occ_n}: AUC={auc_val}, omission={omission}")

        except Exception as e:
            print(f"Error en jackknife {i+1}: {e}")
            fold_results.append({
                "fold": i + 1,
                "auc": np.nan,
                "omission_rate": np.nan
            })

    self.results = pd.DataFrame(fold_results)

    # Selección del mejor modelo
    valid = self.results.dropna(subset=["auc"])
    if valid.empty:
        print(" No hay modelos válidos en jackknife (AUC no calculable).")
        self.model = None
        return self.results

    best_fold = int(valid.loc[valid["auc"].idxmax(), "fold"])
    self.model = self.models[best_fold - 1]
    print(f"Mejor modelo (jackknife): fold {best_fold} con AUC={valid['auc'].max()}")

    return self.results

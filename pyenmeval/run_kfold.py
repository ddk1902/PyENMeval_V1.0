import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from pyenmeval.config import set_seed

set_seed(42)
def run_block(self, min_points_per_block=2):
    """
    Spatial block cross-validation (ENMeval-style).
    Divide el espacio en 4 bloques (2x2).
    """

    import geopandas as gpd

    if not hasattr(self, 'occ_df') or self.occ_df is None:
        raise ValueError("Occurrence data (occ_df) is required.")

    occ = self.occ_df.copy()

    # -------------------------------
    # 1. Convertir a GeoDataFrame si necesario
    # -------------------------------
    if not isinstance(occ, gpd.GeoDataFrame):

        if 'longitude' not in occ.columns or 'latitude' not in occ.columns:
            raise ValueError("occ_df must have 'longitude' and 'latitude' columns.")

        occ = gpd.GeoDataFrame(
            occ,
            geometry=gpd.points_from_xy(occ.longitude, occ.latitude),
            crs="EPSG:4326"
        )

    # -------------------------------
    # 2. PROYECCIÓN A UTM (CRÍTICO)
    # -------------------------------
    occ = occ.to_crs("EPSG:32721")

    # -------------------------------
    # 3. Definir bloques espaciales
    # -------------------------------
    xmin, ymin, xmax, ymax = occ.total_bounds

    x_edges = np.linspace(xmin, xmax, 3)
    y_edges = np.linspace(ymin, ymax, 3)

    occ["block_x"] = np.digitize(occ.geometry.x, x_edges) - 1
    occ["block_y"] = np.digitize(occ.geometry.y, y_edges) - 1
    occ["block_id"] = occ["block_x"] + occ["block_y"] * 2

    # -------------------------------
    # 4. Filtrar bloques pequeños
    # -------------------------------
    block_counts = occ["block_id"].value_counts()
    valid_blocks = block_counts[block_counts >= min_points_per_block].index.tolist()

    if len(valid_blocks) == 0:
        print("No valid blocks → fallback to KFold")
        return self.run_kfold()

    fold_results = []
    self.models = []

    # -------------------------------
    # 5. Preparar matrices
    # -------------------------------
    X = np.array(self.env_values)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y_pres = np.ones(len(occ), dtype=int)

    if self.bg_df is None:
        raise ValueError("bg_df required.")

    bg_array = np.array(self.bg_df)
    if bg_array.ndim == 1:
        bg_array = bg_array.reshape(-1, 1)

    X_all = np.vstack([X, bg_array])
    y_all = np.concatenate([y_pres, np.zeros(len(bg_array), dtype=int)])

    occ_n = len(occ)
    bg_indices = np.arange(occ_n, occ_n + len(bg_array))

    # -------------------------------
    # 6. Cross-validation por bloques
    # -------------------------------
    for block_id in valid_blocks:

        test_occ_idx = occ.index[occ["block_id"] == block_id].to_numpy()
        train_occ_idx = occ.index[occ["block_id"] != block_id].to_numpy()

        if len(train_occ_idx) < 2:
            print(f"Block {block_id} skipped (too few points)")
            continue

        train_idx = np.concatenate([train_occ_idx, bg_indices]).astype(int)

        # Entrenamiento
        try:
            model = self._train_maxent(train_idx, X_all, y_all)
            self.models.append(model)
        except Exception as e:
            print(f"Error training block {block_id}: {e}")
            self.models.append(None)
            continue

        # Evaluación
        try:
            y_true = y_all[test_occ_idx]
            y_score = self._predict_proba_on_indices(model, test_occ_idx, X_all)

            try:
                auc_val = roc_auc_score(y_true, y_score)
            except:
                auc_val = np.nan

            omission = np.mean((y_score < 0.5).astype(int))

            fold_results.append({
                "fold": int(block_id),
                "auc": auc_val,
                "omission_rate": omission
            })

            print(f"Block {block_id}: AUC={auc_val}, omission={omission}")

        except Exception as e:
            print(f"Error evaluating block {block_id}: {e}")

    # -------------------------------
    # 7. Resultados
    # -------------------------------
    self.results = pd.DataFrame(fold_results)

    valid = self.results.dropna(subset=["auc"])

    if valid.empty:
        print("No valid blocks")
        self.model = None
        return self.results

    best_idx = valid["auc"].idxmax()
    best_fold = int(valid.loc[best_idx, "fold"])

    self.model = self.models[best_idx]

    print(f"Best block: {best_fold} (AUC={valid['auc'].max()})")

    return self.results
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from .metrics import auc, omission_rate, tss, accuracy, kappa, predicted_presence_sum
from .metrics_aicc import log_likelihood, compute_aicc
from .parameter_grid import generate_parameter_grid
from .block_partition import block_partition
from .checkerboard_partition import checkerboard_partition, checkerboard_partition_2
from .workspace_manager import WorkspaceManager
from .model_selection import summarize_models, compute_delta_aicc, rank_models, select_best_model
from .maxnet_model import MaxNetModel
from .raster_sampling import sample_raster

from pyenmeval.config import set_seed
from pyenmeval.paths import OUTPUTS_DIR
from pyenmeval.spatial_utils import prepare_occurrences


class ENMevaluate:

    def __init__(
        self,
        occ_df,
        bg_df,
        predictors_dir,
        k=5,
        partition_method="random",
        cleanup_workspace=True,
        seed=42
    ):

        set_seed(seed)
        self.seed = seed

        self.occ_df = prepare_occurrences(occ_df)
        self.bg_df = bg_df

        self.predictors_dir = Path(predictors_dir)

        if not self.predictors_dir.exists():
            raise FileNotFoundError(f"Predictors directory not found: {self.predictors_dir}")

        self.k = k
        self.partition_method = partition_method
        self.cleanup_workspace = cleanup_workspace

        self.workspace_manager = WorkspaceManager()

        self.results = None
        self.tuning_results = None
        self.best_model = None
        self.bg_df = self.bg_df.rename(columns={
        "lon": "longitude",
        "lat": "latitude"
       })

    # -----------------------------------
    # PARTITIONING
    # -----------------------------------
    def _get_splits(self):

        if self.partition_method == "random":
            kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
            return list(kf.split(self.occ_df))

        elif self.partition_method == "block":
            folds = block_partition(self.occ_df)

        elif self.partition_method == "checkerboard":
            folds = checkerboard_partition(self.occ_df)

        elif self.partition_method == "checkerboard2":
            folds = checkerboard_partition_2(self.occ_df)

        else:
            raise ValueError(f"partition_method '{self.partition_method}' no soportado")

        splits = []
        for f in np.unique(folds):
            test_idx = np.where(folds == f)[0]
            train_idx = np.where(folds != f)[0]
            splits.append((train_idx, test_idx))

        return splits

    # -----------------------------------
    # KFOLD (MAXNET)
    # -----------------------------------
    def run_kfold(self, feature_class="LQH", regularization=1, threshold=0.5):

        splits = self._get_splits()
        fold_results = []

        for i, (train_idx, test_idx) in enumerate(splits):

            train_occ = self.occ_df.iloc[train_idx]
            test_occ = self.occ_df.iloc[test_idx]

            # CRS correcto
            train_ll = train_occ.to_crs("EPSG:4326")
            test_ll = test_occ.to_crs("EPSG:4326")

            coords_train = [(g.x, g.y) for g in train_ll.geometry]
            coords_test = [(g.x, g.y) for g in test_ll.geometry]
            coords_bg = list(zip(self.bg_df["longitude"], self.bg_df["latitude"]))

            # -----------------------------
            # VARIABLES AMBIENTALES
            # -----------------------------
            X_train_pres = sample_raster(self.predictors_dir, coords_train)
            X_test_pres = sample_raster(self.predictors_dir, coords_test)
            X_bg = sample_raster(self.predictors_dir, coords_bg)

            # -----------------------------
            # DATASET
            # -----------------------------
            X_train = np.vstack([X_train_pres, X_bg])
            y_train = np.concatenate([np.ones(len(X_train_pres)), np.zeros(len(X_bg))])

            X_test = np.vstack([X_test_pres, X_bg])
            y_test = np.concatenate([np.ones(len(X_test_pres)), np.zeros(len(X_bg))])

            # -----------------------------
            # MODELO MAXNET
            # -----------------------------
            model = MaxNetModel(feature_class, regularization)
            model.fit(X_train, y_train)
            y_pred_train = model.predict_proba(X_train)
            y_pred_test = model.predict_proba(X_test)

            # -----------------------------
            # MÉTRICAS
            # -----------------------------
            auc_train = auc(y_train, y_pred_train)
            auc_test = auc(y_test, y_pred_test)
            auc_diff = auc_train - auc_test

            omission = omission_rate(y_test, y_pred_test, threshold)
            tss_score = tss(y_test, y_pred_test, threshold)
            acc = accuracy(y_test, y_pred_test, threshold)
            kap = kappa(y_test, y_pred_test, threshold)
            pps = predicted_presence_sum(y_pred_test, threshold)

            try:
                logLik = log_likelihood(y_pred_test)
                k = model.get_num_parameters(X_train)
                aicc = compute_aicc(logLik, k, len(y_test))
            except:
                aicc = np.nan

            fold_results.append({
                "fold": i + 1,
                "feature_class": feature_class,
                "regularization": regularization,
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_diff": auc_diff,
                "omission_rate": omission,
                "tss": tss_score,
                "accuracy": acc,
                "kappa": kap,
                "predicted_presence_sum": pps,
                "num_parameters": k,
                "aicc": aicc,
                "model": model
            })

        self.results = pd.DataFrame(fold_results)
        return self.results

    # -----------------------------------
    # TUNING
    # -----------------------------------
    def run_tuning(self, feature_classes, regularization_values, threshold=0.5):

        grid = generate_parameter_grid(feature_classes, regularization_values)

        all_results = []

        for _, row in grid.iterrows():
            fc = row["feature_class"]
            rm = row["regularization"]

            print(f"Evaluando FC={fc} RM={rm}")

            fold_results = self.run_kfold(fc, rm, threshold)
            all_results.append(fold_results)

        all_results = pd.concat(all_results, ignore_index=True)

        summary = summarize_models(all_results)
        summary = compute_delta_aicc(summary)
        ranking = rank_models(summary)
        best_model = select_best_model(summary)
        # =====================================
        # RECUPERAR MODELO DEL MEJOR PARÁMETRO
        # =====================================
        best_fc = best_model["feature_class"]
        best_rm = best_model["regularization"]

        # filtrar todos los folds con esos parámetros
        filtered = all_results[
            (all_results["feature_class"] == best_fc) &
            (all_results["regularization"] == best_rm)
        ]

        # tomar el modelo del último fold
        best_trained_model = filtered.iloc[-1]["model"]

        # agregarlo al best_model
        best_model = best_model.copy()
        best_model["model"] = best_trained_model
        self.tuning_results = ranking

        output_path = OUTPUTS_DIR / "tuning_results.csv"
        ranking.to_csv(output_path, index=False)

        if self.cleanup_workspace:
            self.workspace_manager.cleanup()

        return ranking, best_model
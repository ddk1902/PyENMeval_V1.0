import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.spatial import distance_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os

# ===============================
# VALIDATIONS
# ===============================
def validate_occurrences(df: pd.DataFrame):
    required_cols = ['species', 'lon', 'lat', 'year']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in occurrence DataFrame: {missing}")
    if df.empty:
        raise ValueError("The DataFrame of occurrences is empty.")
    print("DataFrame of occurrences successfully validated.")

def validate_gdf_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have a defined CRS.")
    print("GeoDataFrame has CRS defined correctly.")

def validate_min_points(df: pd.DataFrame, min_points: int = 5):
    if len(df) < min_points:
        raise ValueError(f"Insufficient data: {len(df)} points. At least {min_points} are required.")
    print(f"Sufficient number of points: {len(df)} >= {min_points}")

def validate_rasters(raster_paths: list):
    if not raster_paths:
        raise ValueError("rasters list empty.")
    for r in raster_paths:
        if not os.path.exists(r):
            raise FileNotFoundError(f"Raster not found: {r}")
        with rasterio.open(r):
            pass
    print("Correctly validated raster files.")

# ===============================
# AUXILIARY FUNCTIONS
# ===============================
def _thin_distance_once(coords, min_dist):
    n = coords.shape[0]
    keep = []
    removed = np.zeros(n, dtype=bool)
    distm = distance_matrix(coords, coords)
    np.fill_diagonal(distm, np.inf)

    for i in range(n):
        if removed[i]:
            continue
        keep.append(i)
        too_close = np.where(distm[i] < min_dist)[0]
        removed[too_close] = True

    return np.array(keep, dtype=int)

def thin_points(presences, distance=5):
    pres = presences.copy().reset_index(drop=True)
    coords = pres[['lon', 'lat']].values
    if len(coords) == 0:
        return pres
    keep_idx = _thin_distance_once(coords, distance)
    thinned = pres.iloc[keep_idx]
    print(f"Thinning: {len(pres)} → {len(thinned)} points")
    return thinned

def stratified_downsample_by_region(gdf, region_col, target_n=None, random_state=42):
    counts = gdf[region_col].value_counts()
    if target_n is None:
        target_n = int(counts.min())

    sampled_list = []
    for region, cnt in counts.items():
        sub = gdf[gdf[region_col] == region]
        if len(sub) <= target_n:
            sampled_list.append(sub.copy())
        else:
            sampled_list.append(sub.sample(n=target_n, random_state=random_state))

    result = pd.concat(sampled_list, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry='geometry', crs=gdf.crs)

def generate_background(presences, n=1000, seed=42, buffer=0.1, x_col='lon', y_col='lat'):
    rng = np.random.default_rng(seed)

    xmin, xmax = presences[x_col].min(), presences[x_col].max()
    ymin, ymax = presences[y_col].min(), presences[y_col].max()

    dx = (xmax - xmin) * buffer
    dy = (ymax - ymin) * buffer

    xmin, xmax = xmin - dx, xmax + dx
    ymin, ymax = ymin - dy, ymax + dy

    bg_x = rng.uniform(xmin, xmax, n)
    bg_y = rng.uniform(ymin, ymax, n)

    return pd.DataFrame({x_col: bg_x, y_col: bg_y})

def extract_raster_values(points_df, raster_files):
    values = []
    for raster_path in raster_files:
        with rasterio.open(raster_path) as src:
            coords = list(zip(points_df['lon'], points_df['lat']))
            vals = [val[0] for val in src.sample(coords)]
            values.append(vals)
    return np.array(values).T

# ===============================
# MAIN CLASS
# ===============================
class ENMevaluate:
    def __init__(self, occ_df, env_values, bg_df, k=5, partition_method="kfold",
                 thin_distance=None, balance_region_col=None, seed=42):

        validate_occurrences(occ_df)
        if isinstance(occ_df, gpd.GeoDataFrame):
            validate_gdf_crs(occ_df)
        validate_min_points(occ_df, min_points=5)

        if bg_df is not None and len(bg_df) == 0:
            raise ValueError("bg_df cannot be empty")

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.occ_df = occ_df.copy()
        self.env_values = env_values
        self.bg_df = bg_df
        self.k = k
        self.partition_method = partition_method
        self.thin_distance = thin_distance
        self.balance_region_col = balance_region_col

        self.models = []
        self.results = None
        self.model = None

        # thinning
        if self.thin_distance is not None:
            self.occ_df = thin_points(self.occ_df, self.thin_distance)

        # regional balancing
        if self.balance_region_col is not None:
            if not isinstance(self.occ_df, gpd.GeoDataFrame):
                raise ValueError("balance_region_col requires GeoDataFrame")
            self.occ_df = stratified_downsample_by_region(
                self.occ_df, self.balance_region_col, random_state=self.seed
            )

    # -------------------------------
    def _train_maxent(self, train_idx, X_all, y_all):
        train_X = X_all[train_idx]
        train_y = y_all[train_idx]

        if len(np.unique(train_y)) < 2:
            raise ValueError("Training set contains only one class.")

        model = LogisticRegression(solver='liblinear', max_iter=1000)
        model.fit(train_X, train_y)
        return model

    def _predict_proba_on_indices(self, model, idx, X_all):
        return model.predict_proba(X_all[idx])[:, 1]

    # -------------------------------
    def run_kfold(self):
        if self.bg_df is None:
            raise ValueError("bg_df required for training.")

        X = np.array(self.env_values)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pres = np.ones(len(self.occ_df))

        bg_array = np.array(self.bg_df)
        if bg_array.ndim == 1:
            bg_array = bg_array.reshape(-1, 1)

        X_all = np.vstack([X, bg_array])
        y_all = np.concatenate([y_pres, np.zeros(len(bg_array))])

        occ_n = len(self.occ_df)
        occ_indices = np.arange(occ_n)
        bg_indices = np.arange(occ_n, len(X_all))

        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        fold_results = []
        self.models = []

        for i, (train_occ_idx, test_occ_idx) in enumerate(kf.split(occ_indices)):
            train_idx = np.concatenate([train_occ_idx, bg_indices])

            test_bg_idx = self.rng.choice(
                bg_indices,
                size=min(len(bg_indices), len(test_occ_idx)*10),
                replace=False
            )

            test_idx = np.concatenate([test_occ_idx, test_bg_idx])

            try:
                model = self._train_maxent(train_idx, X_all, y_all)
                self.models.append(model)
            except Exception as e:
                print(f"Error fold {i+1}: {e}")
                self.models.append(None)
                fold_results.append({'fold': i+1, 'auc': np.nan, 'omission_rate': np.nan})
                continue

            y_true = y_all[test_idx]
            y_score = self._predict_proba_on_indices(model, test_idx, X_all)

            auc_val = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan

            pres_scores = y_score[:len(test_occ_idx)]
            omission = np.mean((pres_scores < 0.5).astype(int))

            fold_results.append({'fold': i+1, 'auc': auc_val, 'omission_rate': omission})

            print(f"Fold {i+1}: AUC={auc_val}, omission={omission}")

        self.results = pd.DataFrame(fold_results)

        valid = self.results.dropna(subset=['auc'])
        if valid.empty or valid['auc'].max() <= 0:
            print("There are no valid folds.")
            self.model = None
        else:
            best_fold = int(valid.loc[valid['auc'].idxmax()]['fold'])
            self.model = self.models[best_fold - 1]
            print(f"Best fold: {best_fold}")

        return self.results

    # -------------------------------
    def predict_to_raster(self, raster_files, output_path):
        validate_rasters(raster_files)

        if self.model is None:
            raise ValueError("Run model before predicting.")

        env_stack = []
        for raster_path in raster_files:
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                env_stack.append(data.flatten())
                profile = src.profile

        env_stack = np.array(env_stack).T
        y_pred = self.model.predict_proba(env_stack)[:, 1]
        y_pred_raster = y_pred.reshape(data.shape)

        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(y_pred_raster.astype(rasterio.float32), 1)

        print(f"Raster saved: {output_path}")
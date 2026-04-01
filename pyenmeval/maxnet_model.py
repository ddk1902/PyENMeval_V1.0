import numpy as np
from sklearn.linear_model import LogisticRegression


# -----------------------------------
# FEATURE ENGINEERING (L, Q, H)
# -----------------------------------
def build_features(X, feature_class):
    """
    Construye features tipo maxnet:
    L = linear
    Q = quadratic
    H = hinge
    """

    feats = []

    # Linear
    if "L" in feature_class:
        feats.append(X)

    # Quadratic
    if "Q" in feature_class:
        feats.append(X ** 2)

    # Hinge (aproximación maxnet)
    if "H" in feature_class:
        hinge = np.maximum(0, X - np.mean(X, axis=0))
        feats.append(hinge)

    return np.hstack(feats)


# -----------------------------------
# MODELO MAXNET (Logistic Regression)
# -----------------------------------
class MaxNetModel:
    """
    Implementación tipo maxnet usando sklearn.
    """

    def __init__(self, feature_class="LQH", regularization=1):
        self.feature_class = feature_class
        self.regularization = regularization
        self.model = None

    def fit(self, X, y):
        """
        Entrena el modelo.
        """
        X_feat = X.copy()
        X_feat = build_features(X, self.feature_class)

        self.model = LogisticRegression(
            penalty="l1",              # equivalente a maxnet
            solver="liblinear",
            C=1 / self.regularization,
            max_iter=1000
        )
        print("Shape X original:", X.shape)
        print("Shape X_feat:", X_feat.shape)
        self.model.fit(X_feat, y)

    def predict_proba(self, X):

       X_feat = build_features(X, self.feature_class)

       probs = self.model.predict_proba(X_feat)

       if probs.ndim == 1:
        return probs
       else:
        return probs[:, 1]

    def get_num_parameters(self, X):
        """
        Número de parámetros (para AICc).
        """

        X_feat = build_features(X, self.feature_class)
        return X_feat.shape[1]
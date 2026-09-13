"""Model zoo + feature pipeline. One traced fit(state, X, y) -> sklearn Pipeline."""
import weave
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from .data import FEATURES, CATEGORICAL, POSITIVE_SKEWED

MODEL_ZOO = {
    "knn": lambda hp: KNeighborsClassifier(**hp),
    "logistic_regression": lambda hp: LogisticRegression(max_iter=2000, **hp),
    "random_forest": lambda hp: RandomForestClassifier(random_state=42, **hp),
    "svm": lambda hp: SVC(probability=True, random_state=42, **hp),
    "gradient_boosting": lambda hp: GradientBoostingClassifier(random_state=42, **hp),
}

# Small, sane grids the agent may tune within (SPEC section 5: "small grid per family").
HYPERPARAM_GRID = {
    "knn": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
    "random_forest": {"n_estimators": [100, 300, 500], "max_depth": [None, 4, 8, 12],
                      "min_samples_leaf": [1, 2, 5], "class_weight": [None, "balanced"]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "class_weight": [None, "balanced"]},
    "gradient_boosting": {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1],
                          "max_depth": [2, 3, 4]},
}

_CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL]
_LOG_IDX = [FEATURES.index(c) for c in POSITIVE_SKEWED]
_OTHER_IDX = [i for i in range(len(FEATURES)) if i not in _CAT_IDX]


def validate_hyperparams(family: str, params: dict) -> dict:
    """Keep only params in the family's grid, coerced to allowed values. Never raises:
    a bad suggestion becomes a no-op, which the effect record will show as delta 0."""
    grid = HYPERPARAM_GRID.get(family, {})
    clean = {}
    for k, v in params.items():
        if k not in grid:
            continue
        allowed = grid[k]
        if v in allowed:
            clean[k] = v
        elif isinstance(v, (int, float)) and all(isinstance(a, (int, float)) for a in allowed if a is not None):
            nums = [a for a in allowed if a is not None]
            clean[k] = min(nums, key=lambda a: abs(a - v))  # snap to nearest grid point
    return clean


def build_pipeline(model_family: str, feature_ops: list[str], hyperparams: dict) -> Pipeline:
    steps = []
    # impute_median is always on (the data has NaNs); listing it is a no-op the agent may take
    if "log_scale" in feature_ops:
        steps.append(("log", ColumnTransformer(
            [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), _LOG_IDX)],
            remainder="passthrough", verbose_feature_names_out=False)))
        # ColumnTransformer reorders columns: logged cols first, then the rest
        order = _LOG_IDX + [i for i in range(len(FEATURES)) if i not in _LOG_IDX]
        cat_idx = [order.index(i) for i in _CAT_IDX]
        other_idx = [order.index(i) for i in _OTHER_IDX]
    else:
        cat_idx, other_idx = _CAT_IDX, _OTHER_IDX
    steps.append(("impute", SimpleImputer(strategy="median")))
    if "onehot" in feature_ops:
        num_steps = [StandardScaler()] if "standardize" in feature_ops else ["passthrough"]
        steps.append(("encode", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
            ("num", num_steps[0], other_idx),
        ])))
    elif "standardize" in feature_ops:
        steps.append(("scale", StandardScaler()))
    hp = validate_hyperparams(model_family, hyperparams)
    steps.append(("clf", MODEL_ZOO[model_family](hp)))
    return Pipeline(steps)


@weave.op
def train(state, X_train: np.ndarray, y_train: np.ndarray):
    """Fit current model family with current feature ops. Fully traced."""
    pipe = build_pipeline(state.model_family, list(state.feature_ops), dict(state.hyperparams))
    pipe.fit(X_train, y_train)
    return pipe

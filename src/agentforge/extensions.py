"""Self-extending action space.

When every typed action is exhausted the loop emits `request_code_fix`, asking ARIA (or a
human applying ARIA's diff) to ADD an option here — a new model family or a new feature
op. After `aria_requests/NNN.applied` lands, `check_for_patches()` reloads this module and
the new option appears in the action space on the next iteration: the agent expands its
own action space, and the trace records who expanded it.

Contract for ARIA (keep it this simple):

    register_model("extra_trees", lambda hp: ExtraTreesClassifier(random_state=42, **hp),
                   "randomised trees: lower variance than random_forest on small data",
                   grid={"n_estimators": [100, 300], "max_depth": [None, 6, 10]})
    register_feature_op("quantile", lambda: QuantileTransformer(n_quantiles=50),
                        "rank-transform every feature (robust to outliers/skew)")

Everything above the marker is infrastructure; ARIA appends below it.
"""
from collections.abc import Callable

BASE_MODELS = ["knn", "logistic_regression", "random_forest", "svm", "gradient_boosting"]
BASE_FEATURE_OPS = ["standardize", "impute_median", "onehot", "log_scale"]

EXTRA_MODELS: dict[str, dict] = {}        # name -> {"factory", "description", "grid"}
EXTRA_FEATURE_OPS: dict[str, dict] = {}   # name -> {"factory", "description"}


def register_model(name: str, factory: Callable[[dict], object], description: str, grid: dict | None = None):
    if name in BASE_MODELS:
        raise ValueError(f"{name!r} is a base model family")
    EXTRA_MODELS[name] = {"factory": factory, "description": description, "grid": grid or {}}


def register_feature_op(name: str, factory: Callable[[], object], description: str):
    if name in BASE_FEATURE_OPS:
        raise ValueError(f"{name!r} is a base feature op")
    EXTRA_FEATURE_OPS[name] = {"factory": factory, "description": description}


def model_families() -> list[str]:
    return BASE_MODELS + list(EXTRA_MODELS)


def feature_ops() -> list[str]:
    return BASE_FEATURE_OPS + list(EXTRA_FEATURE_OPS)


def snapshot() -> dict:
    """What the action space currently contains (logged before/after an ARIA patch)."""
    return {"models": model_families(), "feature_ops": feature_ops()}


# ==== ARIA-added extensions go below this line ==============================================

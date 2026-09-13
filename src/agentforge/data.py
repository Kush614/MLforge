"""UCI Heart Disease, split by original collection site.
Run scripts/fetch_data.py first to populate data/.

Sites are revealed to the agent one at a time: it starts on `initial_site` and must
spend an `acquire_data` action to get each additional one.
"""
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

from .tracing import ROOT

DATA_DIR = ROOT / "data"

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"

# feature groups used by transform_features ops
CATEGORICAL = ["cp", "restecg", "slope", "thal"]          # multi-valued codes -> onehot
POSITIVE_SKEWED = ["trestbps", "chol", "oldpeak"]         # log1p candidates
BINARY = ["sex", "fbs", "exang"]


def site_path(site: str) -> Path:
    return DATA_DIR / f"{site}.csv"


def available_sites() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []


def load_sites(sites: list[str], allow_synthetic: bool = False) -> pd.DataFrame:
    """Concatenate the requested sites. Target is binarized: 0 = no disease, 1 = any."""
    frames = []
    for s in sites:
        p = site_path(s)
        if not p.exists():
            if allow_synthetic:
                frames.append(synthetic_site(s))
                continue
            raise FileNotFoundError(f"Missing {p}. Run scripts/fetch_data.py first.")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    # drop rows with missing label; features may be NaN (imputer handles them)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def synthetic_site(site: str, n: int | None = None) -> pd.DataFrame:
    """Deterministic stand-in with the real schema, used ONLY when data/ is absent
    (dry-run / tests). Never used silently in a real run: load_sites raises instead."""
    sizes = {"switzerland": 123, "va": 200, "hungary": 294, "cleveland": 303}
    n = n or sizes.get(site, 150)
    seed = int(hashlib.md5(site.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    age = rng.normal(54, 9, n).clip(29, 77)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    trestbps = rng.normal(131, 17, n).clip(90, 200)
    chol = rng.normal(246, 52, n).clip(100, 564)
    fbs = rng.integers(0, 2, n)
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(150, 23, n).clip(70, 202)
    exang = rng.integers(0, 2, n)
    oldpeak = rng.exponential(1.0, n).clip(0, 6.2)
    slope = rng.integers(1, 4, n)
    ca = rng.integers(0, 4, n)
    thal = rng.choice([3, 6, 7], n)
    logit = (0.04 * (age - 54) + 0.8 * sex + 0.5 * (cp == 4) - 0.02 * (thalach - 150)
             + 0.9 * exang + 0.6 * oldpeak + 0.5 * ca + 0.7 * (thal == 7) - 1.5)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                           restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                           slope=slope, ca=ca, thal=thal, target=target))
    # mimic missingness of the small sites
    for col in ["chol", "ca", "thal", "slope"]:
        df.loc[rng.random(n) < 0.15, col] = np.nan
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the dataset content so the trace can prove which data was used."""
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values).hexdigest()[:12]


def fixed_split(all_sites: list[str], test_size: float, seed: int, allow_synthetic: bool = False,
                poison: dict | None = None) -> dict:
    """Per-site stratified train/test split, computed ONCE for every site (revealed or not).

    The test set is the union of every site's test rows, so it is identical on every
    iteration: acquiring a site only grows the training pool. That is what makes the
    effect ledger (before/after per action) an honest comparison."""
    from sklearn.model_selection import train_test_split
    parts = {}
    for s in all_sites:
        p = site_path(s)
        if p.exists():
            df = pd.read_csv(p)
        elif allow_synthetic:
            df = synthetic_site(s)
        else:
            raise FileNotFoundError(f"Missing {p}. Run scripts/fetch_data.py first.")
        df[TARGET] = (df[TARGET] > 0).astype(int)
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        strat = df[TARGET] if df[TARGET].nunique() > 1 and df[TARGET].value_counts().min() >= 2 else None
        tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
        tr = tr.reset_index(drop=True)
        if poison and poison.get("site") == s and poison.get("label_noise"):
            tr = poison_labels(tr, float(poison["label_noise"]), seed)   # TRAIN rows only
        parts[s] = (tr, te.reset_index(drop=True))
    return parts


def poison_labels(train_df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Flip `frac` of the labels (documented adversarial site). Returns a copy."""
    out = train_df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=int(round(frac * len(out))), replace=False)
    out.loc[idx, TARGET] = 1 - out.loc[idx, TARGET]
    return out


def assemble(parts: dict, train_sites: list[str]):
    """(X_train, X_test, y_train, y_test, fingerprint) for the current revealed sites."""
    train_df = pd.concat([parts[s][0] for s in train_sites], ignore_index=True)
    test_df = pd.concat([parts[s][1] for s in parts], ignore_index=True)
    X_tr = train_df[FEATURES].values.astype(float); y_tr = train_df[TARGET].values
    X_te = test_df[FEATURES].values.astype(float); y_te = test_df[TARGET].values
    return X_tr, X_te, y_tr, y_te, fingerprint(train_df)

"""Honest evaluation on a seeded held-out split. Everything the diagnoser sees comes from here."""
import weave
import numpy as np
from dataclasses import dataclass, asdict
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)


@dataclass
class EvalResult:
    accuracy: float
    balanced_accuracy: float     # guards against majority-class 'wins' on skewed sites
    train_accuracy: float
    train_test_gap: float
    per_class: dict
    confusion: list
    class_balance: dict          # fraction of positives in train / test
    n_train: int
    n_test: int
    n_features: int
    learning_curve: list         # [(n_samples, test_acc)] at 50/75/100% of train
    worst_examples: list         # test indices the model got wrong with highest confidence

    def summary(self) -> str:
        return (
            f"test_acc={self.accuracy:.3f} bal_acc={self.balanced_accuracy:.3f} train_acc={self.train_accuracy:.3f} "
            f"gap={self.train_test_gap:+.3f} n_train={self.n_train} n_test={self.n_test} "
            f"per_class={self.per_class}"
        )


def _learning_curve(model, X_train, y_train, X_test, y_test) -> list:
    """Test accuracy when refit on 50% and 75% of the train set (+ the full-fit point).
    A steep slope = data_starved; flat = model/feature bound."""
    pts = []
    n = len(y_train)
    for frac in (0.5, 0.75):
        k = max(int(n * frac), 20)
        try:
            m = clone(model).fit(X_train[:k], y_train[:k])
            pts.append([k, round(float(accuracy_score(y_test, m.predict(X_test))), 4)])
        except Exception:  # e.g. only one class in the subsample
            pts.append([k, None])
    pts.append([n, round(float(accuracy_score(y_test, model.predict(X_test))), 4)])
    return pts


@weave.op
def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    """Returns a weave-serializable dict (see EvalResult)."""
    y_train = np.asarray(y_train); y_test = np.asarray(y_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1], zero_division=0)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = preds.astype(float)
    wrong = np.where(preds != y_test)[0]
    conf_wrong = wrong[np.argsort(-np.abs(np.asarray(proba)[wrong] - 0.5))][:5]
    res = EvalResult(
        accuracy=float(acc),
        balanced_accuracy=float(balanced_accuracy_score(y_test, preds)),
        train_accuracy=float(train_acc),
        train_test_gap=float(train_acc - acc),
        per_class={"precision": p.round(3).tolist(), "recall": r.round(3).tolist(),
                   "f1": f1.round(3).tolist()},
        confusion=confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        class_balance={"train_pos_frac": round(float(y_train.mean()), 3),
                       "test_pos_frac": round(float(y_test.mean()), 3)},
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        n_features=int(np.asarray(X_train).shape[1]),
        learning_curve=_learning_curve(model, X_train, y_train, X_test, y_test),
        worst_examples=conf_wrong.tolist(),
    )
    return asdict(res)

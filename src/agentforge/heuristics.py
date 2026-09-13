"""Rule-based diagnosis + decision. Used for --dry-run (no network) and as the
last-resort fallback when both LLM heads fail. Always labeled 'heuristic' in traces.

These rules encode the same evidence tags the LLM is asked to use, so a dry run
exercises the identical action space and effect bookkeeping as a live run."""
import weave
from .actions import ActionEnvelope
from . import extensions

FAMILY_ORDER = ["random_forest", "gradient_boosting", "logistic_regression", "svm", "knn"]


@weave.op
def heuristic_diagnose(ev: dict, history_summary: str) -> dict:
    tags, why = [], []
    gap = ev["train_test_gap"]
    rec = ev["per_class"]["recall"]
    lc = ev.get("learning_curve") or []
    if gap > 0.12:
        tags.append("overfitting"); why.append(f"train/test gap {gap:+.3f} > 0.12")
    if ev["train_accuracy"] < 0.80:
        tags.append("underfitting"); why.append(f"train acc {ev['train_accuracy']:.3f} < 0.80")
    if ev["n_train"] < 250:
        tags.append("data_starved"); why.append(f"n_train={ev['n_train']} is small")
    if len(lc) >= 2 and lc[0][1] is not None and lc[-1][1] is not None and lc[-1][1] - lc[0][1] > 0.03:
        if "data_starved" not in tags:
            tags.append("data_starved")
        why.append(f"learning curve still rising ({lc[0][1]:.3f} -> {lc[-1][1]:.3f})")
    if min(rec) < 0.5 and max(rec) > 0.85:
        tags.append("class_imbalance"); why.append(f"per-class recall {rec} lopsided")
    if "standardize" not in history_summary and ev.get("n_features", 13) == 13:
        tags.append("feature_scale"); why.append("raw features have very different scales")
    if "delta -0" in history_summary or "delta +0.000" in history_summary:
        tags.append("plateau"); why.append("recent actions produced no gain")
    return {"reasoning": "heuristic: " + "; ".join(why) if why else "heuristic: no strong signal",
            "evidence_tags": tags, "history_source": "local", "cited_trace_ids": []}


@weave.op
def heuristic_decide(diagnosis: dict, unrevealed_sites: list[str], model_family: str,
                     feature_ops: list[str], tried_families: list[str]) -> dict:
    tags = set(diagnosis.get("evidence_tags", []))
    action = None
    if "data_starved" in tags and unrevealed_sites:
        action = {"kind": "acquire_data", "site": unrevealed_sites[0],
                  "reason": "learning curve / n_train says more data helps most"}
    elif "feature_scale" in tags and "standardize" not in feature_ops and model_family in ("knn", "svm", "logistic_regression"):
        action = {"kind": "transform_features", "op": "standardize", "reason": "distance/linear model on unscaled features"}
    elif "overfitting" in tags and model_family in ("random_forest", "gradient_boosting"):
        action = {"kind": "tune_hyperparams", "params": {"max_depth": 4, "min_samples_leaf": 5} if model_family == "random_forest" else {"max_depth": 2, "learning_rate": 0.03},
                  "reason": "regularize to close the train/test gap"}
    elif "class_imbalance" in tags and model_family in ("random_forest", "logistic_regression", "svm"):
        action = {"kind": "tune_hyperparams", "params": {"class_weight": "balanced"}, "reason": "lopsided recall"}
    if action is None:
        order = FAMILY_ORDER + [f for f in extensions.EXTRA_MODELS if f not in FAMILY_ORDER]
        nxt = [f for f in order if f != model_family and f not in tried_families]
        extra_ops = [o for o in extensions.EXTRA_FEATURE_OPS if o not in feature_ops]
        if nxt:
            action = {"kind": "switch_model", "family": nxt[0], "reason": "try the next strongest family"}
        elif unrevealed_sites:
            action = {"kind": "acquire_data", "site": unrevealed_sites[0], "reason": "families exhausted; more data"}
        elif "onehot" not in feature_ops:
            action = {"kind": "transform_features", "op": "onehot", "reason": "encode categorical codes"}
        elif extra_ops:
            action = {"kind": "transform_features", "op": extra_ops[0], "reason": "try the ARIA-added feature op"}
        else:
            action = {"kind": "request_code_fix",
                      "description": "All typed actions exhausted without beating target "
                                     f"(families tried: {tried_families + [model_family]}, ops: {feature_ops}). "
                                     "Expand the action space: register a NEW model family or feature op in "
                                     "src/agentforge/extensions.py (register_model / register_feature_op).",
                      "failing_trace_ids": diagnosis.get("cited_trace_ids", [])}
    env = ActionEnvelope.model_validate({"action": action})
    return {"provider": "heuristic", "action": env.model_dump()}

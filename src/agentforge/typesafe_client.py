"""Structured action head: diagnosis -> exactly one typed Action.

Provider order (each labeled honestly in the returned 'provider' field):
  1. TypeSafe AI System One (Jev) — when TYPESAFE_API_KEY is set. Jev does not generate
     text: it answers typed Choice questions with calibrated probabilities. So the head is
     a fan-out of independent questions over the same state (docs: "route and fill known
     arguments"): `action_kind` plus speculative `model_family` / `feature_op` / `site` /
     `hyperparam_preset`. Code consumes only the branch that was chosen and builds the
     ActionEnvelope. Confidence + per-option probabilities are kept in the trace.
  2. W&B Inference — prompt-and-parse JSON through llm.py.
  3. heuristic — deterministic rules (dry-run, or both LLM heads failed).
The action is validated by pydantic before it is returned; free text never reaches act()."""
import json
import os
import weave
from dotenv import load_dotenv
from pydantic import ValidationError
from .actions import ActionEnvelope, ACTION_SCHEMA_HINT
from .heuristics import heuristic_decide
from .train import HYPERPARAM_GRID
from . import llm

load_dotenv()

TYPESAFE_MODEL = os.environ.get("TYPESAFE_MODEL", "").strip() or None  # SDK default: jev-latest

FAMILY_DESC = {
    "knn": "k-nearest neighbours: distance-based, needs scaled features, weak on small noisy data",
    "logistic_regression": "linear model: stable on small data, needs scaled features, cannot fit interactions",
    "random_forest": "bagged trees: handles unscaled/missing-ish tabular data, robust default, can overfit small sets",
    "svm": "RBF support vector machine: strong on scaled mid-size data, sensitive to C",
    "gradient_boosting": "boosted trees: highest capacity, overfits small data without regularisation",
}
OP_DESC = {
    "standardize": "z-score every feature (helps knn / svm / logistic_regression)",
    "onehot": "one-hot encode categorical codes cp / restecg / slope / thal",
    "log_scale": "log1p the skewed positives trestbps / chol / oldpeak",
    "impute_median": "median imputation (already always on; choosing it is a no-op)",
}
SITE_DESC = {
    "switzerland": "123 rows, 93% positive, many missing chol/ca/thal",
    "hungary": "294 rows, 36% positive, slope/ca/thal mostly missing",
    "va": "200 rows, 74% positive, ~25% missing on several columns",
    "cleveland": "303 rows, 46% positive, complete features (the classic benchmark site)",
}
# Named presets: Jev picks the intent, code owns the exact sklearn values (all inside HYPERPARAM_GRID).
HYPERPARAM_PRESETS = {
    "knn": {"more_neighbors": ("smooth decision boundary (n_neighbors=11, distance weights)", {"n_neighbors": 11, "weights": "distance"}),
            "fewer_neighbors": ("sharper boundary (n_neighbors=3)", {"n_neighbors": 3})},
    "logistic_regression": {"stronger_regularization": ("C=0.1", {"C": 0.1}),
                            "weaker_regularization": ("C=10", {"C": 10.0}),
                            "balanced_classes": ("class_weight=balanced", {"class_weight": "balanced"})},
    "random_forest": {"regularize": ("shallower trees, bigger leaves (max_depth=4, min_samples_leaf=5)", {"max_depth": 4, "min_samples_leaf": 5}),
                      "more_capacity": ("more, deeper trees (n_estimators=500, max_depth=12)", {"n_estimators": 500, "max_depth": 12}),
                      "balanced_classes": ("class_weight=balanced", {"class_weight": "balanced"})},
    "svm": {"stronger_regularization": ("C=0.1", {"C": 0.1}),
            "weaker_regularization": ("C=10", {"C": 10.0}),
            "linear_kernel": ("kernel=linear", {"kernel": "linear"}),
            "balanced_classes": ("class_weight=balanced", {"class_weight": "balanced"})},
    "gradient_boosting": {"regularize": ("stumps, slow learning (max_depth=2, learning_rate=0.03)", {"max_depth": 2, "learning_rate": 0.03}),
                          "more_capacity": ("n_estimators=200, max_depth=4", {"n_estimators": 200, "max_depth": 4})},
}

DECIDE_SYSTEM = f"""You convert an ML diagnosis into EXACTLY ONE typed action.
Available actions (STRICT JSON, no markdown):
{ACTION_SCHEMA_HINT}
Rules:
- Pick the single action best supported by the diagnosis and NOT already shown useless by history.
- acquire_data only with a site from the offered list. switch_model only to a family not yet tried
  unless the diagnosis explicitly argues for revisiting it.
- tune_hyperparams params must be valid sklearn params for the CURRENT model family.
- request_code_fix is for problems the other four actions cannot express (a bug, a missing feature)."""


def typesafe_configured() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


# ---------------------------------------------------------------- TypeSafe head
def build_questions(unrevealed_sites, model_family, feature_ops, tried_families) -> dict:
    """Independent Choice questions over one state. Only offer options that are legal now."""
    from typesafe_sdk import Choice
    kinds = {"tune_hyperparams": "keep model and data, adjust the current model's regularisation/capacity",
             "request_code_fix": "LAST RESORT: none of the other actions can express the fix; hand a code change to an engineer"}
    other_families = {f: FAMILY_DESC[f] + (" — ALREADY TRIED, do not repeat unless history shows it helped" if f in tried_families else "")
                      for f in FAMILY_DESC if f != model_family}
    remaining_ops = {o: OP_DESC[o] for o in OP_DESC if o not in feature_ops}
    if other_families:
        kinds["switch_model"] = "change the model family (current: " + model_family + ")"
    if remaining_ops:
        kinds["transform_features"] = "add a feature-pipeline step (standardize / onehot / log_scale)"
    if unrevealed_sites:
        kinds["acquire_data"] = "add another hospital site's rows to the training pool (test set never changes)"
    q = {"action_kind": Choice(
        instructions="Given `diagnosis`, `evaluation`, the current setup and `history` of what already did or did not help, "
                     "which single next action is most likely to raise held-out accuracy?",
        criteria=kinds)}
    if other_families:
        q["model_family"] = Choice(instructions="IF switching model family, which family fits this data/diagnosis best?",
                                   criteria=other_families)
    if remaining_ops:
        q["feature_op"] = Choice(instructions="IF adding a feature-pipeline step, which one addresses the diagnosis?",
                                 criteria=remaining_ops)
    if unrevealed_sites:
        q["site"] = Choice(instructions="IF acquiring data, which site adds the most useful signal?",
                           criteria={s: SITE_DESC.get(s) for s in unrevealed_sites})
    presets = HYPERPARAM_PRESETS.get(model_family, {})
    if presets:
        q["hyperparam_preset"] = Choice(instructions=f"IF tuning the current {model_family}, which adjustment does the diagnosis call for?",
                                        criteria={k: v[0] for k, v in presets.items()})
    return q


@weave.op
def typesafe_system_one(state: dict, questions: dict, model: str | None = None) -> dict:
    """One System One request. Returns a serialisable dict: {'model', 'answers': {q: {choice, confidence, probabilities}}}."""
    from typesafe_sdk import TypeSafeClient
    with TypeSafeClient(model=model, timeout=30) as client:
        resp = client.system_one(state=state, questions=questions)
    return {"model": resp.model,
            "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            "answers": {name: {"choice": a.choice, "confidence": round(a.confidence, 4),
                               "probabilities": {k: round(v, 4) for k, v in a.probabilities.items()}}
                        for name, a in resp.answers.items()}}


@weave.op
def typesafe_decide(diagnosis: dict, unrevealed_sites: list[str], model_family: str,
                    feature_ops: list[str], tried_families: list[str], history_summary: str = "") -> dict:
    state = {
        "diagnosis": {"reasoning": diagnosis.get("reasoning", ""), "evidence_tags": diagnosis.get("evidence_tags", [])},
        "evaluation": diagnosis.get("evaluation", {}),
        "current_model": model_family, "feature_ops_applied": feature_ops,
        "families_already_tried": tried_families, "unrevealed_sites": unrevealed_sites,
        "history": history_summary[-3000:] if history_summary else "no prior iterations",
    }
    questions = build_questions(unrevealed_sites, model_family, feature_ops, tried_families)
    out = typesafe_system_one(state, questions, TYPESAFE_MODEL)
    ans = out["answers"]
    kind = ans["action_kind"]["choice"]
    reason = f"typesafe {out['model']} p={ans['action_kind']['probabilities'].get(kind, 0):.2f}"
    if kind == "switch_model":
        action = {"kind": kind, "family": ans["model_family"]["choice"], "reason": reason}
    elif kind == "transform_features":
        action = {"kind": kind, "op": ans["feature_op"]["choice"], "reason": reason}
    elif kind == "acquire_data":
        action = {"kind": kind, "site": ans["site"]["choice"], "reason": reason}
    elif kind == "tune_hyperparams":
        preset = ans["hyperparam_preset"]["choice"] if "hyperparam_preset" in ans else None
        params = HYPERPARAM_PRESETS[model_family][preset][1] if preset else {}
        action = {"kind": kind, "params": params, "reason": f"{reason} preset={preset}"}
    else:
        action = {"kind": "request_code_fix", "description": diagnosis.get("reasoning", "")[:1000],
                  "failing_trace_ids": diagnosis.get("cited_trace_ids", []), "reason": reason}
    env = ActionEnvelope.model_validate({"action": action})
    return {"provider": f"typesafe ({out['model']})", "action": env.model_dump(),
            "typesafe": {"confidence": ans["action_kind"]["confidence"],
                         "probabilities": ans["action_kind"]["probabilities"],
                         "branches": {k: v["choice"] for k, v in ans.items() if k != "action_kind"}}}


# ---------------------------------------------------------------- Inference head
def _user_prompt(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary):
    return (f"DIAGNOSIS:\n{json.dumps(diagnosis)}\n"
            f"CURRENT MODEL: {model_family}\nFAMILIES ALREADY TRIED: {tried_families}\n"
            f"FEATURE OPS ALREADY APPLIED: {feature_ops}\n"
            f"UNREVEALED DATA SITES YOU MAY ACQUIRE: {unrevealed_sites}\n"
            f"HISTORY:\n{history_summary}\n"
            "Return STRICT JSON now.")


def _validate(raw: str, unrevealed_sites: list[str]) -> ActionEnvelope:
    env = ActionEnvelope.model_validate_json(llm.extract_json(raw))
    a = env.action
    if a.kind == "acquire_data" and a.site not in unrevealed_sites:
        raise ValidationError.from_exception_data("ActionEnvelope", [{
            "type": "value_error", "loc": ("action", "site"), "input": a.site,
            "ctx": {"error": f"site must be one of {unrevealed_sites}"}}])
    return env


def inference_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families,
                     history_summary, attempts: list[dict]) -> dict | None:
    user = _user_prompt(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary)
    for attempt in range(2):  # one repair retry
        prompt = user if attempt == 0 else user + "\n\nPrevious reply was invalid: " + attempts[-1]["error"] + "\nJSON only."
        try:
            env = _validate(llm.complete(DECIDE_SYSTEM, prompt, temperature=0.0), unrevealed_sites)
            return {"provider": "wandb_inference", "action": env.model_dump()}
        except Exception as e:
            attempts.append({"provider": "wandb_inference", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if not isinstance(e, (ValidationError, json.JSONDecodeError, ValueError)):
                return None  # network/auth error: don't retry
    return None


# ---------------------------------------------------------------- the head
@weave.op
def decide_action(diagnosis: dict, unrevealed_sites: list[str], model_family: str,
                  feature_ops: list[str] | None = None, tried_families: list[str] | None = None,
                  history_summary: str = "", offline: bool = False,
                  force_provider: str | None = None) -> dict:
    """Returns {'provider': str, 'action': envelope dict, 'attempts': [...], ['typesafe': {...}]}."""
    feature_ops = feature_ops or []
    tried_families = tried_families or []
    if offline:
        out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
        out["provider"] = "heuristic (dry-run)"
        return out

    attempts: list[dict] = []
    if force_provider:
        providers = [force_provider]
    else:
        providers = ([("typesafe")] if typesafe_configured() else []) + (["wandb_inference"] if llm.available() else [])

    for prov in providers:
        if prov == "typesafe":
            try:
                out = typesafe_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary)
                out["attempts"] = attempts
                return out
            except Exception as e:
                attempts.append({"provider": "typesafe", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        elif prov == "wandb_inference":
            out = inference_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary, attempts)
            if out:
                if any(a["provider"] == "typesafe" for a in attempts):
                    out["provider"] = "wandb_inference_fallback (typesafe failed)"
                out["attempts"] = attempts
                return out

    out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
    why = "no LLM provider configured" if not providers else "; ".join(a["error"] for a in attempts)
    out["provider"] = f"heuristic_fallback ({why})"
    out["attempts"] = attempts
    return out

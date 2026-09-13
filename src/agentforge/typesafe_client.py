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
import time
import weave
from dotenv import load_dotenv
from pydantic import ValidationError
from .actions import ActionEnvelope, schema_hint
from . import extensions
from .heuristics import heuristic_decide
from .train import HYPERPARAM_GRID
from . import llm
from .tracing import config

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

DECIDE_SYSTEM_TEMPLATE = """You convert an ML diagnosis into EXACTLY ONE typed action.
Available actions (STRICT JSON, no markdown):
{schema}
Rules:
- Pick the single action best supported by the diagnosis and NOT already shown useless by history.
- acquire_data only with a site from the offered list. switch_model only to a family not yet tried
  unless the diagnosis explicitly argues for revisiting it.
- tune_hyperparams params must be valid sklearn params for the CURRENT model family.
- request_code_fix is for problems the other four actions cannot express (a bug, a missing feature)."""


def decide_system() -> str:
    return DECIDE_SYSTEM_TEMPLATE.format(schema=schema_hint())


def typesafe_configured() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


# ---------------------------------------------------------------- TypeSafe head
def remaining_presets(model_family: str, tried_hyperparams: list[str] | None) -> dict:
    """Presets for this family whose exact params have NOT already been applied (memory is binding)."""
    tried = set(tried_hyperparams or [])
    return {k: v for k, v in HYPERPARAM_PRESETS.get(model_family, {}).items()
            if json.dumps(v[1], sort_keys=True) not in tried}


def build_questions(unrevealed_sites, model_family, feature_ops, tried_families, tried_hyperparams=None) -> dict:
    """Independent Choice questions over one state. Only offer options that are legal now
    and not already shown useless: tried presets/families/ops are withdrawn, not just discouraged."""
    from typesafe_sdk import Choice
    presets = remaining_presets(model_family, tried_hyperparams)
    kinds = {"request_code_fix": "LAST RESORT: none of the other actions can express the fix; hand a code change to an engineer"}
    if presets:
        kinds["tune_hyperparams"] = "keep model and data, adjust the current model's regularisation/capacity (untried settings only)"
    fam_desc = {**FAMILY_DESC, **{k: "ARIA-added: " + v["description"] for k, v in extensions.EXTRA_MODELS.items()}}
    op_desc = {**OP_DESC, **{k: "ARIA-added: " + v["description"] for k, v in extensions.EXTRA_FEATURE_OPS.items()}}
    other_families = {f: fam_desc[f] + (" — ALREADY TRIED, do not repeat unless history shows it helped" if f in tried_families else "")
                      for f in fam_desc if f != model_family}
    remaining_ops = {o: op_desc[o] for o in op_desc if o not in feature_ops}
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
                    feature_ops: list[str], tried_families: list[str], history_summary: str = "",
                    tried_hyperparams: list[str] | None = None) -> dict:
    state = {
        "diagnosis": {"reasoning": diagnosis.get("reasoning", ""), "evidence_tags": diagnosis.get("evidence_tags", [])},
        "evaluation": diagnosis.get("evaluation", {}),
        "current_model": model_family, "feature_ops_applied": feature_ops,
        "families_already_tried": tried_families, "unrevealed_sites": unrevealed_sites,
        "hyperparam_settings_already_tried_for_current_model": tried_hyperparams or [],
        "history": history_summary[-3000:] if history_summary else "no prior iterations",
    }
    # Plain JSON only: inside a weave.Evaluation the inputs arrive as Weave-boxed dict/list
    # types that the SDK's msgspec encoder rejects ("request body could not be encoded").
    state = json.loads(json.dumps(state, default=str))
    unrevealed_sites, feature_ops, tried_families = list(unrevealed_sites), list(feature_ops), list(tried_families)
    questions = build_questions(unrevealed_sites, model_family, feature_ops, tried_families, tried_hyperparams)
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
                         "usage": out.get("usage"),
                         "branches": {k: v["choice"] for k, v in ans.items() if k != "action_kind"}}}


# ---------------------------------------------------------------- Inference head
def _user_prompt(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary,
                 tried_hyperparams=None):
    return (f"DIAGNOSIS:\n{json.dumps(diagnosis)}\n"
            f"CURRENT MODEL: {model_family}\nFAMILIES ALREADY TRIED: {tried_families}\n"
            f"HYPERPARAM SETTINGS ALREADY TRIED FOR THIS MODEL (do not repeat): {tried_hyperparams or []}\n"
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
                     history_summary, attempts: list[dict], tried_hyperparams=None) -> dict | None:
    user = _user_prompt(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary, tried_hyperparams)
    for attempt in range(2):  # one repair retry
        prompt = user if attempt == 0 else user + "\n\nPrevious reply was invalid: " + attempts[-1]["error"] + "\nJSON only."
        try:
            env = _validate(llm.complete(decide_system(), prompt, temperature=0.0), unrevealed_sites)
            return {"provider": "wandb_inference", "action": env.model_dump()}
        except Exception as e:
            attempts.append({"provider": "wandb_inference", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if not isinstance(e, (ValidationError, json.JSONDecodeError, ValueError)):
                return None  # network/auth error: don't retry
    return None


# ---------------------------------------------------------------- accounting
def _cost_usd(provider_key: str, usage: dict) -> float | None:
    prices = (config().get("pricing") or {}).get(provider_key) or {}
    if prices.get("input") is None or usage.get("input_tokens") is None:
        return None  # unknown price sheet: report tokens + latency only, never a made-up dollar figure
    return (usage["input_tokens"] * prices["input"] + (usage.get("output_tokens") or 0) * prices["output"]) / 1e6


def accounting(provider_key: str, latency_s: float, usage: dict | None) -> dict:
    usage = usage or {}
    return {"provider": provider_key, "latency_s": round(latency_s, 3),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "usd": _cost_usd(provider_key, usage)}


# ---------------------------------------------------------------- confidence gate
SECOND_OPINION_SYSTEM_TEMPLATE = """You are the second-opinion reviewer in an autonomous ML loop.
The fast structured decision head (TypeSafe Jev) proposed an action with LOW confidence.
Given the diagnosis, its probability vector and the history of what already did/did not help,
either ACCEPT the proposal or OVERRIDE it with one better-supported action.
Output STRICT JSON only: {{"verdict": "accept"|"override", "critique": str, "action": <action object or null>}}
where <action object> uses exactly one of these shapes:
{schema}"""


@weave.op
def second_opinion(diagnosis: dict, proposal: dict, unrevealed_sites: list[str], model_family: str,
                   feature_ops: list[str], tried_families: list[str], history_summary: str) -> dict:
    """Escalation when Jev's confidence is below action_head.confidence_floor. Returns
    {'verdict', 'critique', 'action' (validated envelope dict or None), 'accounting', 'error'}."""
    user = (f"DIAGNOSIS:\n{json.dumps(diagnosis)}\n"
            f"PROPOSAL: {json.dumps(proposal['action'])}\n"
            f"JEV CONFIDENCE {proposal['typesafe']['confidence']:.2f}; PROBABILITIES {json.dumps(proposal['typesafe']['probabilities'])}\n"
            f"CURRENT MODEL: {model_family}; TRIED: {tried_families}; FEATURE OPS: {feature_ops}; "
            f"UNREVEALED SITES: {unrevealed_sites}\nHISTORY:\n{history_summary}\nSTRICT JSON now.")
    t0 = time.time()
    out = {"verdict": "accept", "critique": "", "action": None, "error": None}
    try:
        raw = llm.complete(SECOND_OPINION_SYSTEM_TEMPLATE.format(schema=schema_hint()), user, temperature=0.0)
        parsed = json.loads(llm.extract_json(raw))
        out["verdict"] = parsed.get("verdict", "accept")
        out["critique"] = str(parsed.get("critique", ""))[:600]
        if out["verdict"] == "override":
            action = parsed.get("action") or {}
            if isinstance(action, dict) and "action" in action and "kind" not in action:
                action = action["action"]                     # tolerate a nested envelope
            try:
                out["action"] = _validate(json.dumps({"action": action}), unrevealed_sites).model_dump()
            except Exception as e:                             # keep the critique, drop the bad override
                out["verdict"], out["error"] = "accept", f"override rejected ({type(e).__name__}); kept proposal"
    except Exception as e:
        out["error"] = f"second opinion failed ({type(e).__name__}); kept proposal"
    out["accounting"] = accounting("wandb_inference", time.time() - t0, dict(llm.LAST_USAGE))
    return out


# ---------------------------------------------------------------- the head
@weave.op
def decide_action(diagnosis: dict, unrevealed_sites: list[str], model_family: str,
                  feature_ops: list[str] | None = None, tried_families: list[str] | None = None,
                  history_summary: str = "", offline: bool = False,
                  force_provider: str | None = None, tried_hyperparams: list[str] | None = None) -> dict:
    """Returns {'provider': str, 'action': envelope dict, 'attempts': [...], ['typesafe': {...}]}."""
    feature_ops = feature_ops or []
    tried_families = tried_families or []
    if offline or force_provider == "heuristic":
        t0 = time.time()
        out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
        out["provider"] = "heuristic (dry-run)" if offline else "heuristic"
        out["accounting"] = accounting("heuristic", time.time() - t0, {"input_tokens": 0, "output_tokens": 0})
        return out

    attempts: list[dict] = []
    if force_provider:
        providers = [force_provider]
    else:
        providers = ([("typesafe")] if typesafe_configured() else []) + (["wandb_inference"] if llm.available() else [])

    for prov in providers:
        t0 = time.time()
        if prov == "typesafe":
            try:
                out = typesafe_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families,
                                      history_summary, tried_hyperparams)
                out["attempts"] = attempts
                out["accounting"] = accounting("typesafe", time.time() - t0, out["typesafe"].get("usage"))
                return out
            except Exception as e:
                attempts.append({"provider": "typesafe", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        elif prov == "wandb_inference":
            out = inference_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families,
                                   history_summary, attempts, tried_hyperparams)
            if out:
                if any(a["provider"] == "typesafe" for a in attempts):
                    out["provider"] = "wandb_inference_fallback (typesafe failed)"
                out["attempts"] = attempts
                out["accounting"] = accounting("wandb_inference", time.time() - t0, dict(llm.LAST_USAGE))
                return out

    t0 = time.time()
    out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
    why = "no LLM provider configured" if not providers else "; ".join(a["error"] for a in attempts)
    out["provider"] = f"heuristic_fallback ({why})"
    out["attempts"] = attempts
    out["accounting"] = accounting("heuristic", time.time() - t0, {"input_tokens": 0, "output_tokens": 0})
    return out

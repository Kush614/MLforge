"""Structured action head: diagnosis -> exactly one typed Action.

Provider order (each labeled honestly in the returned 'provider' field):
  1. TypeSafe AI   — when TYPESAFE_API_KEY + TYPESAFE_BASE_URL are set.
                     Request shape is OpenAI-compatible; CONFIRM with their onsite engineer.
  2. W&B Inference — same prompt through llm.py.
  3. heuristic     — deterministic rules (dry-run, or both LLM heads failed).
The action is validated by pydantic before it is returned; free text never reaches act()."""
import json
import os
import weave
import requests
from dotenv import load_dotenv
from pydantic import ValidationError
from .actions import ActionEnvelope, ACTION_SCHEMA_HINT
from .heuristics import heuristic_decide
from . import llm

load_dotenv()

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
    return bool(os.environ.get("TYPESAFE_API_KEY") and os.environ.get("TYPESAFE_BASE_URL"))


@weave.op
def typesafe_complete(system: str, user: str) -> str:
    """Raw TypeSafe call. TODO(event): confirm endpoint + payload with TypeSafe engineer."""
    r = requests.post(
        os.environ["TYPESAFE_BASE_URL"].rstrip("/") + "/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": os.environ.get("TYPESAFE_MODEL", "default"),
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
              "temperature": 0.0},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


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


@weave.op
def decide_action(diagnosis: dict, unrevealed_sites: list[str], model_family: str,
                  feature_ops: list[str] | None = None, tried_families: list[str] | None = None,
                  history_summary: str = "", offline: bool = False,
                  force_provider: str | None = None) -> dict:
    """Returns {'provider': str, 'action': envelope dict, 'attempts': [...]}."""
    feature_ops = feature_ops or []
    tried_families = tried_families or []
    if offline:
        out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
        out["provider"] = "heuristic (dry-run)"
        return out

    user = _user_prompt(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families, history_summary)
    attempts: list[dict] = []
    providers = []
    if force_provider:
        providers = [force_provider]
    else:
        if typesafe_configured():
            providers.append("typesafe")
        if llm.available():
            providers.append("wandb_inference")

    for prov in providers:
        for attempt in range(2):  # one repair retry per provider
            prompt = user if attempt == 0 else user + "\n\nPrevious reply was invalid: " + attempts[-1]["error"] + "\nJSON only."
            try:
                raw = typesafe_complete(DECIDE_SYSTEM, prompt) if prov == "typesafe" \
                    else llm.complete(DECIDE_SYSTEM, prompt, temperature=0.0)
                env = _validate(raw, unrevealed_sites)
                label = prov if prov == "typesafe" or not typesafe_configured() \
                    else "wandb_inference_fallback (typesafe failed)"
                return {"provider": label, "action": env.model_dump(), "attempts": attempts}
            except Exception as e:
                attempts.append({"provider": prov, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                if not isinstance(e, (ValidationError, json.JSONDecodeError, ValueError)):
                    break  # network/auth error: don't retry the same provider

    out = heuristic_decide(diagnosis, unrevealed_sites, model_family, feature_ops, tried_families)
    why = "no LLM provider configured" if not providers else "; ".join(a["error"] for a in attempts)
    out["provider"] = f"heuristic_fallback ({why})"
    out["attempts"] = attempts
    return out

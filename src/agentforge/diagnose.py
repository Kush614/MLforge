"""Diagnosis step: LLM reasons over the current eval + the agent's own history.

The history dict comes from history.build_history(); its 'source' field is carried
into the Diagnosis so the trace and the lab report say where the memory came from."""
import json
import weave
from pydantic import BaseModel, Field, ValidationError
from . import llm
from .heuristics import heuristic_diagnose

EVIDENCE_TAGS = ["underfitting", "overfitting", "data_starved", "feature_scale",
                 "class_imbalance", "plateau"]


class Diagnosis(BaseModel):
    reasoning: str
    evidence_tags: list[str] = Field(default_factory=list)
    cited_trace_ids: list[str] = Field(default_factory=list)
    history_source: str = "local"      # "weave_api" | "local" | "heuristic"
    provider: str = "wandb_inference"  # who produced the reasoning


SYSTEM = f"""You are the diagnosis module of an autonomous ML improvement loop
(binary heart-disease classification, tabular, small data, multi-site).
You receive the current evaluation result and an honest history of past actions and
their measured effects. Diagnose WHY the model is underperforming using the evidence.
Rules:
- Cite specific numbers from the evaluation (gap, per-class recall, n_train, learning curve).
- Consider what the history shows already did NOT help; do not repeat failed strategies.
- If the history contains [trace ...] ids, list the ones you relied on in cited_trace_ids.
- Output STRICT JSON only:
  {{"reasoning": str, "evidence_tags": [subset of {EVIDENCE_TAGS}], "cited_trace_ids": [str]}}
No markdown, no preamble."""


def _parse(raw: str, history_source: str) -> Diagnosis:
    parsed = json.loads(llm.extract_json(raw))
    return Diagnosis(
        reasoning=str(parsed.get("reasoning", "")).strip(),
        evidence_tags=[t for t in parsed.get("evidence_tags", []) if t in EVIDENCE_TAGS],
        cited_trace_ids=[str(t) for t in parsed.get("cited_trace_ids", [])][:10],
        history_source=history_source,
    )


@weave.op
def diagnose(eval_result: dict, history: dict, offline: bool = False) -> dict:
    """history = {'source': ..., 'summary': ..., 'trace_ids': [...]}. Returns Diagnosis dict."""
    src = history.get("source", "local")
    summary = history.get("summary", "No prior iterations.")
    if offline or not llm.available():
        d = heuristic_diagnose(eval_result, summary)
        d["provider"] = "heuristic (dry-run)" if offline else "heuristic (WANDB_API_KEY not set)"
        return d
    user = (
        f"CURRENT EVALUATION:\n{json.dumps(eval_result, indent=2)}\n\n"
        f"HISTORY (source: {src}):\n{summary}\n\n"
        "Diagnose now. STRICT JSON."
    )
    last_err = None
    for attempt in range(2):
        try:
            raw = llm.complete(SYSTEM, user if attempt == 0 else
                               user + "\n\nYour previous reply was not valid JSON. Reply with JSON only.")
            d = _parse(raw, src)
            d.provider = "wandb_inference"
            return d.model_dump()
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_err = e
        except Exception as e:  # network / auth: do not retry blindly
            last_err = e
            break
    d = heuristic_diagnose(eval_result, summary)
    d["provider"] = f"heuristic_fallback (wandb_inference error: {type(last_err).__name__})"
    return d

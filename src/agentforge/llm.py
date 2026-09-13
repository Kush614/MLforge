"""W&B Inference client (OpenAI-compatible). Diagnosis brain of the loop.

If the call fails, the exception propagates: callers decide whether to fall back,
and every fallback is labeled in the trace (CLAUDE.md rule 2)."""
import os
import weave
from openai import OpenAI
from dotenv import load_dotenv
from .tracing import config

load_dotenv()

_client_cache: OpenAI | None = None


def available() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))


def _client() -> OpenAI:
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(
            base_url=os.environ.get("WANDB_INFERENCE_BASE_URL", "https://api.inference.wandb.ai/v1"),
            api_key=os.environ["WANDB_API_KEY"],
            # W&B Inference bills to a project: entity/project. Entity from env if given.
            default_headers={"OpenAI-Project": _project_header()},
            timeout=60,
        )
    return _client_cache


def _project_header() -> str:
    proj = config()["wandb_project"]
    ent = os.environ.get("WANDB_ENTITY")
    return f"{ent}/{proj}" if ent and "/" not in proj else proj


@weave.op
def complete(system: str, user: str, temperature: float | None = None,
             model: str | None = None) -> str:
    cfg = config()["llm"]
    resp = _client().chat.completions.create(
        model=model or cfg["diagnosis_model"],
        temperature=temperature if temperature is not None else cfg["temperature"],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def extract_json(raw: str) -> str:
    """Strip code fences / preamble; return the outermost {...} block."""
    s = raw.replace("```json", "").replace("```", "").strip()
    start, end = s.find("{"), s.rfind("}")
    return s[start:end + 1] if start != -1 and end > start else s

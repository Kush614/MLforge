"""P4: Weave comparison eval — TypeSafe vs W&B Inference vs heuristic as the action head.

Replays recorded diagnoses (runs/<run>.jsonl, written by the loop) through each
provider with weave.Evaluation, so the comparison shows up side by side in Weave.

    python scripts/compare_action_heads.py [--run runs/latest.jsonl] [--limit 5]

Scorers:
  valid_typed_action  the provider itself produced a pydantic-valid action (no fallback)
  agrees_with_run     same action kind as the one the live loop actually took
  accuracy_delta      COUNTERFACTUAL: apply the proposed action to the recorded state,
                      retrain on the same fixed split, report new_acc - recorded_acc.
                      This is the measured answer to "did its decision help?"
  latency_s           wall-clock seconds for the decision
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import weave  # noqa: E402
from agentforge.tracing import init_tracing  # noqa: E402
from agentforge.typesafe_client import decide_action, typesafe_configured  # noqa: E402
from agentforge import llm  # noqa: E402
from agentforge.tracing import config  # noqa: E402
from agentforge.state import LoopState  # noqa: E402
from agentforge.act import act  # noqa: E402
from agentforge.train import train  # noqa: E402
from agentforge.evaluate import evaluate  # noqa: E402
from agentforge.loop import split_data  # noqa: E402


class ActionHead(weave.Model):
    provider: str

    @weave.op
    def predict(self, diagnosis: dict, decision_context: dict) -> dict:
        t0 = time.time()
        out = decide_action(diagnosis, decision_context["unrevealed_sites"],
                            decision_context["model_family"], decision_context["feature_ops"],
                            decision_context["tried_families"],
                            history_summary=decision_context.get("history_summary", ""),
                            force_provider=self.provider,
                            tried_hyperparams=decision_context.get("tried_hyperparams"))
        out["latency_s"] = time.time() - t0
        if "fallback" in out["provider"]:   # never let a silent fallback masquerade as the provider
            print(f"  !! {self.provider} fell back: {out.get('attempts')}", flush=True)
        return out


@weave.op
def valid_typed_action(output: dict) -> bool:
    return "fallback" not in output["provider"]


@weave.op
def agrees_with_run(action: str, output: dict) -> bool:
    return action.split("(")[0] == output["action"]["action"]["kind"]


@weave.op
def accuracy_delta(decision_context: dict, accuracy: float, output: dict) -> dict:
    """Replay: recorded state + proposed action -> retrain -> held-out delta vs. recorded accuracy."""
    cfg = config()
    all_sites = cfg["dataset"]["sites"]
    unrevealed = list(decision_context["unrevealed_sites"])
    state = LoopState(sites=[s for s in all_sites if s not in unrevealed], unrevealed_sites=unrevealed,
                      model_family=decision_context["model_family"],
                      feature_ops=list(decision_context["feature_ops"]),
                      hyperparams=dict(decision_context.get("hyperparams") or {}),
                      tried_families=list(decision_context.get("tried_families", [])))
    kind = output["action"]["action"]["kind"]
    try:
        desc = act(output["action"], state)
    except ValueError as e:                      # e.g. site not acquirable in that state
        return {"delta": None, "new_accuracy": None, "applied": False, "note": str(e)}
    if kind == "request_code_fix":               # no measurable effect without a human/ARIA
        return {"delta": 0.0, "new_accuracy": accuracy, "applied": True, "note": "code fix request; no retrain"}
    X_tr, X_te, y_tr, y_te, _ = split_data(state, cfg)
    ev = evaluate(train(state, X_tr, y_tr), X_tr, y_tr, X_te, y_te)
    return {"delta": ev["accuracy"] - accuracy, "new_accuracy": ev["accuracy"], "applied": True, "note": desc}


@weave.op
def latency_s(output: dict) -> float:
    return output["latency_s"]


@weave.op
def tokens(output: dict) -> int | None:
    a = output.get("accounting") or {}
    if a.get("input_tokens") is None:
        return None
    return (a["input_tokens"] or 0) + (a.get("output_tokens") or 0)


@weave.op
def usd(output: dict) -> float | None:
    return (output.get("accounting") or {}).get("usd")


def load_examples(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ex = [{"diagnosis": r["diagnosis"], "decision_context": r["decision_context"],
           "action": r["action"], "accuracy": r["accuracy"]}
          for r in rows if r.get("diagnosis") and r.get("decision_context")]
    return ex[:limit]


async def run(examples, providers):
    ev = weave.Evaluation(name="action_head_comparison", dataset=examples,
                          scorers=[valid_typed_action, agrees_with_run, accuracy_delta, latency_s, tokens, usd])
    results = {}
    for prov in providers:
        print(f"== evaluating provider: {prov}")
        results[prov] = await ev.evaluate(ActionHead(provider=prov, name=f"head_{prov}"))
    print_leaderboard(results, len(examples))
    return results


def _get(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def print_leaderboard(results: dict, n: int):
    from rich.console import Console
    from rich.table import Table
    t = Table(title=f"action-head leaderboard ({n} recorded diagnoses; same fixed split; counterfactual retrain)")
    for c in ("head", "valid action", "agrees w/ run", "mean Δacc", "mean latency", "mean tokens", "mean $"):
        t.add_column(c)
    rows = sorted(results.items(), key=lambda kv: -(_get(kv[1], "accuracy_delta", "delta", "mean") or -9))
    for prov, r in rows:
        d = _get(r, "accuracy_delta", "delta", "mean")
        tok = _get(r, "tokens", "mean")
        cost = _get(r, "usd", "mean")
        t.add_row(prov,
                  f"{_get(r, 'valid_typed_action', 'true_count')}/{n}",
                  f"{_get(r, 'agrees_with_run', 'true_count')}/{n}",
                  f"{d:+.3f}" if d is not None else "n/a",
                  f"{_get(r, 'latency_s', 'mean'):.2f} s",
                  f"{tok:.0f}" if tok is not None else "n/a",
                  f"${cost:.5f}" if cost is not None else "n/a (no price sheet)")
    Console().print(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "latest.jsonl"))
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    ts = init_tracing()
    print(f"tracing: {ts['mode']}")
    examples = load_examples(Path(args.run), args.limit)
    if not examples:
        sys.exit(f"no recorded diagnoses in {args.run}; run the loop first")
    providers = []
    if typesafe_configured():
        providers.append("typesafe")
    else:
        print("TypeSafe not configured (TYPESAFE_API_KEY/TYPESAFE_BASE_URL) — skipping, not faking.")
    if llm.available():
        providers.append("wandb_inference")
    providers.append("heuristic")      # the rule-based baseline every LLM head must beat
    asyncio.run(run(examples, providers))


if __name__ == "__main__":
    main()

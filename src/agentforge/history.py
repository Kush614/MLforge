"""The agent's memory: its own Weave trace history.

Source preference (always labeled in the returned dict, CLAUDE.md rule 2):
  1. "weave_api"  — query this project's past `evaluate` / `act` calls through the Weave
                    client. This is the same store the hosted W&B MCP server reads.
  2. "local"      — LoopState.history mirror (offline / dry-run / query failed).

A run's own calls arrive in Weave asynchronously, so the local mirror is always
merged in: nothing the loop just did is ever missing from the prompt."""
import weave
from .tracing import tracing_state


def _short(op_name: str) -> str:
    # weave:///entity/project/op/evaluate:hash -> evaluate
    return op_name.rsplit("/", 1)[-1].split(":")[0]


@weave.op
def fetch_weave_history(limit: int = 40) -> dict:
    """Pull recent evaluate/act calls from Weave. Returns {'ok', 'lines', 'trace_ids', 'error'}."""
    ts = tracing_state()
    client = ts.get("client")
    if ts.get("mode") != "online" or client is None:
        return {"ok": False, "lines": [], "trace_ids": [], "error": f"tracing mode={ts.get('mode')}"}
    try:
        calls = client.get_calls(limit=limit * 4, sort_by=[{"field": "started_at", "direction": "desc"}])
        lines, ids = [], []
        for c in calls:
            name = _short(c.op_name or "")
            if name == "evaluate" and isinstance(c.output, dict) and "accuracy" in c.output:
                o = c.output
                lines.append(f"[trace {c.id}] evaluate: acc={o['accuracy']:.3f} "
                             f"gap={o['train_test_gap']:+.3f} n_train={o['n_train']}")
                ids.append(c.id)
            elif name == "act" and c.output:
                lines.append(f"[trace {c.id}] act: {c.output}")
                ids.append(c.id)
            if len(lines) >= limit:
                break
        lines.reverse()  # chronological
        return {"ok": True, "lines": lines, "trace_ids": ids, "error": None}
    except Exception as e:
        return {"ok": False, "lines": [], "trace_ids": [], "error": f"{type(e).__name__}: {e}"}


@weave.op
def build_history(state, use_weave: bool = True) -> dict:
    """Merge Weave-sourced history with the local mirror. Returns {'source', 'summary', 'trace_ids', 'note'}."""
    local = state.history_summary()
    if not use_weave:
        return {"source": "local", "summary": local, "trace_ids": [], "note": "weave lookup disabled"}
    w = fetch_weave_history()
    if w["ok"] and w["lines"]:
        summary = ("PAST WEAVE TRACES (oldest first):\n" + "\n".join(w["lines"])
                   + "\n\nEFFECT LEDGER (this run, local mirror):\n" + local)
        return {"source": "weave_api", "summary": summary, "trace_ids": w["trace_ids"], "note": None}
    note = w["error"] or "no matching traces yet"
    return {"source": "local", "summary": local, "trace_ids": [],
            "note": f"fallback to local mirror ({note})"}

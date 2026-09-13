"""ARIA handoff: when the agent decides its own CODE is the problem, it writes a
structured fix request that ARIA (W&B's in-app coding agent) implements.
See docs/ARIA.md. Semi-automated is fine; the trace labels what was automated.

Protocol (file-based, so it works with whatever surface ARIA exposes on the day):
  aria_requests/NNN.md        <- written by the loop (emit_fix_request)
  aria_requests/NNN.applied   <- written by ARIA / the human applying ARIA's diff,
                                 containing a one-line note (e.g. "ARIA applied diff X",
                                 "manual-assisted"). check_for_patches() picks it up,
                                 re-runs the acceptance test, and records the outcome.
"""
import importlib
import json
import subprocess
import sys
import time
import weave
from pathlib import Path

from .tracing import ROOT
from . import extensions

REQUESTS_DIR = ROOT / "aria_requests"


def _next_id() -> int:
    REQUESTS_DIR.mkdir(exist_ok=True)
    return len(list(REQUESTS_DIR.glob("*.md"))) + 1


@weave.op
def emit_fix_request(description: str, failing_trace_ids: list[str],
                     module_hint: str = "src/agentforge/extensions.py") -> str:
    n = _next_id()
    path = REQUESTS_DIR / f"{n:03d}.md"
    traces = "\n".join(f"- {t}" for t in failing_trace_ids) or "- (none cited)"
    path.write_text(
        f"# Fix request {n:03d}\n\n"
        f"**Emitted by the loop at** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## What the agent diagnosed\n{description}\n\n"
        f"## Failing Weave traces\n{traces}\n\n"
        f"## Module to change\n`{module_hint}` — append below the ARIA marker using "
        f"`register_model(name, factory, description, grid)` or `register_feature_op(name, factory, description)`. "
        f"Do not edit other modules; keep every `@weave.op`.\n\n"
        f"## Current action space\n```json\n{json.dumps(extensions.snapshot())}\n```\n\n"
        f"## Acceptance\nRe-run `pytest tests/ -x` and one loop iteration; "
        f"the targeted metric must improve or the failure mode must disappear.\n\n"
        f"## How to close this request\nWrite `aria_requests/{n:03d}.applied` containing one line: "
        f"who applied it (`aria` or `manual-assisted`) and a short note.\n"
    )
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@weave.op
def run_acceptance_test() -> dict:
    """The acceptance test ARIA's fix must pass. Returns {'passed', 'output'}."""
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    return {"passed": proc.returncode == 0, "output": (proc.stdout + proc.stderr)[-2000:]}


@weave.op
def check_for_patches() -> list[dict]:
    """Detect newly applied ARIA fixes (NNN.applied without NNN.recorded.json).
    Records honestly who applied them and whether the acceptance test passed."""
    results = []
    if not REQUESTS_DIR.exists():
        return results
    for applied in sorted(REQUESTS_DIR.glob("*.applied")):
        marker = applied.with_suffix(".recorded.json")
        if marker.exists():
            continue
        note = applied.read_text().strip()
        how = "aria" if note.lower().startswith("aria") else "manual-assisted"
        before = extensions.snapshot()
        try:
            importlib.reload(extensions)          # pick up newly registered options
            reload_error = None
        except Exception as e:
            reload_error = f"{type(e).__name__}: {e}"
        after = extensions.snapshot()
        test = run_acceptance_test()
        rec = {"request": applied.stem, "applied_by": how, "note": note,
               "acceptance_passed": test["passed"], "recorded_at": time.time(),
               "reload_error": reload_error,
               "action_space_added": {"models": [m for m in after["models"] if m not in before["models"]],
                                      "feature_ops": [o for o in after["feature_ops"] if o not in before["feature_ops"]]}}
        marker.write_text(json.dumps(rec, indent=2))
        results.append(rec)
    return results

"""Loop state: what the agent currently has, and an honest record of what it tried."""
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .tracing import ROOT

import os
RUNS_DIR = Path(os.environ.get("AGENTFORGE_RUNS_DIR") or ROOT / "runs")   # tests point this elsewhere


@dataclass
class EffectRecord:
    iteration: int            # iteration in which the action was TAKEN
    action_desc: str
    accuracy_before: float    # accuracy measured right before the action
    accuracy_after: float     # accuracy measured on the next iteration
    trace_id: str | None = None

    @property
    def delta(self) -> float:
        return self.accuracy_after - self.accuracy_before


@dataclass
class LoopState:
    sites: list[str]                       # sites the agent currently has
    unrevealed_sites: list[str]            # sites it may acquire
    model_family: str = "knn"
    feature_ops: list[str] = field(default_factory=list)
    hyperparams: dict = field(default_factory=dict)
    history: list[EffectRecord] = field(default_factory=list)
    iteration: int = 0
    # action taken at the end of the previous iteration, whose effect is not yet measured
    pending_action: str | None = None
    pending_trace_id: str | None = None
    pending_accuracy: float | None = None
    tried_families: list[str] = field(default_factory=list)
    tried_hyperparams: dict = field(default_factory=dict)   # family -> [json signature of params applied]

    # ---- effect bookkeeping -------------------------------------------------
    def record_action(self, desc: str, accuracy_before: float, trace_id: str | None = None):
        self.pending_action = desc
        self.pending_accuracy = accuracy_before
        self.pending_trace_id = trace_id

    def settle_effect(self, accuracy_now: float) -> EffectRecord | None:
        """Called after evaluation: attribute the accuracy change to the pending action."""
        if self.pending_action is None:
            return None
        rec = EffectRecord(self.iteration - 1, self.pending_action,
                           self.pending_accuracy, accuracy_now, self.pending_trace_id)
        self.history.append(rec)
        self.pending_action = self.pending_accuracy = self.pending_trace_id = None
        return rec

    def plateaued(self, n: int = 3) -> bool:
        return len(self.history) >= n and all(r.delta <= 0 for r in self.history[-n:])

    # ---- summaries ------------------------------------------------------------
    def history_summary(self) -> str:
        """Compact honest history for the diagnosis prompt (local mirror of traces)."""
        if not self.history:
            return "No prior iterations."
        lines = []
        for r in self.history:
            tid = f" [trace {r.trace_id}]" if r.trace_id else ""
            lines.append(
                f"iter {r.iteration}: {r.action_desc} -> "
                f"{r.accuracy_before:.3f} => {r.accuracy_after:.3f} (delta {r.delta:+.3f}){tid}"
            )
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "iteration": self.iteration,
            "sites": list(self.sites),
            "unrevealed_sites": list(self.unrevealed_sites),
            "model_family": self.model_family,
            "feature_ops": list(self.feature_ops),
            "hyperparams": dict(self.hyperparams),
            "n_history": len(self.history),
            "tried_hyperparams": {k: list(v) for k, v in self.tried_hyperparams.items()},
        }


def log_metrics(run_name: str, row: dict) -> Path:
    """Append one JSON line per iteration to runs/<run_name>.jsonl (read by the lab report)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = RUNS_DIR / f"{run_name}.jsonl"
    with p.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **row}) + "\n")
    # also refresh a stable pointer the notebook can always read
    (RUNS_DIR / "latest.jsonl").write_text(p.read_text())
    return p

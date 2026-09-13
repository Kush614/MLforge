# ARCHITECTURE

## Modules (src/agentforge/)
- loop.py — orchestrator. Owns LoopState, iteration budget, stop conditions. CLI:
  python -m agentforge.loop [--dry-run] [--iterations N]
- state.py — LoopState: current sites, feature pipeline ops, model family,
  history of (action, before, after) tuples.
- data.py — per-site CSVs; fixed_split() holds out a test split from EVERY site once,
  assemble() builds the train pool from revealed sites only. Synthetic stand-in ONLY
  when data/ is absent and the caller allows it (dry-run/tests), labeled as such.
- train.py — sklearn model zoo behind one fit(state) -> model function.
- evaluate.py — EvalResult (accuracy, per-class PRF, train/test gap, learning-curve
  point, worst-K examples). Registers a weave.Evaluation each iteration.
- history.py — the agent's memory: fetch_weave_history() reads past evaluate/act calls
  from Weave (source 'weave_api'); build_history() merges with the local effect
  ledger and falls back to 'local' with the reason.
- diagnose.py — EvalResult + history -> Diagnosis(reasoning, evidence_tags,
  cited_trace_ids, history_source, provider). JSON repair retry, labeled fallback.
- heuristics.py — rule-based diagnose/decide used by --dry-run and as last-resort
  fallback; always labeled 'heuristic'.
- actions.py — pydantic Action union (see SPEC section 5).
- act.py — validated executor; measures effect size; refuses unknown actions.
- llm.py — OpenAI-compatible client pointed at W&B Inference.
- typesafe_client.py — structured-decision head: TypeSafe System One (fan-out of Choice
  questions, code assembles the action) -> W&B Inference (JSON + repair) -> heuristic,
  each attempt recorded; pydantic validation + precondition check before returning.
- aria_handoff.py — writes aria_requests/NNN.md; check_for_patches() detects NNN.applied,
  re-runs the acceptance test, records applied_by (aria | manual-assisted) + result.
- notebook_writer.py — appends marimo cells (marimo files are plain .py).
- tracing.py — weave.init (online / offline, mode reported) + config(); current_call_id().
- state.py — LoopState + effect attribution (record_action / settle_effect) + runs/*.jsonl
  metrics log read by the notebook chart.
- scripts/compare_action_heads.py — P4 weave.Evaluation: TypeSafe vs Inference on
  recorded diagnoses.

## Data flow per iteration
state -> train -> model -> evaluate -> EvalResult
history summary (Weave/MCP) + EvalResult -> diagnose -> Diagnosis
Diagnosis -> (TypeSafe or Inference) -> Action -> act -> new state + effect record
-> notebook_writer.append(iteration)

## Honesty invariants
- A sponsor tool appears in logs/report ONLY if its call succeeded.
- Fallbacks are labeled as fallbacks in the trace.
- Effect sizes are computed from the same seeded split, never cherry-picked.

# PLAN — build order & timeboxes (Sun, doors 9 AM -> submit 1 PM)

Work in priority order. Each phase leaves the repo demo-able. Cut from the bottom.

## P0 — Harness + tracing (9:00-10:00)  [prize-eligibility floor]
- scripts/fetch_data.py pulls UCI heart data, splits into 4 sites.
- train.py, evaluate.py working for knn + random_forest.
- weave.init + @weave.op everywhere; first eval visible in Weave UI.
- ALSO BY 10:00: every teammate signed into AGI House platform + survey done.

## P1 — Close the loop (10:00-11:15)
- llm.py W&B Inference client. diagnose.py with evidence-tag prompt.
- MCP trace-history summary into the diagnosis prompt (fallback: local history dict
  if MCP wiring stalls — still honest, still works, label it as fallback).
- actions.py + act.py executor for switch_model / transform_features /
  acquire_data / tune_hyperparams. Full loop runs autonomously.

## P2 — Self-writing lab report (11:15-11:45)
- notebook_writer.py appends a section per iteration to notebooks/lab_report.py.
- Run training in molab if useful (not strictly needed — nice-to-have).

## P3 — ARIA handoff (11:45-12:20)
- aria_handoff.py: emit structured fix request; execute one live fix via ARIA
  workspace with sponsor-engineer help. Record it in Weave + lab report.

## P4 — TypeSafe action head (12:20-12:40)
- typesafe_client.py; Weave comparison eval TypeSafe vs Inference on 5 recorded
  diagnoses.

## P5 — Submission package (12:40-1:00)  [hard deadline]
- 2-min screen recording; AGI House submission (name, 2-3 sentence description,
  public repo link, track = Best Use of Weave, demo video, all members listed,
  full sponsor-tool list).

## Demo rehearsal (before 1:30 judging)
Run docs/DEMO_SCRIPT.md twice against a stopwatch.

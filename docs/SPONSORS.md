# Sponsor integrations — what is real, where it lives, what to say to judges

Track pick: **Best Use of Weave** (deepest). List all five tools with these roles in the
submission description; sponsor prizes are judged separately on "meaningful use".
Everything below is verified against the code as of the first live run; the
"status" column is the honest state to give in Q&A.

| Sponsor | Role in the loop | Code | Status |
|---|---|---|---|
| **Weave** | Every op traced; **the agent's memory** — diagnosis reads past `evaluate`/`act` traces and cites their IDs | `tracing.py`, `history.py`, every `@weave.op` | LIVE. History source `weave_api` from iteration 1 on. Hosted W&B MCP server not wired (Weave client API used instead — same store). |
| **W&B Inference** | Reasoning brain: diagnosis (`gpt-oss-120b`) + fallback action head | `llm.py`, `diagnose.py` | LIVE. |
| **TypeSafe AI** | **Action head**: System One (Jev) answers typed `Choice` questions over the diagnosis state → code assembles one pydantic-validated `Action`; confidence + probabilities in the trace | `typesafe_client.py`, `scripts/compare_action_heads.py` | LIVE (`jev-1.13.0`). Drove a full run to 0.859 in 8 iterations. Comparison eval in Weave: see below. |
| **ARIA** | Code-fixing arm: `request_code_fix` → `aria_requests/NNN.md` → `NNN.applied` → acceptance test re-run → recorded | `aria_handoff.py`, `docs/ARIA.md` | Protocol implemented + tested; no live ARIA fix executed yet. |
| **marimo** | Self-writing lab report: one cell per iteration + live chart vs. target | `notebook_writer.py`, `notebooks/lab_report.py` | LIVE. molab optional (models are small). |

## Weave — "its memory is its own trace history"
- `train`, `evaluate`, `build_history`, `diagnose`, `decide_action`, `act`, `emit_fix_request`,
  `check_for_patches`, `append_iteration`, `run_iteration` are all `@weave.op`: one iteration
  is one navigable trace tree.
- `history.fetch_weave_history()` queries the project's past calls via `client.get_calls`,
  turns `evaluate`/`act` outputs into `[trace <id>] ...` lines, and the diagnosis prompt
  asks the LLM to list the trace IDs it relied on → `cited_trace_ids` in the trace and
  in the lab report. A judge can click any cited ID.
- Fallback to the local effect ledger is labeled `history: local — fallback (...)`.
- `scripts/compare_action_heads.py` is a `weave.Evaluation` (see TypeSafe).

Demo line: *"Here it saw a 0.29 train/test gap, called it overfitting, and cited the two
traces where model-switching already failed — so it acquired the Hungary site instead."*

## W&B Inference — the reasoning brain
- OpenAI-compatible client, `OpenAI-Project: <entity>/<project>` header (required —
  an entity-less header returns 400 "Invalid Authentication").
- Diagnosis returns strict JSON `{reasoning, evidence_tags, cited_trace_ids}`; one
  JSON-repair retry; network errors fall through to the heuristic brain, labeled
  `heuristic_fallback (wandb_inference error: <Type>)`.

## TypeSafe AI — the action head (measured, not decorative)
- Two LLM jobs need different strengths: free-text reasoning (W&B Inference) vs. a strict
  structured decision (TypeSafe). Jev does not generate text at all — it returns typed
  answers with calibrated probabilities — so there is nothing to parse and nothing to repair.
- Implementation (`typesafe_decide`): one `POST /v1/systemone` with the diagnosis, the eval
  numbers, current setup and the effect history as `state`, and a fan-out of independent
  `Choice` questions: `action_kind` plus speculative branches `model_family`, `feature_op`,
  `site`, `hyperparam_preset`. Only legal options are offered (no `acquire_data` when no
  sites remain; already-tried families are marked). Code consumes the chosen branch and
  builds the `ActionEnvelope`; hyperparameters come from named presets mapped to grid
  values in code (Jev picks the intent, code owns the numbers).
- Trace + lab report carry Jev's confidence and the full probability vector, e.g.
  `acquire_data 0.81, tune_hyperparams 0.11, switch_model 0.07`. Confidence visibly drops
  (0.29–0.45) when the agent is stuck — an honest uncertainty signal judges can see.
- Provider chain in `decide_action()`: `typesafe` → `wandb_inference` → `heuristic`,
  every attempt recorded, provider label carried into the trace + lab report.
- **Confidence gate — the agent knows when it doesn't know.** `action_head.confidence_floor`
  (0.40): when Jev's confidence on `action_kind` is below it, the loop escalates to a second
  opinion (W&B Inference critiques the proposal against the effect history and may override
  with one validated action). Trace + report show floor, confidence, verdict, critique. In the
  adversarial run Jev's confidence fell 0.89 → 0.26 the decision after the poisoned site hurt.
- **Memory is binding, not advisory.** Hyperparameter settings already applied to a family
  are withdrawn from the options Jev sees (and `tune_hyperparams` disappears when none remain);
  tried families/ops are marked. This stopped a +0.114/−0.114 preset oscillation.
- **Per-decision accounting**: latency, input/output tokens, and `$` only if `pricing:` in
  `config.yaml` is filled (it is null → the report says "n/a (no price sheet)", never a guess).
- **Comparison eval / leaderboard** (`make compare`, a `weave.Evaluation` over the run's
  recorded diagnoses; scorers `valid_typed_action`, `agrees_with_run`, `latency_s`, `tokens`,
  `usd`, and `accuracy_delta` = apply the proposed action to the recorded state, retrain on
  the same fixed split, measure the held-out delta). Three heads, live result on the 9
  diagnoses of the adversarial run:

  | head | valid typed action | agrees w/ run | mean counterfactual Δacc | mean latency | mean tokens |
  |---|---|---|---|---|---|
  | heuristic (rules) | 9/9 | 5/9 | +0.054 | 0.00 s | 0 |
  | **TypeSafe Jev 1.13** | 9/9 | 8/9 | +0.046 | **0.35 s** | 2469 |
  | W&B Inference gpt-oss-120b | 9/9 | 5/9 | +0.038 | 7.19 s | 2482 |

  Honest read: n=9, differences in Δ are within noise; the rule baseline is competitive on a
  toy dataset. What is not noise: Jev is ~20× faster than the reasoning model at equal token
  cost and produces a probability on every option. The script prints `!! fell back` if any
  head silently degrades to the heuristic, so a fallback can never pose as a provider.

Demo line: *"The reasoning model explains; the TypeSafe model decides — in 300 ms, with a
probability on every option — and when its confidence drops, it asks for a second opinion."*

## ARIA — the code-fixing arm
- `request_code_fix` is the escape hatch when the four data/model actions can't express
  the fix. `emit_fix_request()` writes diagnosis, failing trace IDs, target module, and
  acceptance test to `aria_requests/NNN.md`.
- Close the loop by writing `aria_requests/NNN.applied` (`aria ...` or `manual-assisted ...`);
  the next iteration re-runs `pytest`, records `applied_by` + `acceptance_passed` in Weave
  and the lab report. Manual assistance is labeled, never hidden.
- **Self-extending action space.** The fix request now asks ARIA to `register_model(...)` or
  `register_feature_op(...)` in `src/agentforge/extensions.py`. When `NNN.applied` lands the
  loop reloads the module, logs `action_space_added`, and the new family/op appears in Jev's
  questions, the Inference schema and the heuristic on the next iteration. The agent expands
  its own action space; the trace says who expanded it.
- Hit the ARIA table first: one live fix, even semi-automated, is the "Should" in SPEC §6.

## marimo — the self-writing lab report
- Notebook is plain `.py`; `append_iteration()` inserts an idempotent `_iter_N` cell with
  accuracy/gap/learning curve, effect of the previous action, diagnosis in the agent's
  words, evidence tags, cited trace IDs, decision provider.
- Header cell charts accuracy + train accuracy vs. the target and the 0.787 published
  baseline from `runs/latest.jsonl`, auto-refreshing every 5 s, with Jev's confidence on
  each point. `make report` opens it; `make reset` clears it.
- **Cockpit — the report is where the human steers.** A slider (target accuracy) and a
  multiselect (freeze sites) write `runs/controls.json`; the loop reads it at the top of every
  iteration and records "target overridden / sites frozen by human via lab report" in the
  trace, metrics and the report itself. marimo is the control surface, not just the log.

## Adversarial reveal (`--poison va`)
`python -m agentforge.loop --poison va` flips 40% of VA's TRAIN labels (test rows untouched;
documented in the run header, every metrics row and every report section). The agent is not
told. Live result: acquiring VA scored Δ −0.038 in the effect ledger and Jev's confidence
collapsed on the next decision; the run plateaued at 0.83 and stopped honestly — "more data
hurt, and its own traces show it". Without `--poison` the same loop beats 0.85.

## Q&A honesty sheet
- Stubbed vs real: ARIA live fix (protocol built, not yet executed), hosted MCP (not wired;
  Weave API used). Weave, Inference, TypeSafe and marimo all ran live.
- Metric gaming: fixed stratified test split across all four sites from iteration 0;
  acquiring a site only grows the train pool; same seed every run.
- Plateau: 3 consecutive non-positive deltas stop the loop and say so.

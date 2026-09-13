# SPEC — AgentForge

## 1. One-liner
An autonomous agent that closes the ML iteration loop: it trains, evaluates, diagnoses
its own failures from real statistical + trace signal, and fixes itself — until it beats
a published heart-disease benchmark, with no human in the loop.

## 2. Problem
The slow, human part of applied ML is diagnosis: staring at metrics and guessing whether
you need more data, different features, or a different model. We make that step an
agent's job, and make every decision observable.

## 3. Scope

### In scope (the weekend)
- Tabular classification on UCI Heart Disease, multi-site (4 hospitals' worth of data),
  agent starts on the weakest site only.
- Closed loop: TRAIN -> EVALUATE -> DIAGNOSE -> ACT, up to max_iterations.
- Five action types (see section 5). All typed, all traced, all logged honestly.
- Weave as the agent's memory: diagnosis queries its own past traces/evals via W&B MCP.
- Self-writing marimo lab report (one section appended per iteration).
- ARIA handoff path for code-level fixes (semi-automated is acceptable).
- TypeSafe structured action head + Weave comparison eval vs W&B Inference.
- Minimal terminal/rich live view. (A web dashboard is a stretch goal, not core.)

### Out of scope
- Real patient data, PHI, or clinical claims. UCI data is public and anonymized.
- Production deployment, auth, multi-user anything.
- RL training of the agent itself (the self-improvement is the loop, not weight updates).
- Any code written before the hackathon.

## 4. The loop (functional spec)

Each iteration i:
1. TRAIN: train(state) -> Model. Fits current model family on current data with
   current feature pipeline. Traced with dataset fingerprint + hyperparams.
2. EVALUATE: evaluate(model, state) -> EvalResult. Held-out accuracy, per-class
   precision/recall, train-vs-test gap, learning-curve point, per-example errors.
   Logged as a weave.Evaluation.
3. DIAGNOSE: diagnose(eval_result, history) -> Diagnosis. LLM (W&B Inference)
   receives current EvalResult + a summary of ALL past iterations pulled from Weave
   (via MCP tools: what was tried, what changed, which examples keep failing).
   Returns free-text reasoning + evidence tags: underfitting | overfitting |
   data_starved | feature_scale | class_imbalance | plateau.
4. ACT: act(diagnosis) -> Action. TypeSafe model (fallback: W&B Inference)
   converts diagnosis into ONE typed Action. Executor applies it and records
   before/after so effect size is measurable next iteration.
5. REPORT: append iteration section to the marimo lab report.

Stop when accuracy >= target, or no action has helped for 3 consecutive iterations,
or max_iterations reached.

## 5. Action space (typed, pydantic)
| Action | Params | Precondition |
|---|---|---|
| switch_model | family | family in config.models |
| transform_features | op in {standardize, impute_median, onehot, log_scale} | - |
| acquire_data | site | unrevealed sites remain |
| tune_hyperparams | small grid per family | - |
| request_code_fix | description, failing_trace_ids | routed to ARIA (docs/ARIA.md) |

## 6. Success criteria
- Must: loop runs 5+ autonomous iterations end-to-end live; final accuracy beats
  the baseline (78.7%); every decision visible in Weave; lab report readable by judges.
- Should: beats best-published (83.3%); at least one live ARIA-executed fix; TypeSafe-
  vs-Inference comparison eval in Weave.
- Demo test: a judge can pick any decision the agent made and we can show, in one
  click, the trace + the evidence it used.

## 7. Non-functional
- One iteration under ~60s live (models are small; LLM call dominates).
- Deterministic seeds for train/test split so runs are comparable.
- No silent failures: every sponsor-API error surfaces in the trace and the lab report.

## 8. Risks
| Risk | Mitigation |
|---|---|
| ARIA integration shape unknown | Talk to ARIA table at 9 AM; semi-automated handoff is acceptable and still meaningful |
| TypeSafe API details thin | Their form tonight + onsite engineer; W&B Inference fallback already wired |
| LLM diagnosis is vague | Force evidence tags + require it to cite trace IDs it used |
| Loop improves by luck | Log effect size per action; diagnosis sees what did not help |

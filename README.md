# AgentForge — a self-diagnosing, self-improving ML loop

**CoreWeave Hacks (Agent Loops), Sep 12–13 2026**

An autonomous agent that trains an ML model, evaluates itself, diagnoses *why* it is
underperforming using real statistical signal from its own trace history, and then
acts to fix itself — swapping models, transforming features, acquiring more data, or
delegating code changes — looping until it beats a target benchmark. No human in the loop.

> Built entirely at the hackathon. Rename this project to whatever your team picks —
> search-and-replace `agentforge`.

## The loop

```
TRAIN → EVALUATE → DIAGNOSE → ACT → (repeat)
            │           ▲
            ▼           │
        Weave traces ───┘   (the agent's memory: diagnosis READS past traces via W&B MCP)
```

## Sponsor integrations (all load-bearing)

| Sponsor | Role in the loop | Where |
|---|---|---|
| **Weave** | Every op traced; diagnosis step queries its own trace/eval history via the hosted W&B MCP server — Weave IS the agent's memory | `src/agentforge/tracing.py`, `diagnose.py` |
| **W&B Inference** | LLM backend for the diagnosis reasoning step | `src/agentforge/llm.py` |
| **ARIA** | When diagnosis says the *code* is wrong, the loop emits a structured fix request handed to ARIA in a Models Workspace | `src/agentforge/aria_handoff.py` |
| **marimo / molab** | The agent appends every iteration (diagnosis, action, before/after metrics) to a marimo notebook = live self-writing lab report; molab GPUs for training | `notebooks/lab_report.py`, `src/agentforge/notebook_writer.py` |
| **TypeSafe AI** | Structured action head: System One (Jev) answers typed `Choice` questions over the diagnosis → one validated action with calibrated probabilities; compared against W&B Inference in a Weave eval (`make compare`) | `src/agentforge/typesafe_client.py`, `scripts/compare_action_heads.py` |

**Track selection:** Best Use of Weave (deepest integration). Also eligible: Best Loop Design.

## Quickstart

```bash
make setup                 # uv venv (Python 3.12) + deps
make data                  # fetch UCI heart disease, 4 sites -> data/
make dry-run               # full loop OFFLINE: heuristic brain, no upload — must always pass
cp .env.example .env       # WANDB_API_KEY (+ WANDB_ENTITY if your account has no default entity)
make run                   # the real loop: W&B Inference diagnoses, Weave traces, notebook writes itself
make report                # marimo edit notebooks/lab_report.py  (chart + one section per iteration)
make compare               # P4: TypeSafe vs Inference action-head eval in Weave
make test                  # pytest + dry-run, required before every commit
```

## How the evaluation stays honest
- A stratified 20% test split is held out from **every** site up front (seed in `config.yaml`).
  Acquiring a site only grows the training pool; the test set never changes, so the
  before/after delta recorded for each action is a real effect size.
- The agent starts on Switzerland (123 rows, 93% positive): accuracy ≈ 0.55 on the mixed
  test set. Target 0.85 is reachable only by combining data acquisition, feature ops,
  and model choice — the dry run reaches ~0.87 in 5 iterations.
- Every provider label (`wandb_inference`, `typesafe`, `heuristic_fallback (...)`) and
  history source (`weave_api` vs `local`) is recorded in the trace and the lab report.

## ARIA handoff protocol (file-based)
`request_code_fix` → `aria_requests/NNN.md`. When ARIA (or a human pasting ARIA's diff)
has applied it, write `aria_requests/NNN.applied` with one line (`aria ...` or
`manual-assisted ...`). The next iteration detects it, re-runs `pytest`, and records who
applied it and whether acceptance passed — in Weave and in the lab report.

## Prize-eligibility checklist
- [x] Public GitHub repo, all commits made during the hackathon
- [x] Uses W&B (Weave tracing + Inference + eval logging)
- [ ] All team members registered on AGI House platform + participant survey done (by 10 AM Sun)
- [ ] Submission on AGI House platform by 1:00 PM Sunday
- [ ] < 2 min screen-recording demo

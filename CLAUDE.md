# CLAUDE.md — instructions for Claude Code on this repo

## What this project is
A self-improving ML agent loop for the CoreWeave Hacks hackathon. The loop:
train → evaluate → diagnose → act, repeating until `target_accuracy` in `config.yaml`
is beaten or `max_iterations` is hit. Diagnosis is an LLM call (W&B Inference) that
receives current metrics PLUS the agent's own Weave trace history (via W&B MCP), and
returns a structured action.

## Hard rules
1. **Everything observable.** Every function that trains, evaluates, diagnoses, or acts
   MUST be decorated with `@weave.op`. Never remove tracing to "clean up".
2. **Honest logs only.** Never log that an integration ran unless it actually ran.
   If a sponsor API call fails, log the failure and fall back explicitly.
3. **Actions are typed.** All agent actions are `Action` pydantic models (src/agentforge/actions.py).
   Never act on free text.
4. **Hackathon-built.** Do not import or vendor code from prior projects.
5. Keep the loop runnable end-to-end at ALL times. Prefer a working stub over a broken
   full implementation. `python -m agentforge.loop --dry-run` must always pass.

## Build order (see docs/PLAN.md for detail)
P0: harness + Weave tracing → P1: LLM diagnosis closes the loop → P2: marimo notebook
writer → P3: ARIA handoff → P4: TypeSafe action head. Cut from the bottom if time runs out.

## Environment
- Python 3.11+, deps in requirements.txt. Secrets in `.env` (never commit).
- W&B project name: from `config.yaml: wandb_project`. Don't create new projects ad hoc.
- Dataset: UCI Heart Disease (Cleveland + 3 more sites), fetched by `scripts/fetch_data.py`
  into `data/` (gitignored). Sites are revealed to the agent one at a time — data
  acquisition is one of its actions.

## Testing
`pytest tests/ -x` and `python -m agentforge.loop --dry-run --iterations 2` before every commit.
Commit early and often — judges use git history to verify weekend-built work.

## Demo constraints
3 minutes, strictly enforced. Everything must be visible live: the Weave trace UI,
the marimo lab report updating, and the metric climbing past the benchmark line.

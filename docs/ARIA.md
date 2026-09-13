# ARIA integration notes
ARIA is W&B's in-app coding agent (docs.wandb.ai/aria). Exact programmatic surface TBD —
confirm with the ARIA sponsor table (Julia Rose is PM + judge).

Pattern (works even if only semi-automated):
1. request_code_fix action -> aria_handoff.py writes aria_requests/NNN.md:
   diagnosis text, failing weave trace IDs/links, the module to change, acceptance test.
2. ARIA (in a Models Workspace) implements against the repo.
3. Loop detects the patched module (file watch or a manual "apply" keypress — label
   which one in the trace), re-runs the acceptance test, records before/after.
Honesty rule: if the handoff was manual-assisted, the trace says so.

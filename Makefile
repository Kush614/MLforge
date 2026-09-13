PY ?= .venv/bin/python

.PHONY: setup data dry-run run test report reset compare

setup:            ## create venv + install deps
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt -e .

data:             ## fetch UCI heart disease (4 sites) into data/
	$(PY) scripts/fetch_data.py

dry-run:          ## offline smoke of the full loop (no LLM, no upload)
	$(PY) -m agentforge.loop --dry-run --iterations 2

run:              ## the real loop (needs WANDB_API_KEY in .env)
	$(PY) -m agentforge.loop --reset-notebook

run-poison:       ## adversarial reveal: 40% label noise on VA's train rows (documented)
	$(PY) -m agentforge.loop --reset-notebook --poison va --run-name poison_va

test:             ## what CLAUDE.md requires before every commit
	$(PY) -m pytest tests/ -x -q && $(PY) -m agentforge.loop --dry-run --iterations 2

serve:            ## backend + live frontend at http://127.0.0.1:8008
	$(PY) -m agentforge.server

report:           ## open the self-writing lab report
	.venv/bin/marimo edit notebooks/lab_report.py

reset:            ## wipe agent-written notebook cells + run metrics for a fresh demo
	$(PY) -c "from agentforge.notebook_writer import reset_notebook; reset_notebook()" && rm -f runs/*.jsonl

compare:          ## three-way action-head leaderboard in Weave (TypeSafe / Inference / rules)
	$(PY) scripts/compare_action_heads.py

demo3d:           ## embed the latest run into the 3D loop theatre and open it
	$(PY) scripts/embed_run.py runs/latest.jsonl && open demo/loop3d.html

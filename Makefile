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

test:             ## what CLAUDE.md requires before every commit
	$(PY) -m pytest tests/ -x -q && $(PY) -m agentforge.loop --dry-run --iterations 2

report:           ## open the self-writing lab report
	.venv/bin/marimo edit notebooks/lab_report.py

reset:            ## wipe agent-written notebook cells + run metrics for a fresh demo
	$(PY) -c "from agentforge.notebook_writer import reset_notebook; reset_notebook()" && rm -f runs/*.jsonl

compare:          ## P4: TypeSafe vs Inference action-head eval in Weave
	$(PY) scripts/compare_action_heads.py

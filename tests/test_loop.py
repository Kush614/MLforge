"""End-to-end: the dry-run loop, the notebook writer, and the ARIA handoff protocol."""
import json
import re
from pathlib import Path

from agentforge.tracing import ROOT


def test_dry_run_loop_end_to_end(tmp_path):
    from agentforge import loop
    nb = tmp_path / "report.py"
    nb.write_text((ROOT / "notebooks" / "lab_report.py").read_text())
    acc = loop.main(["--dry-run", "--iterations", "2", "--notebook", str(nb), "--run-name", "pytest"])
    assert 0.0 <= acc <= 1.0
    text = nb.read_text()
    assert "def _iter_0(mo)" in text and "## Iteration 0" in text
    compile(text, str(nb), "exec")  # the agent-written notebook is valid Python
    rows = [json.loads(l) for l in (ROOT / "runs" / "pytest.jsonl").read_text().splitlines()]
    assert rows and rows[0]["iteration"] == 0 and "action" in rows[0]


def test_notebook_writer_is_idempotent_per_iteration(tmp_path):
    from agentforge.notebook_writer import append_iteration, reset_notebook
    nb = tmp_path / "r.py"
    nb.write_text((ROOT / "notebooks" / "lab_report.py").read_text())
    ev = {"accuracy": .6, "balanced_accuracy": .5, "train_accuracy": .7, "train_test_gap": .1, "n_train": 10,
          "n_test": 5, "per_class": {"precision": [0, 1], "recall": [0, 1]}, "learning_curve": [[5, .5], [10, None]]}
    d = {"reasoning": 'uses """ quotes and {braces}', "evidence_tags": ["plateau"], "history_source": "local"}
    for _ in range(2):
        append_iteration(3, ev, d, "switch_model({'family': 'svm'})", "heuristic", str(nb))
    text = nb.read_text()
    assert text.count("def _iter_3(mo)") == 1
    assert "{braces}" in text and "'''" in text
    compile(text, str(nb), "exec")
    reset_notebook(str(nb))
    assert "_iter_3" not in nb.read_text()


def test_aria_handoff_protocol(tmp_path, monkeypatch):
    from agentforge import aria_handoff as ah
    monkeypatch.setattr(ah, "REQUESTS_DIR", tmp_path)
    monkeypatch.setattr(ah, "run_acceptance_test", lambda: {"passed": True, "output": "stub"})
    p = ah.emit_fix_request("train.py drops the 'thal' column", ["trace-1"])
    md = Path(p) if Path(p).is_absolute() else tmp_path / Path(p).name
    assert md.exists() and "trace-1" in md.read_text()
    assert ah.check_for_patches() == []                      # nothing applied yet
    (tmp_path / "001.applied").write_text("manual-assisted: pasted ARIA diff")
    recs = ah.check_for_patches()
    assert len(recs) == 1 and recs[0]["applied_by"] == "manual-assisted" and recs[0]["acceptance_passed"]
    assert ah.check_for_patches() == []                      # recorded once, never double counted

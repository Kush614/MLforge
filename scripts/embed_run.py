"""Embed a run's metrics into the 3D demo page.

    python scripts/embed_run.py [runs/latest.jsonl] [--out demo/loop3d.html]

Reads runs/<run>.jsonl (+ notebooks/lab_report.py for Jev probabilities when a row
lacks them) and rewrites the JSON block between the RUN_DATA markers in demo/loop3d.html.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["iteration", "accuracy", "balanced_accuracy", "train_accuracy", "gap", "n_train", "model", "sites",
        "feature_ops", "target", "action", "provider", "evidence_tags", "history_source", "stop",
        "accounting", "jev_confidence", "jev_probabilities", "escalation", "poison", "controls"]


def probs_from_report(nb: Path) -> dict:
    """iteration -> {kind: p} parsed from the lab report's 'Action-head probabilities' lines."""
    out = {}
    if not nb.exists():
        return out
    text = nb.read_text()
    for m in re.finditer(r"## Iteration (\d+)(.*?)(?=## Iteration|\Z)", text, re.S):
        it, body = int(m.group(1)), m.group(2)
        pm = re.search(r"Action-head probabilities\*\* \(TypeSafe Jev, confidence ([\d.]+)\): ([^\n]+)", body)
        if pm:
            out[it] = {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", pm.group(2))}
    return out


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    report_probs = probs_from_report(ROOT / "notebooks" / "lab_report.py")
    out = []
    for r in rows:
        slim = {k: r.get(k) for k in KEEP}
        slim["diagnosis"] = (r.get("diagnosis") or {}).get("reasoning", "")
        slim["cited"] = (r.get("diagnosis") or {}).get("cited_trace_ids", [])
        slim["diagnosis_provider"] = (r.get("diagnosis") or {}).get("provider", "")
        if not slim.get("jev_probabilities") and r["iteration"] in report_probs:
            slim["jev_probabilities"] = report_probs[r["iteration"]]
        out.append(slim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=str(ROOT / "runs" / "latest.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "demo" / "loop3d.html"))
    ap.add_argument("--name", default=None, help="label shown for this run")
    args = ap.parse_args()
    rows = load_rows(Path(args.run))
    if not rows:
        sys.exit(f"no rows in {args.run}")
    payload = {"name": args.name or Path(args.run).stem, "rows": rows}
    html_path = Path(args.out)
    html = html_path.read_text()
    new = re.sub(r"(/\*RUN_DATA_START\*/)(.*?)(/\*RUN_DATA_END\*/)",
                 lambda m: m.group(1) + json.dumps(payload) + m.group(3), html, flags=re.S)
    if new == html:
        sys.exit("RUN_DATA markers not found")
    html_path.write_text(new)
    print(f"embedded {len(rows)} iterations from {args.run} into {html_path}")


if __name__ == "__main__":
    main()

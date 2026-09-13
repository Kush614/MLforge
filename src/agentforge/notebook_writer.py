"""Self-writing lab report: appends one marimo cell per loop iteration.
marimo notebooks are plain Python, so the agent can extend its own report.

Cells are inserted before the `if __name__ == "__main__":` footer. The notebook's
fixed header cells (chart of accuracy vs. target) read runs/latest.jsonl."""
import re
import weave
from pathlib import Path
from .tracing import config, ROOT

_CELL_NAME_RE = re.compile(r"def _iter_(\d+)\(")


def notebook_path(override: str | None = None) -> Path:
    p = Path(override or config()["notebook_path"])
    return p if p.is_absolute() else ROOT / p


def _md_escape(s: str) -> str:
    # cell body is a raw triple-quoted string: only the delimiter needs escaping
    return str(s).replace('"""', "'''")


@weave.op
def append_iteration(iteration: int, eval_result: dict, diagnosis: dict, action_desc: str,
                     provider: str, notebook: str | None = None, extra: dict | None = None) -> str:
    nb = notebook_path(notebook)
    tags = ", ".join(diagnosis.get("evidence_tags", [])) or "none"
    cited = diagnosis.get("cited_trace_ids") or []
    cited_md = ("  \n**Traces it cited:** " + ", ".join(f"`{t}`" for t in cited)) if cited else ""
    extra = extra or {}
    effect_md = ""
    if extra.get("effect"):
        e = extra["effect"]
        effect_md = (f"\n**Effect of previous action** `{_md_escape(e['action_desc'])}`: "
                     f"{e['accuracy_before']:.3f} → {e['accuracy_after']:.3f} "
                     f"(Δ {e['accuracy_after'] - e['accuracy_before']:+.3f})\n")
    aria_md = ""
    if extra.get("aria"):
        aria_md = "\n**ARIA fixes detected this iteration:** " + "; ".join(
            f"{r['request']} applied by *{r['applied_by']}* — acceptance "
            f"{'passed' if r['acceptance_passed'] else 'FAILED'}" for r in extra["aria"]) + "\n"
    per = eval_result["per_class"]
    lc = eval_result.get("learning_curve") or []
    lc_md = " → ".join(f"{n}:{a:.3f}" if a is not None else f"{n}:n/a" for n, a in lc)
    body = f"""## Iteration {iteration}
**Accuracy:** {eval_result['accuracy']:.3f} (balanced {eval_result.get('balanced_accuracy', float('nan')):.3f}, train {eval_result['train_accuracy']:.3f}, gap {eval_result['train_test_gap']:+.3f}) — n_train={eval_result['n_train']}, n_test={eval_result['n_test']}, sites={extra.get('sites', '?')}, model=`{extra.get('model_family', '?')}`, feature_ops={extra.get('feature_ops', [])}

**Per-class recall:** {per['recall']} · **precision:** {per['precision']} · **learning curve:** {lc_md}
{effect_md}
**Diagnosis** (history: *{diagnosis.get('history_source', 'local')}*, by *{diagnosis.get('provider', '?')}*): {_md_escape(diagnosis.get('reasoning', ''))}

**Evidence tags:** {tags}{cited_md}

**Action taken** (decided by *{_md_escape(provider)}*): `{_md_escape(action_desc)}`
{aria_md}
---"""
    cell = f'''

@app.cell
def _iter_{iteration}(mo):
    mo.md(
        r"""
{body}
        """
    )
    return
'''
    text = nb.read_text() if nb.exists() else _fresh_notebook()
    # idempotent per iteration: replace an existing cell for this iteration if re-run
    pat = re.compile(rf"\n\n@app\.cell\ndef _iter_{iteration}\(mo\):.*?\n    return\n", re.S)
    if pat.search(text):
        text = pat.sub(cell, text, count=1)
    else:
        marker = 'if __name__ == "__main__":'
        idx = text.rfind(marker)
        text = text[:idx].rstrip("\n") + cell + "\n\n" + text[idx:] if idx != -1 else text + cell
    nb.parent.mkdir(parents=True, exist_ok=True)
    nb.write_text(text)
    return str(nb)


def _fresh_notebook() -> str:
    return (ROOT / "notebooks" / "lab_report.py").read_text()


def reset_notebook(notebook: str | None = None) -> Path:
    """Strip all agent-written iteration cells (start of a fresh demo run)."""
    nb = notebook_path(notebook)
    text = nb.read_text()
    text = re.sub(r"\n\n@app\.cell\ndef _iter_\d+\(mo\):.*?\n    return\n", "", text, flags=re.S)
    nb.write_text(text)
    return nb

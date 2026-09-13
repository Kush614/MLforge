"""Self-writing lab report: appends one marimo cell per loop iteration.
marimo notebooks are plain Python, so the agent can extend its own report.

Cells are inserted before the `if __name__ == "__main__":` footer. The notebook's
fixed header cells (chart of accuracy vs. target) read runs/latest.jsonl."""
import re
import weave
from pathlib import Path
from .tracing import config, ROOT

def _cell_pattern(iteration: str) -> re.Pattern:
    """Matches one agent-written cell WITHOUT consuming the newline after its `return`,
    so adjacent cells still match after a neighbour is removed/replaced."""
    return re.compile(rf"\n+@app\.cell\ndef _iter_{iteration}\(mo\):.*?\n    return(?=\n)", re.S)


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
    ts_md = ""
    if extra.get("typesafe"):
        t = extra["typesafe"]
        probs = ", ".join(f"{k} {v:.2f}" for k, v in sorted(t["probabilities"].items(), key=lambda kv: -kv[1]))
        ts_md = f"\n**Action-head probabilities** (TypeSafe Jev, confidence {t['confidence']:.2f}): {probs}\n"
    esc_md = ""
    if extra.get("escalation"):
        e = extra["escalation"]
        esc_md = (f"\n**Confidence gate:** Jev confidence {e['jev_confidence']:.2f} < floor {e['floor']} → second opinion "
                  f"(W&B Inference) said **{e['verdict']}**: {_md_escape(e.get('critique', ''))}\n")
    acct_md = ""
    if extra.get("accounting"):
        a = extra["accounting"]
        usd = f", ${a['usd']:.5f}" if a.get("usd") is not None else ""
        acct_md = f"\n**Decision cost:** {a['latency_s']:.2f} s, tokens in/out {a.get('input_tokens')}/{a.get('output_tokens')}{usd} (`{a['provider']}`)\n"
    pz_md = ""
    if extra.get("poison"):
        pz = extra["poison"]
        pz_md = (f"\n> ⚠️ **Adversarial reveal active** (documented): {pz['label_noise']:.0%} of `{pz['site']}` train labels "
                 f"flipped; test set clean; the agent was not told.\n")
    ctl_md = ""
    if extra.get("frozen_now") or (extra.get("controls") or {}).get("target_accuracy"):
        c = extra.get("controls") or {}
        ctl_md = (f"\n**Human steering via this report:** target={c.get('target_accuracy', 'default')}, "
                  f"frozen sites={c.get('frozen_sites', [])}\n")
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
{ts_md}{esc_md}{acct_md}{aria_md}{pz_md}{ctl_md}
---"""
    cell = f'''

@app.cell
def _iter_{iteration}(mo):
    mo.md(
        r"""
{body}
        """
    )
    return'''
    text = nb.read_text() if nb.exists() else _fresh_notebook()
    # idempotent per iteration: replace an existing cell for this iteration if re-run
    pat = _cell_pattern(str(iteration))
    if pat.search(text):
        text = pat.sub(lambda _m: cell, text, count=1)
    else:
        marker = 'if __name__ == "__main__":'
        idx = text.rfind(marker)
        text = text[:idx].rstrip("\n") + cell + "\n\n\n" + text[idx:] if idx != -1 else text + cell + "\n"
    nb.parent.mkdir(parents=True, exist_ok=True)
    nb.write_text(text)
    return str(nb)


def _fresh_notebook() -> str:
    return (ROOT / "notebooks" / "lab_report.py").read_text()


def reset_notebook(notebook: str | None = None) -> Path:
    """Strip all agent-written iteration cells (start of a fresh demo run)."""
    nb = notebook_path(notebook)
    text = nb.read_text()
    text = _cell_pattern(r"\d+").sub("", text)
    nb.write_text(text)
    return nb

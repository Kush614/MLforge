import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # AgentForge — self-written lab report

    This notebook is written **by the agent**, one section per loop iteration:
    what it measured, what it diagnosed, and what it decided to do about it.
    The chart and the cockpit are the only human-authored cells; everything below the
    rule was written by the agent.
    """)
    return


@app.cell
def _(mo):
    # Auto-refresh so the chart follows a live run without touching the keyboard.
    refresh = mo.ui.refresh(default_interval="5s", options=["2s", "5s", "15s"])
    refresh
    return (refresh,)


@app.cell
def _(mo, refresh):
    # Live chart: accuracy per iteration vs. the target benchmark line.
    # Reads runs/latest.jsonl, which the loop rewrites after every iteration.
    import json
    from pathlib import Path

    refresh  # dependency: re-run on every tick
    _rows = []
    _p = Path(__file__).resolve().parents[1] / "runs" / "latest.jsonl"
    if _p.exists():
        _rows = [json.loads(l) for l in _p.read_text().splitlines() if l.strip()]
    if _rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, _ax = plt.subplots(figsize=(7, 3.2))
        _its = [r["iteration"] for r in _rows]
        _ax.plot(_its, [r["accuracy"] for r in _rows], marker="o", label="held-out accuracy")
        _ax.plot(_its, [r["train_accuracy"] for r in _rows], marker=".", alpha=0.5, label="train accuracy")
        _ax.axhline(_rows[-1]["target"], ls="--", color="tab:red", label=f"target {_rows[-1]['target']}")
        _ax.axhline(0.787, ls=":", color="gray", label="published baseline 0.787")
        for r in _rows:
            if r.get("action"):
                _lbl = r["action"].split("(")[0]
                if r.get("jev_confidence") is not None:
                    _lbl += f" p={r['jev_confidence']:.2f}"
                _ax.annotate(_lbl, (r["iteration"], r["accuracy"]),
                             textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        _ax.set_xlabel("iteration"); _ax.set_ylabel("accuracy"); _ax.set_ylim(0.4, 1.0)
        _ax.legend(fontsize=7, loc="lower right"); _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _last = _rows[-1]
        _acct = [r["accounting"] for r in _rows if r.get("accounting")]
        _cost = ""
        if _acct:
            _cost = (f" · decisions: {len(_acct)}, mean {sum(a['latency_s'] for a in _acct)/len(_acct):.2f} s, "
                     f"{sum((a.get('input_tokens') or 0) + (a.get('output_tokens') or 0) for a in _acct)} tokens")
        _pz = _last.get("poison")
        _chart = mo.vstack([
            mo.md(f"**Run:** {len(_rows)} iteration(s) · latest accuracy **{_last['accuracy']:.3f}** "
                  f"· model `{_last['model']}` · sites {_last['sites']}{_cost}"
                  + (" · *synthetic data (dry run)*" if _last.get("synthetic") else "")
                  + (f" · ⚠️ *adversarial reveal: {_pz['label_noise']:.0%} of {_pz['site']} train labels flipped (documented)*" if _pz else "")),
            _fig,
        ])
    else:
        _chart = mo.md("_No run metrics yet — start `python -m agentforge.loop`._")
    _chart
    return


@app.cell
def _(mo):
    # Cockpit: the human steers the running loop from the lab report. Values are written to
    # runs/controls.json, which the loop reads at the top of every iteration and records
    # honestly ("target overridden by human via lab report").
    target_slider = mo.ui.slider(0.60, 0.95, step=0.01, value=0.85, label="target accuracy")
    freeze_sites = mo.ui.multiselect(["hungary", "va", "cleveland"], value=[], label="freeze sites (agent may not acquire)")
    apply_btn = mo.ui.run_button(label="Apply to running loop")
    clear_btn = mo.ui.run_button(label="Clear overrides")
    mo.vstack([mo.md("### Cockpit — human steering"), mo.hstack([target_slider, freeze_sites]), mo.hstack([apply_btn, clear_btn])])
    return apply_btn, clear_btn, freeze_sites, target_slider


@app.cell
def _(apply_btn, clear_btn, freeze_sites, mo, target_slider):
    import json as _json
    from pathlib import Path as _Path

    _ctl = _Path(__file__).resolve().parents[1] / "runs" / "controls.json"
    _ctl.parent.mkdir(exist_ok=True)
    if clear_btn.value:
        _ctl.unlink(missing_ok=True)
        _status = mo.md("_Overrides cleared; the loop uses config.yaml again._")
    elif apply_btn.value:
        _ctl.write_text(_json.dumps({"target_accuracy": target_slider.value,
                                     "frozen_sites": list(freeze_sites.value)}, indent=2))
        _status = mo.md(f"**Applied:** target={target_slider.value}, frozen={list(freeze_sites.value)} → `runs/controls.json`")
    else:
        _status = mo.md(f"_Current file:_ `{_ctl.read_text().strip() if _ctl.exists() else 'none'}`")
    _status
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return

@app.cell
def _iter_0(mo):
    mo.md(
        r"""
## Iteration 0
**Accuracy:** 0.551 (balanced 0.500, train 0.939, gap +0.387) — n_train=98, n_test=185, sites=['switzerland'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.0, 1.0] · **precision:** [0.0, 0.551] · **learning curve:** 49:0.551 → 73:0.551 → 98:0.551

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is dramatically overfitting to the overwhelmingly positive training set (train_accuracy 0.939 vs test_accuracy 0.551, train‑test gap 0.387). The training data is extremely imbalanced (train_pos_frac 0.939) and contains very few negative examples, so the classifier learns to predict only the positive class (confusion matrix shows 0 true negatives, recall for class 0 = 0). The learning curve is flat (accuracy ≈ 0.551 for n_train = 49, 73, 98), indicating a performance plateau caused by lack of informative minority samples rather than model capacity. Past attempts to mitigate this—switching to random_forest and gradient_boosting, tuning depth, and applying class_weight='balanced' (traces 01a09bd2-1c13, 01a09bd2-1c1a, 01a09bd2-1c1b, 01a09bd2-1c15)—did not improve the situation, confirming that the core issue is data‑starvation of the minority class and severe class imbalance, leading to overfitting and a plateau.

**Evidence tags:** overfitting, class_imbalance, data_starved, plateau  
**Traces it cited:** `01a09bd2-1c13-7d8d-98b4-1ca7874b3113`, `01a09bd2-1c1a-75c9-9820-30c483636d54`, `01a09bd2-1c1b-72cc-bb82-dbec8a7d4727`, `01a09bd2-1c15-7d8c-bbca-a3b40737bbb5`, `01a09bd2-1cdd-7ed5-b1fd-ebed62b866b3`, `01a09c11-2b84-7289-96db-88fd46e3dc5d`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.89): acquire_data 0.91, switch_model 0.04, tune_hyperparams 0.02, transform_features 0.02, request_code_fix 0.01

**Decision cost:** 0.31 s, tokens in/out 2316/222 (`typesafe`)

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

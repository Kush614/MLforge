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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model now predicts only the positive class (confusion [[0,83],[0,102]], recall 0 for class 0, 1.0 for class 1). This stems from extreme class imbalance in the tiny training set (train_pos_frac 0.939, only ~6 negatives out of 98) combined with over‑fitting: train accuracy is 0.939 while test accuracy is 0.551, giving a huge train‑test gap of 0.387. Earlier runs with far larger, more balanced training data (e.g., trace 01a09bd2-0c4b-7a5f-87fe-a3d9d9f305e4: acc 0.843, n_train 500; trace 01a09bd2-0d7b-7c40-901a-d9d1824f07b9: acc 0.870, n_train 500; trace 01a09bd2-1dcb-7d1e-9fbe-65fca8754d2b: acc 0.859, n_train 735) achieved high accuracy and much smaller gaps, showing that the current failure is not due to model choice or hyper‑parameters (class_weight balanced was already tried) but to insufficient minority‑class examples and resulting over‑fit to the majority. The learning curve is flat (0.551 across 49‑98 samples), confirming a plateau caused by data starvation rather than model capacity.

**Evidence tags:** class_imbalance, data_starved, overfitting, plateau  
**Traces it cited:** `01a09bd2-0c4b-7a5f-87fe-a3d9d9f305e4`, `01a09bd2-0d7b-7c40-901a-d9d1824f07b9`, `01a09bd2-1dcb-7d1e-9fbe-65fca8754d2b`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.98): acquire_data 0.99, switch_model 0.01, transform_features 0.00, tune_hyperparams 0.00, request_code_fix 0.00

**Decision cost:** 0.30 s, tokens in/out 2665/221 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_1(mo):
    mo.md(
        r"""
## Iteration 1
**Accuracy:** 0.724 (balanced 0.717, train 0.809, gap +0.084) — n_train=340, n_test=185, sites=['switzerland', 'cleveland'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.651, 0.784] · **precision:** [0.711, 0.734] · **learning curve:** 170:0.714 → 255:0.697 → 340:0.724

**Effect of previous action** `acquire_data({'site': 'cleveland'})`: 0.551 → 0.724 (Δ +0.173)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is limited mainly by the small amount of training data (n_train=340). The learning curve shows accuracy rising from 0.7135 at 170 samples to 0.7243 at 340, and historical runs with larger datasets (e.g., 575 samples → acc 0.697, 735 samples → acc 0.859) demonstrate that more data improves performance. This data starvation also hampers the model's ability to learn the minority (negative) class, reflected in the lower recall (0.651) and the confusion matrix (29 false positives, 22 false negatives). The train‑test gap of 0.0845 indicates mild overfitting, but the relatively low train accuracy (0.809) compared to earlier runs (≈0.87) suggests the model is also underfitting given the limited data. Together, insufficient data and class imbalance are the primary drivers of the current underperformance.

**Evidence tags:** data_starved, class_imbalance, underfitting, overfitting  
**Traces it cited:** `01a09bd2-1c26-7c43-b2d8-d47d352493dd`, `01a09bd2-1c45-700c-b8fd-8660e3282c17`, `01a09bd2-1dcb-7d1e-9fbe-65fca8754d2b`, `01a09c11-2b84-7289-96db-88fd46e3dc5d`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'hungary'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.91): acquire_data 0.92, switch_model 0.05, tune_hyperparams 0.02, transform_features 0.01, request_code_fix 0.00

**Decision cost:** 0.17 s, tokens in/out 2555/212 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_2(mo):
    mo.md(
        r"""
## Iteration 2
**Accuracy:** 0.697 (balanced 0.702, train 0.788, gap +0.091) — n_train=575, n_test=185, sites=['switzerland', 'cleveland', 'hungary'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.747, 0.657] · **precision:** [0.639, 0.761] · **learning curve:** 287:0.714 → 431:0.724 → 575:0.697

**Effect of previous action** `acquire_data({'site': 'hungary'})`: 0.724 → 0.697 (Δ -0.027)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.697) is notably lower than its training accuracy (0.788), yielding a train‑test gap of 0.09, which signals overfitting to the training distribution. More tellingly, the learning curve drops as more samples are added (0.7135 → 0.7243 → 0.6973), and the most recent data acquisition from the Hungarian site caused a performance dip (0.724 → 0.697, Δ‑0.027). This suggests a domain shift or quality issue in the newly added data that harms generalisation. Additionally, the positive class recall falls to 0.657 while the test set has a higher positive fraction (0.551 vs 0.501 in training), indicating a mild class‑imbalance mismatch that further hurts the minority class. Together, these point to overfitting compounded by distributional drift from the added Hungarian data and a slight class‑imbalance mismatch.

**Evidence tags:** overfitting, class_imbalance  
**Traces it cited:** `01a09c11-f2c1-734b-93e5-ee5224440e69`, `01a09bd2-1c10-74d0-974b-62b508fb07ab`, `01a09bd2-1c13-7d8d-98b4-1ca7874b3113`

**Action taken** (decided by *wandb_inference override (jev confidence 0.13 < 0.4)*): `acquire_data({'site': 'va'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.13): switch_model 0.30, tune_hyperparams 0.27, transform_features 0.23, acquire_data 0.19, request_code_fix 0.01

**Confidence gate:** Jev confidence 0.13 < floor 0.4 → second opinion (W&B Inference) said **override**: Switching to random_forest has already been attempted without yielding a performance gain. The diagnosis points to domain shift from the Hungarian data and a modest class‑imbalance mismatch. The most promising remedy is to enrich the training set with data from a different source to counteract the drift, rather than further model changes that have already been explored.

**Decision cost:** 0.25 s, tokens in/out 2514/203 (`typesafe`)

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

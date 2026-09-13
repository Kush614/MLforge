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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model predicts only the positive class (confusion matrix shows 0 true negatives, recall for class 0 is 0.0) despite a high training accuracy (0.939). This is driven by extreme class imbalance in the training set (train_pos_frac 0.939) combined with a very small training sample size (n_train 98). The large train‑test gap (0.387) and flat learning curve (accuracy stays at 0.551 regardless of increasing from 49 to 98 samples) indicate severe overfitting to the majority class and data starvation. Past attempts that added more data (n_train up to 735) and used class_weight balanced yielded much higher accuracy (~0.86) and smaller gaps, showing that the current failure is not due to model choice or hyperparameters but to insufficient, imbalanced data.

**Evidence tags:** overfitting, data_starved, class_imbalance  
**Traces it cited:** `01a09c11-2b84-7289-96db-88fd46e3dc5d`, `01a09c3c-adbf-7a3d-8d3d-535cb7c0515c`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.99): acquire_data 1.00, transform_features 0.00, switch_model 0.00, request_code_fix 0.00, tune_hyperparams 0.00

**Decision cost:** 0.30 s, tokens in/out 2335/221 (`typesafe`)

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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.724) is noticeably lower than its training accuracy (0.809), giving a train‑test gap of 0.0845. This modest but consistent gap, together with the learning curve that still rises when the training set grows from 255 (0.697) to 340 samples (0.724), indicates the model has not yet saturated its performance – it is data‑starved and under‑fitting the underlying patterns. Historical runs with larger training sets (e.g., trace 01a09bd2-1dcb-7d1e-9fbe-65fca8754d2b, n_train=735, acc=0.859, gap=0.141) show even larger gaps, suggesting that increasing capacity without more diverse data leads to over‑fitting. The class balance is only mildly skewed (train_pos_frac ≈ 0.60), and tree‑based models used (random forest, gradient boosting) do not require feature scaling, so feature‑scale and class‑imbalance are not primary causes. The evidence points to a combination of data scarcity (under‑fitting) and residual over‑fitting manifested by the train‑test gap, especially evident in past evaluations with larger datasets.

**Evidence tags:** underfitting, overfitting, data_starved  
**Traces it cited:** `01a09bd2-1cdd-7ed5-b1fd-ebed62b866b3`, `01a09bd2-1cf5-740a-b3e3-6fc6f17ba64d`, `01a09bd2-1dcb-7d1e-9fbe-65fca8754d2b`, `01a09c41-9d46-7f6c-90fb-ccbdc3698bb9`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'va'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.66): acquire_data 0.74, tune_hyperparams 0.15, switch_model 0.09, transform_features 0.02, request_code_fix 0.00

**Decision cost:** 0.28 s, tokens in/out 2506/211 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_2(mo):
    mo.md(
        r"""
## Iteration 2
**Accuracy:** 0.697 (balanced 0.686, train 0.810, gap +0.113) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.578, 0.794] · **precision:** [0.696, 0.698] · **learning curve:** 250:0.697 → 375:0.730 → 500:0.697

**Effect of previous action** `acquire_data({'site': 'va'})`: 0.724 → 0.697 (Δ -0.027)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.697) lags behind its training accuracy (0.81) giving a train‑test gap of 0.113, indicating moderate over‑fitting. The per‑class recall is highly asymmetric (0.578 for the negative class vs 0.794 for the positive class), and the class‑positive fraction differs between train (0.644) and test (0.551), showing a class‑imbalance and distribution shift problem. The learning curve plateaus and even dips when the training set grows from 375 to 500 samples (0.7297 → 0.697), suggesting the model is data‑starved and cannot benefit from more examples, likely because the added data come from heterogeneous sites (Hungary, Cleveland, VA) that introduce covariate shift. Past actions such as applying class_weight='balanced', limiting tree depth, and switching between RandomForest and GradientBoosting have not closed the gap, so the core issue is the combination of modest data size, site heterogeneity, and residual class imbalance rather than model choice or scaling.

**Evidence tags:** overfitting, data_starved, class_imbalance, plateau  
**Traces it cited:** `01a09bd2-1cd0-780a-96be-f1bb991a640c`, `01a09bd2-1cd6-7bf4-892a-b09decdedcd3`, `01a09bd2-1cdd-7ed5-b1fd-ebed62b866b3`, `01a09bd2-1cf5-740a-b3e3-6fc6f17ba64d`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `transform_features({'op': 'onehot'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.46): transform_features 0.57, switch_model 0.20, tune_hyperparams 0.12, acquire_data 0.09, request_code_fix 0.02

**Decision cost:** 0.21 s, tokens in/out 2593/205 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_3(mo):
    mo.md(
        r"""
## Iteration 3
**Accuracy:** 0.692 (balanced 0.681, train 0.808, gap +0.116) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`knn`, feature_ops=['onehot']

**Per-class recall:** [0.578, 0.784] · **precision:** [0.686, 0.696] · **learning curve:** 250:0.686 → 375:0.730 → 500:0.692

**Effect of previous action** `transform_features({'op': 'onehot'})`: 0.697 → 0.692 (Δ -0.005)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is struggling mainly due to overfitting on a very small, imbalanced dataset. Train accuracy (0.808) is noticeably higher than test accuracy (0.692) giving a train‑test gap of 0.116, a classic sign of overfitting. The learning curve confirms this: performance rises to 0.7297 at 375 samples then drops to 0.6919 at 500 samples, indicating the model cannot generalize as data grows (plateau then decline). With only 500 training rows for 13 features, the data is starved, limiting the model’s ability to learn robust patterns. Class imbalance (train positive fraction 0.644 vs test 0.551) further hurts the minority class, whose recall is only 0.578 compared to 0.784 for the majority class. Past actions that tried to fix the issue—adding more Hungarian data, switching between RandomForest and GradientBoosting, tuning class_weight, and one‑hot encoding—did not improve performance (e.g., accuracy fell from 0.724 to 0.697 after adding VA data and to 0.692 after one‑hot). The best historic runs (acc ≈ 0.86‑0.87 with gap ≈ 0.06‑0.14) still show a gap and were achieved with the same limited data size, confirming that data scarcity and imbalance are the root causes rather than model choice or hyper‑parameters.

**Evidence tags:** overfitting, data_starved, class_imbalance, plateau  
**Traces it cited:** `01a09bd2-1cd0-780a-96be-f1bb991a640c`, `01a09bd2-1cdd-7ed5-b1fd-ebed62b866b3`, `01a09c41-9d46-7f6c-90fb-ccbdc3698bb9`, `01a09c42-5a7f-7f90-99dd-269ae89dab8a`, `01a09c43-2a7d-75bc-8c70-9fe3823fa0ad`

**Action taken** (decided by *wandb_inference override (jev confidence 0.21 < 0.4)*): `switch_model({'family': 'logistic_regression'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.21): acquire_data 0.38, tune_hyperparams 0.31, switch_model 0.29, transform_features 0.01, request_code_fix 0.01

**Confidence gate:** Jev confidence 0.21 < floor 0.4 → second opinion (W&B Inference) said **override**: The diagnosis identifies data scarcity and class imbalance as the primary issues. Acquiring additional Hungarian data has already been attempted repeatedly with minimal or negative impact, indicating diminishing returns from that source. Switching to a model that is less prone to overfitting on small, imbalanced datasets—such as logistic regression with regularization—offers a more promising path to improve generalization.

**Decision cost:** 0.25 s, tokens in/out 2717/200 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_4(mo):
    mo.md(
        r"""
## Iteration 4
**Accuracy:** 0.849 (balanced 0.837, train 0.816, gap -0.033) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`logistic_regression`, feature_ops=['onehot']

**Per-class recall:** [0.723, 0.951] · **precision:** [0.923, 0.808] · **learning curve:** 250:0.854 → 375:0.784 → 500:0.849

**Effect of previous action** `switch_model({'family': 'logistic_regression'})`: 0.692 → 0.849 (Δ +0.157)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.849) exceeds its training accuracy (0.816), giving a negative train‑test gap of –0.033, which signals underfitting rather than overfitting. The learning curve is unstable (0.854 at 250 samples, drops to 0.784 at 375, then rises to 0.849 at 500), indicating the model cannot consistently capture patterns with the current data size. Per‑class recall is imbalanced (0.723 vs 0.951), and the minority class suffers the most, a symptom of both underfitting and modest class imbalance (train_pos_frac 0.644, test_pos_frac 0.551). Past actions that increased model capacity (gradient boosting) produced large positive gaps (+0.141) – clear overfitting – while the recent switch to logistic regression (trace 01a09c43-8e08-7db5-8adf-705f303cc397) reduced the gap but left the model under‑powered for the heterogeneous multi‑site data. Thus the primary cause is underfitting due to limited model capacity on a relatively small, heterogeneous dataset, compounded by mild class imbalance.

**Evidence tags:** underfitting, data_starved, class_imbalance  
**Traces it cited:** `01a09c43-8e08-7db5-8adf-705f303cc397`, `01a09bd2-1cdd-7ed5-b1fd-ebed62b866b3`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'hungary'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.40): acquire_data 0.52, tune_hyperparams 0.43, switch_model 0.05, request_code_fix 0.00, transform_features 0.00

**Decision cost:** 0.25 s, tokens in/out 2689/209 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_5(mo):
    mo.md(
        r"""
## Iteration 5
**Accuracy:** 0.870 (balanced 0.863, train 0.814, gap -0.057) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'va', 'hungary'], model=`logistic_regression`, feature_ops=['onehot']

**Per-class recall:** [0.795, 0.931] · **precision:** [0.904, 0.848] · **learning curve:** 367:0.784 → 551:0.870 → 735:0.870

**Effect of previous action** `acquire_data({'site': 'hungary'})`: 0.849 → 0.870 (Δ +0.022)

**Diagnosis** (history: *n/a*, by *n/a*): Accuracy 0.870 >= target 0.85. No further action.

**Evidence tags:** none

**Action taken** (decided by *n/a*): `none (target beaten)`

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

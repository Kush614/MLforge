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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model predicts only the positive class (confusion [[0,83],[0,102]]), giving recall 0 for the negative class and precision 0 for it. This stems from the extreme class imbalance in the tiny training set (train_pos_frac = 0.939 with only n_train = 98) while the test set is much more balanced (test_pos_frac = 0.551). The high train accuracy (0.939) versus low test accuracy (0.551) yields a huge train‑test gap (0.387), indicating the model has memorized the majority class but cannot generalize to the minority. The learning curve is flat (accuracy stays at 0.551 regardless of adding up to 98 samples), showing a performance plateau caused by insufficient minority examples. Thus the primary failure modes are severe class imbalance and data starvation for the minority class.

**Evidence tags:** class_imbalance, data_starved, plateau  
**Traces it cited:** `01a09c11-2b84-7289-96db-88fd46e3dc5d`, `01a09c3c-adbf-7a3d-8d3d-535cb7c0515c`, `01a09c41-9d73-7bc0-b8b5-ce611a60424b`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.48): acquire_data 0.58, switch_model 0.29, tune_hyperparams 0.09, request_code_fix 0.03, transform_features 0.01

**Decision cost:** 0.19 s, tokens in/out 2489/222 (`typesafe`)

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

**Diagnosis** (history: *local*, by *heuristic_fallback (wandb_inference error: APITimeoutError)*): heuristic: raw features have very different scales

**Evidence tags:** feature_scale

**Action taken** (decided by *typesafe (jev-1.13.0)*): `transform_features({'op': 'standardize'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.95): transform_features 0.96, switch_model 0.03, acquire_data 0.01, request_code_fix 0.00, tune_hyperparams 0.00

**Decision cost:** 0.26 s, tokens in/out 2335/211 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_2(mo):
    mo.md(
        r"""
## Iteration 2
**Accuracy:** 0.838 (balanced 0.833, train 0.888, gap +0.050) — n_train=340, n_test=185, sites=['switzerland', 'cleveland'], model=`knn`, feature_ops=['standardize']

**Per-class recall:** [0.783, 0.882] · **precision:** [0.844, 0.833] · **learning curve:** 170:0.849 → 255:0.822 → 340:0.838

**Effect of previous action** `transform_features({'op': 'standardize'})`: 0.724 → 0.838 (Δ +0.114)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.838) is modestly below its train accuracy (0.888) with a 0.05 gap, indicating no severe overfitting. The learning curve flattens: performance drops from 0.8486 at 170 samples to 0.8216 at 255, then recovers only to 0.8378 at 340, showing a plateau despite added data. This suggests the model is underfitting given the limited capacity relative to the problem (underfitting, plateau). The dataset is small (n_train=340) and the learning curve does not improve with more examples, confirming a data‑starved regime. Additionally, per‑class recall is imbalanced (0.783 for the negative class vs 0.882 for the positive), and the class fractions (0.597 vs 0.551) are skewed, indicating class imbalance is hurting sensitivity on the minority class. Past actions that tried to fix these issues—acquiring more sites, one‑hot encoding, and switching to logistic regression—either gave modest gains or introduced negative gaps, so repeating those has limited value. The current bottleneck is likely a combination of underfitting due to model capacity, insufficient data, and residual class imbalance.

**Evidence tags:** underfitting, data_starved, class_imbalance, plateau  
**Traces it cited:** `01a09c49-d73f-70c9-8cf1-f4ae43409b97`, `01a09c44-3c6a-7af8-a590-08ed49027f74`, `01a09c11-f2b6-79f2-bd33-17b23fc1a70b`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.78): tune_hyperparams 0.83, switch_model 0.14, acquire_data 0.02, transform_features 0.01, request_code_fix 0.00

**Decision cost:** 0.25 s, tokens in/out 2403/206 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_3(mo):
    mo.md(
        r"""
## Iteration 3
**Accuracy:** 0.843 (balanced 0.839, train 1.000, gap +0.157) — n_train=340, n_test=185, sites=['switzerland', 'cleveland'], model=`knn`, feature_ops=['standardize']

**Per-class recall:** [0.795, 0.882] · **precision:** [0.846, 0.841] · **learning curve:** 170:0.822 → 255:0.843 → 340:0.843

**Effect of previous action** `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`: 0.838 → 0.843 (Δ +0.005)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.843, yielding a large train‑test gap of 0.157. The learning curve plateaus early (0.8432 at 255 samples, no gain at 340), indicating that adding more of the same data does not improve generalisation. Past actions that tried to address data scarcity (acquiring additional sites) either did not reduce the gap or even increased it, and feature scaling already helped but did not close the gap. The remaining imbalance in recall (0.795 vs 0.882) also points to the model favouring the majority class, a typical symptom of over‑fitting on limited data. Hence the primary issue is overfitting combined with a plateau in learning, not feature scaling or class imbalance.

**Evidence tags:** overfitting, plateau, data_starved  
**Traces it cited:** `01a09c3c-adbf-7a3d-8d3d-535cb7c0515c`, `01a09c42-5a8a-7787-a7bf-dc3daa1cddca`, `01a09c4c-a2d1-792c-ae33-8398d250aed4`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_neighbors': 3}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.46): tune_hyperparams 0.56, switch_model 0.38, acquire_data 0.04, request_code_fix 0.01, transform_features 0.01

**Decision cost:** 0.21 s, tokens in/out 2325/202 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_4(mo):
    mo.md(
        r"""
## Iteration 4
**Accuracy:** 0.811 (balanced 0.806, train 1.000, gap +0.189) — n_train=340, n_test=185, sites=['switzerland', 'cleveland'], model=`knn`, feature_ops=['standardize']

**Per-class recall:** [0.759, 0.853] · **precision:** [0.808, 0.813] · **learning curve:** 170:0.827 → 255:0.795 → 340:0.811

**Effect of previous action** `tune_hyperparams({'params': {'n_neighbors': 3}})`: 0.843 → 0.811 (Δ -0.032)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.811, giving a train‑test gap of 0.189. The learning curve shows no consistent improvement with more data (0.827 → 0.795 → 0.811) and even a dip at 255 samples, indicating a plateau rather than a benefit from additional data. Per‑class recall is imbalanced (0.759 for the negative class vs 0.854 for the positive class), reflecting the modest class imbalance (train_pos_frac ≈ 0.60) and the model’s bias toward the majority class. Recent actions that did help—standardizing features (trace 01a09c4c-a2c0-7a51-8761-19738efcfe95) and modest K‑NN hyper‑parameter tuning (trace 01a09c4d-3caf-794f-9999-b4836ea8bd4a)—only yielded tiny gains (+0.114 then +0.005) and could not overcome the overfit. The earlier switch to logistic regression (trace 01a09c44-3cb0-739f-97c9-52534ffc9a39) achieved higher test accuracy (0.849) with a negative gap, suggesting a less complex model fits the data better. Therefore the primary failure mode is overfitting due to a high‑capacity K‑NN on a small, moderately imbalanced dataset, compounded by a plateau in learning from more data.

**Evidence tags:** overfitting, class_imbalance, plateau  
**Traces it cited:** `01a09c4c-a2c0-7a51-8761-19738efcfe95`, `01a09c4d-3caf-794f-9999-b4836ea8bd4a`, `01a09c44-3cb0-739f-97c9-52534ffc9a39`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `switch_model({'family': 'logistic_regression'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.91): switch_model 0.94, acquire_data 0.06, request_code_fix 0.00, transform_features 0.00

**Decision cost:** 0.23 s, tokens in/out 2492/163 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_5(mo):
    mo.md(
        r"""
## Iteration 5
**Accuracy:** 0.843 (balanced 0.842, train 0.868, gap +0.024) — n_train=340, n_test=185, sites=['switzerland', 'cleveland'], model=`logistic_regression`, feature_ops=['standardize']

**Per-class recall:** [0.831, 0.853] · **precision:** [0.821, 0.861] · **learning curve:** 170:0.832 → 255:0.854 → 340:0.843

**Effect of previous action** `switch_model({'family': 'logistic_regression'})`: 0.811 → 0.843 (Δ +0.032)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s performance has plateaued around 0.84 accuracy despite modest training‑test gap (0.024) and recent hyperparameter tuning. The learning curve shows a peak at 255 samples (0.854) then a dip at 340 samples (0.843), indicating that adding more of the current data does not yield gains and the model may be under‑fitting the underlying pattern. Historical evidence shows that increasing the training set size dramatically improves accuracy (e.g., from 0.724 at n_train=340 to 0.870 at n_train=735, trace 01a09c44-b0c1-7d32-8d55-b9ea73cb77b9). Conversely, model‑centric changes (standardization, k‑NN tuning) only yielded marginal improvements (+0.005 max). Thus the primary bottleneck is insufficient, diverse training data (data_starved) causing a performance plateau (plateau) and limited model capacity for the available data (underfitting).

**Evidence tags:** data_starved, plateau, underfitting  
**Traces it cited:** `01a09c42-5a8a-7787-a7bf-dc3daa1cddca`, `01a09c44-b0c1-7d32-8d55-b9ea73cb77b9`, `01a09c4c-a2c0-7a51-8761-19738efcfe95`, `01a09c4d-3caf-794f-9999-b4836ea8bd4a`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'hungary'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.98): acquire_data 0.99, tune_hyperparams 0.01, switch_model 0.00, transform_features 0.00, request_code_fix 0.00

**Decision cost:** 0.28 s, tokens in/out 2538/215 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_6(mo):
    mo.md(
        r"""
## Iteration 6
**Accuracy:** 0.849 (balanced 0.844, train 0.843, gap -0.005) — n_train=575, n_test=185, sites=['switzerland', 'cleveland', 'hungary'], model=`logistic_regression`, feature_ops=['standardize']

**Per-class recall:** [0.795, 0.892] · **precision:** [0.857, 0.843] · **learning curve:** 287:0.843 → 431:0.860 → 575:0.849

**Effect of previous action** `acquire_data({'site': 'hungary'})`: 0.843 → 0.849 (Δ +0.005)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s train and test accuracies are almost identical (train_accuracy = 0.8435, test_accuracy = 0.8486, train_test_gap ≈ ‑0.005), indicating no over‑fitting. The learning curve rises to 0.8595 at 431 samples then drops to 0.8486 at 575 samples, showing a plateau rather than continued improvement with more data. Per‑class recall is lower for the negative class (0.795 vs 0.892), suggesting the model is not capturing all patterns. Past actions that helped most were feature standardization (accuracy jumped from 0.724 to 0.838) and adding the Hungary site data (tiny lift to 0.849), but further hyper‑parameter tweaks or model switches gave only marginal or negative changes. This pattern points to underfitting caused by limited informative data and model capacity – the model has hit a performance ceiling with the current 575 training points and 13 features. Hence the primary issues are underfitting, data starvation, and a performance plateau.

**Evidence tags:** underfitting, data_starved, plateau  
**Traces it cited:** `01a09c4c-a2c0-7a51-8761-19738efcfe95`, `01a09c44-b066-75b4-9b02-43538caa75b3`, `01a09c44-3cb0-739f-97c9-52534ffc9a39`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'va'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.59): acquire_data 0.66, tune_hyperparams 0.25, switch_model 0.05, transform_features 0.03, request_code_fix 0.01

**Decision cost:** 0.27 s, tokens in/out 2591/206 (`typesafe`)

---
        """
    )
    return

@app.cell
def _iter_7(mo):
    mo.md(
        r"""
## Iteration 7
**Accuracy:** 0.854 (balanced 0.845, train 0.816, gap -0.038) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'hungary', 'va'], model=`logistic_regression`, feature_ops=['standardize']

**Per-class recall:** [0.759, 0.931] · **precision:** [0.9, 0.826] · **learning curve:** 367:0.854 → 551:0.854 → 735:0.854

**Effect of previous action** `acquire_data({'site': 'va'})`: 0.849 → 0.854 (Δ +0.005)

**Diagnosis** (history: *n/a*, by *n/a*): Accuracy 0.854 >= target 0.85. No further action.

**Evidence tags:** none

**Action taken** (decided by *n/a*): `none (target beaten)`

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

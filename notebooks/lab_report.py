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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model predicts only the positive class (confusion [[0,83],[0,102]]), giving recall 0 for the negative class and perfect recall for the positive class. This stems from extreme class imbalance in the training set (train_pos_frac = 0.939) combined with a very small training set (n_train = 98). The huge train‑test accuracy gap (0.9388 – 0.5514 = 0.387) shows the model overfits to the majority class and cannot generalize to the minority. Earlier runs with more data (n_train = 575) and a balanced class_weight (trace 01a09bc5-b948) achieved high accuracy (0.849) and a near‑zero gap, indicating that the imbalance handling was effective when enough data were present. The current run lacks that mitigation and is also data‑starved, so the model collapses to a trivial majority‑class predictor.

**Evidence tags:** class_imbalance, overfitting, data_starved  
**Traces it cited:** `01a09bc5-b948`, `01a09bc4-0227`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.82): acquire_data 0.86, switch_model 0.13, tune_hyperparams 0.01, request_code_fix 0.00, transform_features 0.00

**Decision cost:** 0.36 s, tokens in/out 2150/222 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s accuracy (0.724) is well below the peak 0.849 achieved earlier with logistic regression on a larger training set (n_train=575, trace 01a09bc5-647c). The learning curve shows performance rising as data increase (0.7135→0.7243 from 170→340 samples) but still far from the plateau seen at 575–735 samples (0.843–0.849, trace 01a09bc6-7573). The current train‑test gap (0.084) is modest, indicating the issue is not severe overfitting but rather underfitting caused by insufficient data (data_starved) and a modest class imbalance (train_pos_frac = 0.597, test_pos_frac = 0.551) that depresses recall for the negative class (0.651). Earlier attempts to tune depth, learning rate, or class_weight did not help when data were scarce, confirming that the bottleneck is the limited sample size rather than hyper‑parameters or feature scaling. Therefore the primary reason for underperformance is underfitting due to data starvation, compounded by residual class imbalance.

**Evidence tags:** underfitting, data_starved, class_imbalance  
**Traces it cited:** `01a09bc5-647c`, `01a09bc5-0ed8`, `01a09bc6-7573`, `01a09bc4-0227`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'va'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.85): acquire_data 0.88, switch_model 0.12, tune_hyperparams 0.00, request_code_fix 0.00, transform_features 0.00

**Decision cost:** 0.25 s, tokens in/out 2220/212 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_2(mo):
    mo.md(
        r"""
## Iteration 2
**Accuracy:** 0.686 (balanced 0.685, train 0.762, gap +0.076) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.675, 0.696] · **precision:** [0.644, 0.724] · **learning curve:** 250:0.697 → 375:0.719 → 500:0.686

**Effect of previous action** `acquire_data({'site': 'va'})`: 0.724 → 0.686 (Δ -0.038)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.686) is well below the earlier peak (0.849) achieved after logistic‑regression tuning on 575 samples. Adding the VA site data caused a performance drop (0.724 → 0.686, see trace 01a09bc9-28d7-7e6a-8803-a3c72f88010f), indicating a distribution shift that the current model cannot handle. The learning curve peaks at 375 samples (0.7189) then falls at 500 samples (0.6865), showing a plateau and even degradation as more heterogeneous data are added. The train‑test gap (0.075) and higher train accuracy (0.762) suggest mild over‑fitting to the original sites, while the added site introduces heterogeneity that the model’s capacity or regularisation does not accommodate. Thus the primary issue is a plateau caused by multi‑site data heterogeneity and slight over‑fitting to the original distribution.

**Evidence tags:** plateau, overfitting  
**Traces it cited:** `01a09bc9-28d7-7e6a-8803-a3c72f88010f`, `01a09bc6-7573-788b-adc4-0ce5d79dec64`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `switch_model({'family': 'random_forest'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.53): switch_model 0.62, transform_features 0.20, tune_hyperparams 0.17, acquire_data 0.01, request_code_fix 0.00

**Decision cost:** 0.28 s, tokens in/out 2202/205 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_3(mo):
    mo.md(
        r"""
## Iteration 3
**Accuracy:** 0.805 (balanced 0.797, train 1.000, gap +0.195) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`random_forest`, feature_ops=[]

**Per-class recall:** [0.711, 0.882] · **precision:** [0.831, 0.789] · **learning curve:** 250:0.849 → 375:0.822 → 500:0.805

**Effect of previous action** `switch_model({'family': 'random_forest'})`: 0.686 → 0.805 (Δ +0.119)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.805, yielding a large train‑test gap of 0.195. The learning curve shows performance degrading as more data is added (0.8486 → 0.8216 → 0.8054), indicating the model’s capacity exceeds what the 500‑sample training set can support. Attempts to mitigate this—standardizing features, balancing class weights, and switching to simpler linear models—did not close the gap (e.g., logistic regression still had a gap of ~0.005, class_weight balanced gave the same gap). The random forest that now gives the best test score still overfits (train 1.0). Additionally, recall for the negative class is low (0.711 vs 0.882 for the positive class), reflecting residual class‑imbalance effects that the earlier class_weight tweak failed to resolve. Thus the primary issues are overfitting due to a model that is too complex for the limited data (data‑starved) and lingering class‑imbalance impact on the minority class.

**Evidence tags:** overfitting, data_starved, class_imbalance  
**Traces it cited:** `01a09bc5-6469-7c29-8565-eadac26f0bec`, `01a09bc5-b948-72fc-82c7-b5636131bcf2`, `01a09bc6-2872-7898-ab32-c332386c3509`, `01a09bc9-5123-7ff2-8f74-6c899d8a3fd3`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'max_depth': 4, 'min_samples_leaf': 5}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.46): tune_hyperparams 0.57, acquire_data 0.40, switch_model 0.03, request_code_fix 0.00, transform_features 0.00

**Decision cost:** 0.34 s, tokens in/out 2369/214 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_4(mo):
    mo.md(
        r"""
## Iteration 4
**Accuracy:** 0.816 (balanced 0.797, train 0.796, gap -0.020) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`random_forest`, feature_ops=[]

**Per-class recall:** [0.614, 0.98] · **precision:** [0.962, 0.758] · **learning curve:** 250:0.854 → 375:0.838 → 500:0.816

**Effect of previous action** `tune_hyperparams({'params': {'max_depth': 4, 'min_samples_leaf': 5}})`: 0.805 → 0.816 (Δ +0.011)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s performance is plateauing around 0.81 despite adding more data (learning curve drops from 0.854 at 250 samples to 0.816 at 500). Train accuracy (0.796) is slightly lower than test accuracy (0.816), giving a negative train‑test gap (‑0.020), which signals underfitting rather than overfitting. The random‑forest was constrained to max_depth=4 and min_samples_leaf=5 (trace 01a09bc9-851c-799c-a588-f11d3da5f768), limiting its capacity; this explains why accuracy degrades as the dataset grows. Additionally, per‑class recall is highly imbalanced (0.614 for the negative class vs 0.98 for the positive class), indicating the model is biased toward the majority/positive class despite only modest class imbalance (train_pos_frac≈0.59). Earlier attempts with logistic regression (trace 01a09bc5-647c-7d1f-8e48-53c9db097024) achieved higher accuracy (0.849) with a near‑zero gap, suggesting the current tree depth is too restrictive. Therefore the primary issues are underfitting due to overly limited model capacity and residual class‑imbalance bias.

**Evidence tags:** underfitting, class_imbalance, plateau  
**Traces it cited:** `01a09bc9-851c-799c-a588-f11d3da5f768`, `01a09bc5-647c-7d1f-8e48-53c9db097024`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_estimators': 500, 'max_depth': 12}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.81): tune_hyperparams 0.86, switch_model 0.14, request_code_fix 0.00, transform_features 0.00, acquire_data 0.00

**Decision cost:** 0.28 s, tokens in/out 2455/207 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_5(mo):
    mo.md(
        r"""
## Iteration 5
**Accuracy:** 0.827 (balanced 0.813, train 0.868, gap +0.041) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`random_forest`, feature_ops=[]

**Per-class recall:** [0.675, 0.951] · **precision:** [0.918, 0.782] · **learning curve:** 250:0.843 → 375:0.843 → 500:0.827

**Effect of previous action** `tune_hyperparams({'params': {'n_estimators': 500, 'max_depth': 12}})`: 0.816 → 0.827 (Δ +0.011)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s test accuracy (0.827) is only 0.041 lower than train accuracy (0.868), indicating a modest over‑fit. The learning curve shows a plateau at 250‑375 samples (0.843) then a drop at 500 samples (0.827), a classic sign that adding more data does not improve generalisation and the model is beginning to over‑fit to the limited 500‑sample training set (data_starved). Moreover, per‑class recall is highly asymmetric: recall for the negative class is 0.675 while the positive class is 0.951, and balanced accuracy (0.813) is noticeably lower than raw accuracy (0.827). This points to residual class‑imbalance effects despite the modest imbalance (train_pos_frac ≈ 0.59). Past attempts to fix it – applying class_weight='balanced' (trace 01a09bc5-b95d-7e73-a7d0-b835ef008e16) and adding more sites (VA data, trace 01a09bc6-7573-788b-adc4-0ce5d79dec64) – did not raise performance, confirming that the issue is not simply feature scaling or data quantity. The current random‑forest (max_depth 12) is relatively complex for only 500 examples, which together with the class‑imbalance explains the lingering gap and the low recall for the minority class. The diagnosis therefore combines overfitting, data starvation, and class‑imbalance as the primary reasons for the under‑performance.

**Evidence tags:** overfitting, data_starved, class_imbalance, plateau  
**Traces it cited:** `01a09bc5-b95d-7e73-a7d0-b835ef008e16`, `01a09bc9-8557-7787-9be8-a9c1ba43ec29`, `01a09bc9-dd2c-76a7-add3-2430b95e6c73`, `01a09bc9-dcfe-7a25-94b6-b7730f629ae0`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'class_weight': 'balanced'}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.73): tune_hyperparams 0.79, switch_model 0.11, acquire_data 0.08, transform_features 0.01, request_code_fix 0.01

**Decision cost:** 2.30 s, tokens in/out 2590/200 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_6(mo):
    mo.md(
        r"""
## Iteration 6
**Accuracy:** 0.811 (balanced 0.806, train 0.886, gap +0.075) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`random_forest`, feature_ops=[]

**Per-class recall:** [0.759, 0.853] · **precision:** [0.808, 0.813] · **learning curve:** 250:0.843 → 375:0.854 → 500:0.811

**Effect of previous action** `tune_hyperparams({'params': {'class_weight': 'balanced'}})`: 0.827 → 0.811 (Δ -0.016)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model shows a classic over‑fitting pattern on a very small, heterogeneous dataset. Test accuracy (0.811) is 0.075 lower than train accuracy (0.886), and the learning curve peaks at 375 samples (0.854) then drops to 0.811 at 500 samples, indicating that adding more (noisy) data hurts performance. The per‑class recall is imbalanced (0.759 for the negative class vs 0.853 for the positive), reflecting the modest class imbalance (train pos 0.592, test pos 0.551) but not enough to explain the gap. Past actions that increased model capacity (switch to random_forest, depth 12, 500 trees) gave the best accuracy (0.827) but still left a sizable gap, and further regularisation attempts (class_weight balanced) actually reduced performance. Earlier attempts to acquire more data from other sites did not improve accuracy, suggesting the new data may be of lower quality or distribution‑shifted, exacerbating over‑fitting. Thus the primary issue is overfitting due to a high‑capacity model on a data‑starved, multi‑site dataset, with secondary contribution from modest class imbalance.

**Evidence tags:** overfitting, data_starved, class_imbalance  
**Traces it cited:** `01a09bc9-8557-7787-9be8-a9c1ba43ec29`, `01a09bca-243b-7cf8-a60f-41d867a5daaf`, `01a09bca-2376-7065-bf47-8f8e4691313f`, `01a09bc8-fd2f-7169-9150-cab1f89fb8d1`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `switch_model({'family': 'logistic_regression'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.74): switch_model 0.80, transform_features 0.11, acquire_data 0.07, request_code_fix 0.01

**Decision cost:** 0.25 s, tokens in/out 2533/163 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_7(mo):
    mo.md(
        r"""
## Iteration 7
**Accuracy:** 0.811 (balanced 0.802, train 0.770, gap -0.041) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`logistic_regression`, feature_ops=[]

**Per-class recall:** [0.711, 0.892] · **precision:** [0.843, 0.791] · **learning curve:** 250:0.854 → 375:0.843 → 500:0.811

**Effect of previous action** `switch_model({'family': 'logistic_regression'})`: 0.811 → 0.811 (Δ +0.000)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is underfitting. Train accuracy (0.77) is lower than test accuracy (0.811), giving a negative train‑test gap of -0.0408, which signals insufficient capacity rather than over‑fitting. The learning curve confirms this: accuracy drops from 0.8541 at 250 samples to 0.8108 at 500 samples, showing that adding more data hurts performance because the model cannot capture the underlying patterns. The best historical result (0.827) was achieved with a higher‑capacity RandomForest (n_estimators=500, max_depth=12) – a modest gain over the current LogisticRegression (0.811). Subsequent attempts to balance class weights reduced performance, indicating that class imbalance is not the primary driver. Therefore, the dominant issue is underfitting due to a model that is too simple for the available features and data distribution.

**Evidence tags:** underfitting  
**Traces it cited:** `01a09bc5-6469-7c29-8565-eadac26f0bec`, `01a09bc5-647c-7d1f-8e48-53c9db097024`, `01a09bc9-8557-7787-9be8-a9c1ba43ec29`, `01a09bca-243b-7cf8-a60f-41d867a5daaf`

**Action taken** (decided by *typesafe (jev-1.13.0) — low confidence 0.36 < 0.4, second opinion: accept*): `switch_model({'family': 'random_forest'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.36): switch_model 0.50, tune_hyperparams 0.41, transform_features 0.08, acquire_data 0.01, request_code_fix 0.00

**Confidence gate:** Jev confidence 0.36 < floor 0.4 → second opinion (W&B Inference) said **accept**: The diagnosis correctly identifies underfitting and points to the higher‑capacity RandomForest (n_estimators=500, max_depth=12) that previously achieved 0.827 accuracy. Switching from the under‑performing LogisticRegression back to a RandomForest aligns with the evidence and is the most supported remedy, despite the low confidence score.

**Decision cost:** 0.36 s, tokens in/out 2566/215 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return

@app.cell
def _iter_8(mo):
    mo.md(
        r"""
## Iteration 8
**Accuracy:** 0.805 (balanced 0.797, train 1.000, gap +0.195) — n_train=500, n_test=185, sites=['switzerland', 'cleveland', 'va'], model=`random_forest`, feature_ops=[]

**Per-class recall:** [0.711, 0.882] · **precision:** [0.831, 0.789] · **learning curve:** 250:0.849 → 375:0.822 → 500:0.805

**Effect of previous action** `switch_model({'family': 'random_forest'})`: 0.811 → 0.805 (Δ -0.005)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.805, giving a large train‑test gap of 0.1946. The learning curve drops as more data is added (0.8486 → 0.8216 → 0.8054), confirming that the high‑capacity random‑forest continues to memorize the training set rather than generalize. Per‑class recall is imbalanced (0.711 for the negative class vs 0.882 for the positive), and the confusion matrix shows many false positives (24) and false negatives (12), indicating the model is biased toward the majority class (train_pos_frac = 0.592). Past interventions that increased model capacity (more trees, deeper depth) gave only marginal gains, and attempts to address class imbalance with class_weight='balanced' actually hurt performance, showing that the core issue is over‑parameterization relative to the modest dataset (n_train = 500, n_features = 13). Further data acquisition from additional sites has not improved performance, suggesting the new data may be of lower quality or distribution‑shifted, reinforcing the overfitting plateau. The diagnosis therefore points to overfitting as the primary cause, compounded by modest class imbalance.

**Evidence tags:** overfitting, class_imbalance  
**Traces it cited:** `01a09bc9-8557-7787-9be8-a9c1ba43ec29`, `01a09bc9-dd2c-76a7-add3-2430b95e6c73`, `01a09bca-8b88-735b-8b90-e01827f74b57`, `01a09bca-e56c-7023-913d-00273a93dead`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `switch_model({'family': 'svm'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.58): switch_model 0.68, transform_features 0.27, acquire_data 0.03, request_code_fix 0.01

**Decision cost:** 0.34 s, tokens in/out 2604/161 (`typesafe`)

> ⚠️ **Adversarial reveal active** (documented): 40% of `va` train labels flipped; test set clean; the agent was not told.

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

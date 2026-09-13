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
    Nothing below the chart was written by a human.
    """)
    return


@app.cell
def _(mo):
    # Live chart: accuracy per iteration vs. the target benchmark line.
    # Reads runs/latest.jsonl, which the loop rewrites after every iteration.
    import json
    from pathlib import Path

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
                _ax.annotate(r["action"].split("(")[0], (r["iteration"], r["accuracy"]),
                             textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        _ax.set_xlabel("iteration"); _ax.set_ylabel("accuracy"); _ax.set_ylim(0.4, 1.0)
        _ax.legend(fontsize=7, loc="lower right"); _ax.grid(alpha=0.3)
        _fig.tight_layout()
        _chart = mo.vstack([
            mo.md(f"**Run:** {len(_rows)} iteration(s) · latest accuracy **{_rows[-1]['accuracy']:.3f}** "
                  f"· model `{_rows[-1]['model']}` · sites {_rows[-1]['sites']}"
                  + (" · *synthetic data (dry run)*" if _rows[-1].get("synthetic") else "")),
            _fig,
        ])
    else:
        _chart = mo.md("_No run metrics yet — start `python -m agentforge.loop`._")
    _chart
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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely over‑fitting to a training set that is both tiny (n_train=98) and heavily skewed toward the positive class (train_pos_frac=0.939). This leads to a degenerate predictor that labels everything as positive (recall for the negative class is 0, confusion matrix shows 0 true negatives). The test distribution is much less imbalanced (test_pos_frac=0.551), so the learned bias does not transfer, producing a large train‑test gap of 0.387 and a balanced accuracy of only 0.5. The learning curve is flat (accuracy stays at 0.551 regardless of training size), indicating a plateau caused by insufficient and unrepresentative data rather than model capacity. Past attempts to add more data (e.g., traces 01a0999e-c4f2-7dc8-9244-3342242dd99d, 01a0999f-5e5d-760c-8ec4-7b41d85a7cd9) and to switch models (random_forest, gradient_boosting, logistic_regression) improved overall accuracy only when the training set was larger (735 samples) but the current run reverted to a small, imbalanced subset, undoing those gains. Feature standardization also failed to address the core imbalance. Therefore the primary reasons are overfitting to a small, class‑imbalanced training set and a distribution shift between train and test, causing a performance plateau.

**Evidence tags:** overfitting, data_starved, class_imbalance, plateau  
**Traces it cited:** `01a0999e-7d5c-76ba-95af-3966868436f3`, `01a0999e-c4f2-7dc8-9244-3342242dd99d`, `01a0999f-5e5d-760c-8ec4-7b41d85a7cd9`, `01a099f-b687-79a1-a6ca-e7da11efba2a`, `01a099a3-4f08-78a6-a488-29a75b2f12a2`, `01a09ba1-5f4d-7310-874d-8e09acb08055`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'cleveland'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.99): acquire_data 1.00, request_code_fix 0.00, switch_model 0.00, transform_features 0.00, tune_hyperparams 0.00

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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model shows a modest train‑test gap (0.084) and a train accuracy of only 0.809, indicating it is not fully exploiting the signal in the data. The learning curve rises from 0.7135 at 170 samples to 0.7243 at the full 340 samples, showing that performance still improves with more data and has not plateaued. Per‑class recall is especially low for the negative class (0.651 vs 0.784 for positives), and the confusion matrix reveals many false positives (29). These symptoms point to underfitting caused by insufficient training data (data‑starved) rather than severe overfitting. Historical attempts confirm this: the random forest run (trace 01a099f-b687-79a1-a6ca-e7da11efba2a) achieved high accuracy (0.854) but a large gap (+0.146), evidencing over‑capacity overfit on the same data size. Subsequent experiments with very small subsets (n_train≈98) and repeated standardization (traces 01a099a2‑..., 01a099a3‑..., 01a09ba1‑...) failed to improve performance, reinforcing that the bottleneck is the limited amount of labeled data and model capacity relative to the problem complexity. Therefore the primary reason for the current underperformance is data starvation leading to underfitting.

**Evidence tags:** underfitting, data_starved  
**Traces it cited:** `01a099f-b687-79a1-a6ca-e7da11efba2a`, `01a099a2-a4bd-7553-b3f6-b44d07cbbca7`, `01a099a3-4f08-78a6-a488-29a75b2f12a2`, `01a099a3-d1d7-7aba-a400-582f3d30b6ac`, `01a09ba1-5e9f-75f5-bbc4-930a7b87f550`, `01a09ba1-5f4d-7310-874d-8e09acb08055`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'hungary'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.89): acquire_data 0.91, switch_model 0.05, tune_hyperparams 0.03, transform_features 0.01, request_code_fix 0.00

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

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model’s performance stalls around 70% accuracy despite adding more training samples. The learning curve shows accuracy rising to 0.724 at 431 examples then dropping to 0.697 at 575, indicating a plateau and slight degradation when heterogeneous site data are added. The train‑test gap (~0.09) is modest, but the drop after incorporating the VA site suggests the model is over‑fitting to site‑specific patterns and not generalizing across sites (domain shift). Earlier experiments confirm this: a random‑forest on 735 samples achieved 0.854 accuracy but a large gap (+0.146) (trace 01a099b‑...); adding the Hungary site alone gave 0.854 with almost no gap (trace 01a099a4-18a7-72b3-906f-68725ce11a29), while mixing sites later reduces accuracy (current 0.697). Thus the primary issue is over‑fitting to site‑specific distributions combined with a plateau caused by heterogeneous data, not class imbalance or feature scaling.

**Evidence tags:** overfitting, plateau  
**Traces it cited:** `01a099a4-18a7-72b3-906f-68725ce11a29`, `01a099b-...`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.29): tune_hyperparams 0.43, switch_model 0.35, transform_features 0.14, acquire_data 0.07, request_code_fix 0.01

---
        """
    )
    return

@app.cell
def _iter_3(mo):
    mo.md(
        r"""
## Iteration 3
**Accuracy:** 0.735 (balanced 0.738, train 1.000, gap +0.265) — n_train=575, n_test=185, sites=['switzerland', 'cleveland', 'hungary'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.771, 0.706] · **precision:** [0.681, 0.791] · **learning curve:** 287:0.714 → 431:0.735 → 575:0.735

**Effect of previous action** `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`: 0.697 → 0.735 (Δ +0.038)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is dramatically overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.735, yielding a huge train‑test gap of 0.265. The learning curve flattens after 431 samples (0.735) and does not improve with the full 575 examples, indicating the model’s capacity exceeds what the data can support. Prior attempts to standardize features (traces 01a09ba1-5e9f-75f5-bbc4-930a7b87f550, 01a09ba1-5f4d-7310-874d-8e09acb08055) did not close the gap, and switching to a k‑NN with tuned k=11 only raised test accuracy modestly (0.697→0.735, trace 01a09baf-387c-73dd-bc27-f615a55fc564). The modest class imbalance (train_pos_frac 0.501 vs test_pos_frac 0.551) and balanced per‑class recall (0.771, 0.706) suggest the issue is not class imbalance but model over‑complexity on a limited dataset. Hence the primary failure mode is overfitting, compounded by data scarcity relative to model capacity.

**Evidence tags:** overfitting, data_starved  
**Traces it cited:** `01a099f-5e6c-7737-b642-ef8aaf5e17ff`, `01a099f-b687-79a1-a6ca-e7da11efba2a`, `01a09baf-387c-73dd-bc27-f615a55fc564`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `acquire_data({'site': 'va'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.69): acquire_data 0.76, tune_hyperparams 0.18, switch_model 0.06, request_code_fix 0.00, transform_features 0.00

---
        """
    )
    return

@app.cell
def _iter_4(mo):
    mo.md(
        r"""
## Iteration 4
**Accuracy:** 0.741 (balanced 0.740, train 1.000, gap +0.259) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'hungary', 'va'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.735, 0.745] · **precision:** [0.701, 0.776] · **learning curve:** 367:0.735 → 551:0.724 → 735:0.741

**Effect of previous action** `acquire_data({'site': 'va'})`: 0.735 → 0.741 (Δ +0.005)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model shows a classic overfitting pattern: train accuracy is perfect (1.0) while test accuracy is only 0.74, yielding a large train‑test gap of 0.259. Past experiments that increased model capacity (random_forest, gradient_boosting) achieved higher training scores but still left sizable gaps (e.g., random_forest acc=0.854 gap=+0.146). Simpler models (logistic_regression) performed worse, indicating the issue is not under‑capacity. The learning curve (367→735 samples) still climbs modestly (0.735→0.741), suggesting the model would benefit from more data rather than further complexity. Thus the dominant causes are overfitting due to limited data (data_starved) and insufficient regularization.

**Evidence tags:** overfitting, data_starved  
**Traces it cited:** `01a099f-b687-79a1-a6ca-e7da11efba2a`, `01a099a3-4f08-78a6-a488-29a75b2f12a2`, `01a099a3-d1d7-7aba-a400-582f3d30b6ac`, `01a09ba1-5f4d-7310-874d-8e09acb08055`, `01a09baf-c523-7fec-bc3b-a946fc1c4e24`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.79): tune_hyperparams 0.85, transform_features 0.08, switch_model 0.07, request_code_fix 0.00

---
        """
    )
    return

@app.cell
def _iter_5(mo):
    mo.md(
        r"""
## Iteration 5
**Accuracy:** 0.741 (balanced 0.740, train 1.000, gap +0.259) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'hungary', 'va'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.735, 0.745] · **precision:** [0.701, 0.776] · **learning curve:** 367:0.735 → 551:0.724 → 735:0.741

**Effect of previous action** `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`: 0.741 → 0.741 (Δ +0.000)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely overfitting: train accuracy is perfect (1.0) while test accuracy is only 0.74, yielding a large train‑test gap of 0.259. The learning curve still rises modestly as more samples are added (0.735 → 0.724 → 0.740), indicating the model has not yet reached a performance plateau and could benefit from more data. Prior attempts to fix the issue—standardizing features (traces 01a09ba1-5f4d-7310-874d-8e09acb08055, 01a09ba1-62f3-7d5a-a277-3d206600dcfa) and switching model families (gradient boosting, logistic regression) – did not close the gap. Hyperparameter tuning of k‑NN gave only marginal gains. Thus the dominant problem is overfitting compounded by data starvation, not class imbalance or scaling issues.

**Evidence tags:** overfitting, data_starved  
**Traces it cited:** `01a09ba1-5f4d-7310-874d-8e09acb08055`, `01a09ba1-62f3-7d5a-a277-3d206600dcfa`, `01a09bae-f0c0-7626-b131-c90b27331af4`, `01a09baf-3877-7e76-8cca-82cb47ad2318`, `01a09bb0-02f4-7f95-aae1-ff33f348c920`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.45): tune_hyperparams 0.59, switch_model 0.35, request_code_fix 0.04, transform_features 0.02

---
        """
    )
    return

@app.cell
def _iter_6(mo):
    mo.md(
        r"""
## Iteration 6
**Accuracy:** 0.741 (balanced 0.740, train 1.000, gap +0.259) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'hungary', 'va'], model=`knn`, feature_ops=[]

**Per-class recall:** [0.735, 0.745] · **precision:** [0.701, 0.776] · **learning curve:** 367:0.735 → 551:0.724 → 735:0.741

**Effect of previous action** `tune_hyperparams({'params': {'n_neighbors': 11, 'weights': 'distance'}})`: 0.741 → 0.741 (Δ +0.000)

**Diagnosis** (history: *weave_api*, by *wandb_inference*): The model achieves perfect training accuracy (1.0) but only 0.74 test accuracy, yielding a large train‑test gap of 0.259. This gap signals severe overfitting. The learning curve shows test performance barely improves as the training set grows from 367 to 735 samples (0.7351 → 0.7405) and then plateaus, indicating that simply adding more data yields diminishing returns. Past interventions—switching model families, repeated standardization, and k‑nearest‑neighbors hyperparameter tuning—produced only marginal gains (e.g., +0.005 after adding the VA site, +0.000 after repeated tuning). Acquiring additional sites also gave negligible improvement. Thus the core issue is a high‑variance model that overfits the limited data, and the dataset size, while larger than earlier runs, remains insufficient for the model’s capacity, leading to a performance plateau.

**Evidence tags:** overfitting, plateau, data_starved  
**Traces it cited:** `01a09bb0-02f4-7f95-aae1-ff33f348c920`, `01a09bb0-7b5c-710a-bc6b-e73d99cb94e3`, `01a09ba1-5f4d-7310-874d-8e09acb08055`

**Action taken** (decided by *typesafe (jev-1.13.0)*): `switch_model({'family': 'logistic_regression'})`

**Action-head probabilities** (TypeSafe Jev, confidence 0.36): switch_model 0.52, tune_hyperparams 0.35, transform_features 0.10, request_code_fix 0.03

---
        """
    )
    return

@app.cell
def _iter_7(mo):
    mo.md(
        r"""
## Iteration 7
**Accuracy:** 0.859 (balanced 0.851, train 0.818, gap -0.042) — n_train=735, n_test=185, sites=['switzerland', 'cleveland', 'hungary', 'va'], model=`logistic_regression`, feature_ops=[]

**Per-class recall:** [0.771, 0.931] · **precision:** [0.901, 0.833] · **learning curve:** 367:0.843 → 551:0.854 → 735:0.860

**Effect of previous action** `switch_model({'family': 'logistic_regression'})`: 0.741 → 0.859 (Δ +0.119)

**Diagnosis** (history: *n/a*, by *n/a*): Accuracy 0.859 >= target 0.85. No further action.

**Evidence tags:** none

**Action taken** (decided by *n/a*): `none (target beaten)`

---
        """
    )
    return


if __name__ == "__main__":
    app.run()

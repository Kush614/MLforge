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
    mo.md(r"""
    ## Iteration 0
    **Accuracy:** 0.551 (balanced 0.500, train 0.939, gap +0.387) — n_train=98, n_test=185, sites=['switzerland'], model=`knn`, feature_ops=[]

    **Per-class recall:** [0.0, 1.0] · **precision:** [0.0, 0.551] · **learning curve:** 49:0.551 → 73:0.551 → 98:0.551

    **Diagnosis** (history: *weave_api*, by *wandb_inference*): The model now predicts only the positive class (confusion [[0,83],[0,102]], per‑class recall 0.0 for negatives, precision 0.0), driven by an extreme training‑set imbalance (train_pos_frac = 0.939) and a tiny sample size (n_train = 98). This yields a huge train‑test gap (0.9388 vs 0.5513 → gap = 0.387) indicating severe overfitting. The learning curve is flat (accuracy 0.5514 at 49, 73, 98 samples), showing a performance plateau and data starvation for the minority class. Earlier traces where more data were acquired (n_train = 735) and a random forest was used achieved 0.854 accuracy and a much smaller gap (0.146), confirming that the current regression is caused by the small, imbalanced dataset rather than model choice or scaling.

    **Evidence tags:** overfitting, data_starved, class_imbalance, plateau
    **Traces it cited:** `01a0999e-7d5c-76ba-95af-3966868436f3`, `01a0999f-b687-79a1-a6ca-e7da11efba2a`

    **Action taken** (decided by *wandb_inference*): `switch_model({'family': 'gradient_boosting'})`

    ---
    """)
    return


@app.cell
def _iter_1(mo):
    mo.md(r"""
    ## Iteration 1
    **Accuracy:** 0.708 (balanced 0.678, train 1.000, gap +0.292) — n_train=98, n_test=185, sites=['switzerland'], model=`gradient_boosting`, feature_ops=[]

    **Per-class recall:** [0.386, 0.971] · **precision:** [0.914, 0.66] · **learning curve:** 49:n/a → 73:0.595 → 98:0.708

    **Effect of previous action** `switch_model({'family': 'gradient_boosting'})`: 0.551 → 0.708 (Δ +0.157)

    **Diagnosis** (history: *weave_api*, by *wandb_inference*): The model is severely overfitting: train_accuracy is 1.0 while test accuracy is only 0.708, giving a train‑test gap of 0.292. The minority (negative) class is almost absent in training (train_pos_frac = 0.939 → ≈9 negatives out of 98), leading to a very low recall of 0.386 for that class, while the majority class recall is 0.971. This class imbalance and data starvation for the negative class cause the model to predict positives almost always, inflating overall accuracy but hurting balanced metrics (balanced_accuracy = 0.678). The learning curve shows modest gains with more data (0.5946 at 73 samples, 0.708 at 98) but the gap remains large, indicating that simply adding more data from the same distribution has not solved the imbalance. Past interventions—acquiring additional sites, switching to random_forest (trace 01a0999f-b641-7365-9da1-860570a06e85) and gradient_boosting (trace 01a099a2-a493-7df7-8216-32d31422faa7)—improved overall accuracy but did not reduce the gap or improve minority recall, confirming that model family changes alone are insufficient. The core issue is overfitting driven by extreme class imbalance and insufficient negative examples.

    **Evidence tags:** overfitting, data_starved, class_imbalance
    **Traces it cited:** `01a0999e-7d5c-76ba-95af-3966868436f3`, `01a0999f-b641-7365-9da1-860570a06e85`, `01a099a2-a4bd-7553-b3f6-b44d07cbbca7`, `01a099a2-a493-7df7-8216-32d31422faa7`

    **Action taken** (decided by *wandb_inference*): `switch_model({'family': 'logistic_regression'})`

    ---
    """)
    return


@app.cell
def _iter_2(mo):
    mo.md(r"""
    ## Iteration 2
    **Accuracy:** 0.568 (balanced 0.520, train 0.918, gap +0.351) — n_train=98, n_test=185, sites=['switzerland'], model=`logistic_regression`, feature_ops=[]

    **Per-class recall:** [0.06, 0.98] · **precision:** [0.714, 0.562] · **learning curve:** 49:n/a → 73:0.546 → 98:0.568

    **Effect of previous action** `switch_model({'family': 'logistic_regression'})`: 0.708 → 0.568 (Δ -0.141)

    **Diagnosis** (history: *weave_api*, by *wandb_inference*): The current model suffers from severe overfitting (train_accuracy = 0.918 vs test_accuracy = 0.568, train_test_gap = 0.351) and a mismatch in class distribution (train_pos_frac = 0.939 vs test_pos_frac = 0.551). This is reflected in the per‑class recall: recall for the negative class is only 0.06 while the positive class is 0.98, yielding a low balanced_accuracy of 0.520. The training set is very small (n_train = 98) compared with earlier runs that used 333, 575, and 735 samples and achieved higher accuracy and smaller gaps (e.g., gap = 0.091 at n_train = 575, trace 01a0999f-5e6c-7737-b642-ef8aaf5e17ff). The recent switch back to a 98‑sample dataset (trace 01a099a2-a4bd-7553-b3f6-b44d07cbbca7) re‑introduced data starvation, undoing the gains from earlier data acquisition. Model family changes (random_forest, gradient_boosting, logistic_regression) did not resolve the issue, indicating the problem is not model choice but insufficient, imbalanced data. Therefore the underperformance is driven by overfitting caused by data starvation and class imbalance.

    **Evidence tags:** overfitting, data_starved, class_imbalance
    **Traces it cited:** `01a0999e-7d5c-76ba-95af-3966868436f3`, `01a0999e-c502-74a2-84a9-9127843256c3`, `01a0999f-5e6c-7737-b642-ef8aaf5e17ff`, `01a0999f-b641-7365-9da1-860570a06e85`, `01a099a2-a4bd-7553-b3f6-b44d07cbbca7`, `01a099a3-4f08-78a6-a488-29a75b2f12a2`

    **Action taken** (decided by *wandb_inference*): `acquire_data({'site': 'hungary'})`

    ---
    """)
    return


@app.cell
def _iter_3(mo):
    mo.md(r"""
    ## Iteration 3
    **Accuracy:** 0.854 (balanced 0.850, train 0.847, gap -0.007) — n_train=333, n_test=185, sites=['switzerland', 'hungary'], model=`logistic_regression`, feature_ops=[]

    **Per-class recall:** [0.807, 0.892] · **precision:** [0.859, 0.85] · **learning curve:** 166:0.789 → 249:0.849 → 333:0.854

    **Effect of previous action** `acquire_data({'site': 'hungary'})`: 0.568 → 0.854 (Δ +0.286)

    **Diagnosis** (history: *n/a*, by *n/a*): Accuracy 0.854 >= target 0.85. No further action.

    **Evidence tags:** none

    **Action taken** (decided by *n/a*): `none (target beaten)`

    ---
    """)
    return


if __name__ == "__main__":
    app.run()

"""The orchestrator. Run: python -m agentforge.loop [--dry-run] [--iterations N]

--dry-run : no LLM, no network, no Weave upload. Heuristic diagnosis/decision heads
            exercise the SAME action space + effect bookkeeping. Uses synthetic data
            only if data/ has not been fetched, and says so.
"""
import argparse
import time
import weave
from rich.console import Console
from rich.table import Table

from .tracing import init_tracing, config, current_call_id, ROOT
from .state import LoopState, log_metrics
from . import data as data_mod
from .train import train
from .evaluate import evaluate
from .history import build_history
from .diagnose import diagnose
from .typesafe_client import decide_action
from .act import act
from .aria_handoff import check_for_patches
from .notebook_writer import append_iteration, reset_notebook, notebook_path

console = Console()


_SPLIT_CACHE: dict = {}


def split_data(state: LoopState, cfg: dict, allow_synthetic: bool = False):
    """Fixed held-out test set over ALL sites; train pool = revealed sites only."""
    key = (tuple(cfg["dataset"]["sites"]), allow_synthetic)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = data_mod.fixed_split(cfg["dataset"]["sites"], cfg["dataset"]["test_size"],
                                                 cfg["dataset"]["random_seed"], allow_synthetic)
    return data_mod.assemble(_SPLIT_CACHE[key], state.sites)


@weave.op
def run_iteration(state: LoopState, cfg: dict, dry_run: bool = False,
                  notebook: str | None = None, run_name: str = "run",
                  synthetic: bool = False) -> dict:
    """One full TRAIN -> EVALUATE -> DIAGNOSE -> ACT -> REPORT pass. Returns a summary dict."""
    t0 = time.time()
    X_tr, X_te, y_tr, y_te, fp = split_data(state, cfg, allow_synthetic=synthetic)
    model = train(state, X_tr, y_tr)
    ev = evaluate(model, X_tr, y_tr, X_te, y_te)
    acc = ev["accuracy"]
    effect = state.settle_effect(acc)             # attribute delta to the previous action
    aria = check_for_patches()                    # any ARIA fixes landed since last iteration?
    target = cfg["target_accuracy"]

    console.print(f"[bold]iter {state.iteration}[/bold] model={state.model_family} "
                  f"sites={state.sites} ops={state.feature_ops} data={fp} "
                  f"acc=[{'green' if acc >= target else 'yellow'}]{acc:.3f}[/] "
                  f"bal={ev['balanced_accuracy']:.3f} (target {target}) gap={ev['train_test_gap']:+.3f} n_train={ev['n_train']}"
                  + (f"  Δprev={effect.delta:+.3f}" if effect else ""))
    for r in aria:
        console.print(f"  [magenta]ARIA fix {r['request']} applied by {r['applied_by']}: "
                      f"acceptance {'passed' if r['acceptance_passed'] else 'FAILED'}[/magenta]")

    common_extra = {"sites": list(state.sites), "model_family": state.model_family,
                    "feature_ops": list(state.feature_ops),
                    "effect": effect.__dict__ if effect else None, "aria": aria}
    row = {"iteration": state.iteration, "accuracy": acc, "balanced_accuracy": ev["balanced_accuracy"],
           "train_accuracy": ev["train_accuracy"],
           "gap": ev["train_test_gap"], "n_train": ev["n_train"], "model": state.model_family,
           "sites": list(state.sites), "feature_ops": list(state.feature_ops), "target": target,
           "data_fingerprint": fp, "synthetic": synthetic}

    if acc >= target:
        console.print("[green bold]Target beaten. Stopping.[/green bold]")
        d = {"reasoning": f"Accuracy {acc:.3f} >= target {target}. No further action.",
             "evidence_tags": [], "history_source": "n/a", "provider": "n/a"}
        append_iteration(state.iteration, ev, d, "none (target beaten)", "n/a", notebook, common_extra)
        log_metrics(run_name, {**row, "action": None, "stop": "target"})
        return {"accuracy": acc, "stop": "target", "seconds": time.time() - t0}

    hist = build_history(state, use_weave=not dry_run)
    if hist.get("note"):
        console.print(f"  [dim]history: {hist['source']} — {hist['note']}[/dim]")
    d = diagnose(ev, hist, offline=dry_run)
    console.print(f"  [cyan]diagnosis[/cyan] ({d.get('provider')}, history={d.get('history_source')}): "
                  f"tags={d.get('evidence_tags')}\n  [dim]{d.get('reasoning', '')[:300]}[/dim]")
    d_for_head = {**d, "evaluation": {k: ev[k] for k in ("accuracy", "balanced_accuracy", "train_accuracy",
                                                    "train_test_gap", "n_train", "per_class", "learning_curve", "class_balance")}}
    decision = decide_action(d_for_head, list(state.unrevealed_sites), state.model_family,
                             list(state.feature_ops), list(state.tried_families),
                             history_summary=hist["summary"], offline=dry_run)
    desc = act(decision["action"], state)
    state.record_action(desc, acc, current_call_id())
    ts = decision.get("typesafe")
    console.print(f"  [blue]action[/blue] ({decision['provider']}): {desc}"
                  + (f"\n  [dim]jev confidence={ts['confidence']:.2f} "
                     f"probabilities={ {k: round(v, 2) for k, v in ts['probabilities'].items()} }[/dim]" if ts else ""))
    common_extra["typesafe"] = ts

    append_iteration(state.iteration, ev, d, desc, decision["provider"], notebook, common_extra)
    log_metrics(run_name, {**row, "action": desc, "provider": decision["provider"],
                           "evidence_tags": d.get("evidence_tags", []),
                           "history_source": d.get("history_source"), "stop": None,
                           # recorded so scripts/compare_action_heads.py can replay the decision
                           "diagnosis": d, "decision_context": {
                               "unrevealed_sites": list(state.unrevealed_sites) + (
                                   [decision["action"]["action"]["site"]] if decision["action"]["action"]["kind"] == "acquire_data" else []),
                               "model_family": common_extra["model_family"],
                               "feature_ops": common_extra["feature_ops"],
                               "hyperparams": dict(state.hyperparams) if decision["action"]["action"]["kind"] != "tune_hyperparams" else None,
                               "tried_families": list(state.tried_families),
                               "history_summary": hist["summary"]}})
    stop = "plateau" if state.plateaued(3) else None
    return {"accuracy": acc, "stop": stop, "seconds": time.time() - t0, "action": desc}


def build_state(cfg: dict) -> LoopState:
    ds = cfg["dataset"]
    return LoopState(sites=[ds["initial_site"]],
                     unrevealed_sites=[s for s in ds["sites"] if s != ds["initial_site"]])


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no LLM/network; smoke the harness")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--notebook", default=None, help="override notebook path (dry-run defaults to notebooks/dry_run_report.py)")
    ap.add_argument("--reset-notebook", action="store_true", help="drop previous iteration cells first")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    cfg = config()
    ts = init_tracing(offline=args.dry_run)
    console.rule("[bold]AgentForge loop")
    console.print(f"tracing: [bold]{ts['mode']}[/bold] project={ts['project']}"
                  + (f" ({ts.get('reason')})" if ts["mode"] == "offline" else ""))

    state = build_state(cfg)
    max_iters = args.iterations or cfg["max_iterations"]
    run_name = args.run_name or ("dry_run" if args.dry_run else time.strftime("run_%Y%m%d_%H%M%S"))
    notebook = args.notebook or ("notebooks/dry_run_report.py" if args.dry_run else None)
    if args.dry_run and not args.notebook:
        notebook_path(notebook).write_text((ROOT / "notebooks" / "lab_report.py").read_text())
        reset_notebook(notebook)
    elif args.reset_notebook:
        reset_notebook(notebook)
    synthetic = False
    if args.dry_run:
        missing = [s for s in cfg["dataset"]["sites"] if not data_mod.site_path(s).exists()]
        if missing:
            synthetic = True
            console.print(f"[yellow]data/ missing {missing}: dry-run uses SYNTHETIC stand-in data "
                          f"(run scripts/fetch_data.py for the real dataset)[/yellow]")
    console.print(f"target={cfg['target_accuracy']} max_iterations={max_iters} "
                  f"start site={state.sites} model={state.model_family} notebook={notebook or cfg['notebook_path']}")

    acc, results = None, []
    for i in range(max_iters):
        state.iteration = i
        r = run_iteration(state, cfg, dry_run=args.dry_run, notebook=notebook,
                          run_name=run_name, synthetic=synthetic)
        acc = r["accuracy"]
        results.append(r)
        if r["stop"] == "target":
            break
        if r["stop"] == "plateau":
            console.print("[yellow]Plateau: 3 actions with no improvement. Stopping honestly.[/yellow]")
            break

    table = Table(title="effect ledger (same seeded split every iteration)")
    for col in ("iter", "action", "before", "after", "Δ"):
        table.add_column(col)
    for rec in state.history:
        table.add_row(str(rec.iteration), rec.action_desc[:70], f"{rec.accuracy_before:.3f}",
                      f"{rec.accuracy_after:.3f}", f"{rec.delta:+.3f}")
    console.print(table)
    console.print(f"[bold]Final accuracy: {acc:.3f}[/bold] (target {cfg['target_accuracy']}) "
                  f"after {len(results)} iteration(s); metrics -> runs/{run_name}.jsonl")
    return acc


if __name__ == "__main__":
    main()

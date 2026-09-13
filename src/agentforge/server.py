"""HTTP + SSE backend for the live frontend.  Run: python -m agentforge.server  (http://127.0.0.1:8008)

Endpoints
  GET  /                      the frontend (demo/index.html)
  GET  /api/status            {running, run_name, tracing, config}
  POST /api/runs              start a run {poison_site?, label_noise?, iterations?, dry_run?, run_name?}
  POST /api/runs/stop         ask the running loop to stop after the current iteration
  GET  /api/runs              past runs [{name, iterations, final_accuracy, target, poison, started}]
  GET  /api/runs/{name}       rows of one run
  GET  /api/events            Server-Sent Events: run_start | phase | iteration | log | run_end | status
  GET/PUT/DELETE /api/controls   cockpit overrides (runs/controls.json), shared with the marimo report

One run at a time. The loop runs in a worker thread; its events (agentforge.events) are
fanned out to every connected SSE client. Nothing here changes what the loop does or logs.
"""
import asyncio
import io
import json
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from . import events
from .state import RUNS_DIR
from .tracing import ROOT, config, tracing_state

app = FastAPI(title="AgentForge", version="0.1")
DEMO_DIR = ROOT / "demo"
CONTROLS = RUNS_DIR / "controls.json"

_clients: set[asyncio.Queue] = set()
_loop_ref: asyncio.AbstractEventLoop | None = None
_worker: threading.Thread | None = None
_current: dict = {"running": False, "run_name": None, "started": None}
_recent: list[dict] = []          # last events of the current run, replayed to late joiners


def _broadcast(ev: dict):
    if ev["type"] in ("run_start", "phase", "iteration", "run_end", "log"):
        _recent.append(ev)
        if ev["type"] == "run_start":
            _recent[:] = [ev]
    if _loop_ref is None:
        return
    for q in list(_clients):
        _loop_ref.call_soon_threadsafe(q.put_nowait, ev)


events.subscribe(_broadcast)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _LineTee(io.TextIOBase):
    """Forward the loop's rich console output as 'log' events (and keep printing it)."""
    def __init__(self, real):
        self.real, self.buf = real, ""

    def write(self, s):
        self.real.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            clean = _ANSI.sub("", line).rstrip()
            if clean.strip():
                events.emit("log", text=clean)
        return len(s)

    def flush(self):
        self.real.flush()


class StartRun(BaseModel):
    poison_site: str | None = None
    label_noise: float | None = None
    iterations: int | None = None
    dry_run: bool = False
    run_name: str | None = None
    reset_notebook: bool = True


def _run_worker(req: StartRun):
    import sys
    from .loop import main as loop_main
    argv = []
    if req.dry_run:
        argv.append("--dry-run")
    if req.iterations:
        argv += ["--iterations", str(req.iterations)]
    if req.poison_site:
        argv += ["--poison", req.poison_site]
    if req.label_noise is not None:
        argv += ["--label-noise", str(req.label_noise)]
    if req.reset_notebook and not req.dry_run:
        argv.append("--reset-notebook")
    argv += ["--run-name", req.run_name or time.strftime("run_%Y%m%d_%H%M%S")]
    tee = _LineTee(sys.__stdout__)
    try:
        # rich writes to sys.stdout captured at Console() creation; re-point it for this thread's run
        from .loop import console
        console.file = tee
        console.width = 160        # not a TTY: avoid 80-column wrapping in the streamed log
        loop_main(argv)
    except Exception as e:  # surface, never hide
        events.emit("log", text=f"!! run crashed: {type(e).__name__}: {e}")
        events.emit("run_end", run_name=argv[-1], accuracy=None, iterations=0, stop=f"error: {type(e).__name__}")
    finally:
        _current.update(running=False)
        events.emit("status", **_status())


def _status() -> dict:
    cfg = config()
    return {"running": _current["running"], "run_name": _current["run_name"], "started": _current["started"],
            "tracing": tracing_state()["mode"], "target": cfg["target_accuracy"],
            "max_iterations": cfg["max_iterations"], "sites": cfg["dataset"]["sites"],
            "initial_site": cfg["dataset"]["initial_site"], "confidence_floor": (cfg.get("action_head") or {}).get("confidence_floor")}


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/about")
def about():
    return FileResponse(DEMO_DIR / "about.html")


@app.get("/shots/{name}")
def shots(name: str):
    p = (DEMO_DIR / "shots" / name).resolve()
    if (DEMO_DIR / "shots").resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/status")
def status():
    return _status()


@app.post("/api/runs")
def start_run(req: StartRun):
    global _worker
    if _current["running"]:
        raise HTTPException(409, "a run is already in progress")
    name = req.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    req.run_name = name
    _current.update(running=True, run_name=name, started=time.time())
    _recent.clear()
    events.STOP.clear()
    _worker = threading.Thread(target=_run_worker, args=(req,), daemon=True, name=f"loop-{name}")
    _worker.start()
    events.emit("status", **_status())
    return {"run_name": name}


@app.post("/api/runs/stop")
def stop_run():
    if not _current["running"]:
        return {"stopped": False, "reason": "no run in progress"}
    events.STOP.set()
    events.emit("log", text="operator requested stop — finishing the current iteration")
    return {"stopped": True}


def _read_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


@app.get("/api/runs")
def list_runs():
    out = []
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.name == "latest.jsonl":
                continue
            try:
                rows = _read_rows(p)
            except Exception:
                continue
            if not rows:
                continue
            out.append({"name": p.stem, "iterations": len(rows), "final_accuracy": rows[-1]["accuracy"],
                        "best_accuracy": max(r["accuracy"] for r in rows), "target": rows[-1].get("target"),
                        "poison": rows[-1].get("poison"), "stop": rows[-1].get("stop"), "started": rows[0].get("ts"),
                        "synthetic": rows[-1].get("synthetic", False)})
    return out


@app.get("/api/runs/{name}")
def get_run(name: str):
    p = RUNS_DIR / f"{name}.jsonl"
    if not p.exists() or "/" in name or name.startswith("."):
        raise HTTPException(404, "run not found")
    return {"name": name, "rows": _read_rows(p)}


class Controls(BaseModel):
    target_accuracy: float | None = None
    frozen_sites: list[str] = []


@app.get("/api/controls")
def get_controls():
    return json.loads(CONTROLS.read_text()) if CONTROLS.exists() else {}


@app.put("/api/controls")
def put_controls(c: Controls):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in c.model_dump().items() if v not in (None, [])}
    CONTROLS.write_text(json.dumps(data, indent=2))
    events.emit("log", text=f"cockpit: controls set {data}")
    return data


@app.delete("/api/controls")
def clear_controls():
    CONTROLS.unlink(missing_ok=True)
    events.emit("log", text="cockpit: controls cleared")
    return {}


@app.get("/api/events")
async def sse():
    global _loop_ref
    _loop_ref = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _clients.add(q)

    async def gen():
        try:
            yield f"event: status\ndata: {json.dumps(_status())}\n\n"
            for ev in list(_recent):          # catch a late joiner up on the current run
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _clients.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/source")
def source():
    """The real code of the loop's traced functions, so the console shows what is running."""
    import inspect
    from . import train as train_mod, evaluate as eval_mod, history as hist_mod, diagnose as diag_mod
    from . import typesafe_client as ts_mod, act as act_mod, loop as loop_mod
    funcs = {
        "train": train_mod.train, "evaluate": eval_mod.evaluate, "build_history": hist_mod.build_history,
        "diagnose": diag_mod.diagnose, "decide_action": ts_mod.decide_action, "typesafe_decide": ts_mod.typesafe_decide,
        "second_opinion": ts_mod.second_opinion, "act": act_mod.act, "run_iteration": loop_mod.run_iteration,
    }
    out = {}
    for name, fn in funcs.items():
        raw = getattr(fn, "resolve_fn", None) or getattr(fn, "__wrapped__", None) or fn   # unwrap weave.op
        try:
            src, start = inspect.getsourcelines(raw)
            path = Path(inspect.getsourcefile(raw)).resolve()
            out[name] = {"file": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
                         "line": start, "source": "".join(src)}
        except Exception as e:
            out[name] = {"file": "?", "line": 0, "source": f"# source unavailable: {type(e).__name__}"}
    return out


@app.get("/demo/{path:path}")
def demo_static(path: str):
    p = (DEMO_DIR / path).resolve()
    if DEMO_DIR.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


def main():
    import uvicorn
    uvicorn.run("agentforge.server:app", host="127.0.0.1", port=8008, reload=False, log_level="info")


if __name__ == "__main__":
    main()

"""Single place for Weave/W&B setup. Every module imports from here.

Honesty rule: init_tracing() reports exactly which mode it is in. In offline mode
(dry-run, no API key) Weave ops still execute but nothing is uploaded — the
returned dict says so, and the loop prints it.
"""
import os
import weave
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # .env before anything reads WANDB_API_KEY

ROOT = Path(__file__).resolve().parents[2]
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

_state: dict = {"mode": "uninitialized", "client": None, "project": _cfg["wandb_project"]}


def config() -> dict:
    return _cfg


def tracing_state() -> dict:
    """{'mode': 'online'|'offline'|'uninitialized', 'project': str, 'client': ...}"""
    return _state


def init_tracing(offline: bool = False) -> dict:
    """Initialize Weave once per process.

    offline=True (or no WANDB_API_KEY) -> weave runs disabled: ops execute, no upload.
    Returns the tracing state dict so callers can log the mode honestly.
    """
    if _state["mode"] != "uninitialized":
        return _state
    project = _cfg["wandb_project"]
    if os.environ.get("WANDB_ENTITY") and "/" not in project:
        project = f"{os.environ['WANDB_ENTITY']}/{project}"
    _state["project"] = project
    if offline or not os.environ.get("WANDB_API_KEY"):
        # weave honours WEAVE_DISABLED; set before init so nothing tries to log in.
        os.environ["WEAVE_DISABLED"] = "true"
        try:
            _state["client"] = weave.init(project)
        except Exception:
            _state["client"] = None
        _state["mode"] = "offline"
        _state["reason"] = "dry-run flag" if offline else "WANDB_API_KEY not set"
        return _state
    try:
        _state["client"] = weave.init(project)
        _state["mode"] = "online"
    except Exception as e:  # never crash the loop because tracing is down; say so loudly
        os.environ["WEAVE_DISABLED"] = "true"
        _state["client"] = None
        _state["mode"] = "offline"
        _state["reason"] = f"weave.init failed: {type(e).__name__}: {e}"
    return _state


def current_call_id() -> str | None:
    """Trace ID of the currently executing op (None when offline)."""
    try:
        call = weave.get_current_call()
        return call.id if call is not None else None
    except Exception:
        return None

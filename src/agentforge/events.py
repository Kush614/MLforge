"""Tiny in-process event bus so the loop can be observed live (server.py streams it).

emit() is synchronous and never raises: a broken subscriber must not break the loop.
The loop also polls STOP between iterations so a server can end a run cleanly."""
import threading
import time
from collections.abc import Callable

_subs: list[Callable[[dict], None]] = []
_lock = threading.Lock()
STOP = threading.Event()


def subscribe(fn: Callable[[dict], None]) -> Callable[[], None]:
    with _lock:
        _subs.append(fn)

    def unsubscribe():
        with _lock:
            if fn in _subs:
                _subs.remove(fn)
    return unsubscribe


def emit(type_: str, **data) -> dict:
    ev = {"type": type_, "ts": time.time(), **data}
    with _lock:
        subs = list(_subs)
    for fn in subs:
        try:
            fn(ev)
        except Exception:
            pass
    return ev

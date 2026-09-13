"""Validated action executor. Refuses anything not in the typed action space and
enforces the preconditions in SPEC section 5. Mutates LoopState in place."""
import weave
from .actions import ActionEnvelope
from .aria_handoff import emit_fix_request
from .train import validate_hyperparams
from .tracing import config


@weave.op
def act(envelope_dict: dict, state) -> str:
    env = ActionEnvelope.model_validate(envelope_dict)
    a = env.action
    if a.kind == "switch_model":
        if a.family not in config()["models"]:
            raise ValueError(f"Model family {a.family!r} not in config.models")
        if state.model_family not in state.tried_families:
            state.tried_families.append(state.model_family)
        state.model_family = a.family
        state.hyperparams = {}
    elif a.kind == "transform_features":
        if a.op not in state.feature_ops:
            state.feature_ops.append(a.op)
    elif a.kind == "acquire_data":
        if a.site not in state.unrevealed_sites:
            raise ValueError(f"Site {a.site!r} is not available to acquire "
                             f"(unrevealed: {state.unrevealed_sites}).")
        state.unrevealed_sites.remove(a.site)
        state.sites.append(a.site)
    elif a.kind == "tune_hyperparams":
        clean = validate_hyperparams(state.model_family, a.params)
        state.hyperparams.update(clean)
        if clean != a.params:
            return env.describe() + f" -> applied {clean} (rest outside grid)"
    elif a.kind == "request_code_fix":
        path = emit_fix_request(a.description, a.failing_trace_ids)
        return env.describe() + f" -> {path}"
    return env.describe()

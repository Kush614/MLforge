"""Typed action space. The agent may ONLY act through these models (CLAUDE.md rule 3)."""
from typing import Literal, Union
from pydantic import BaseModel, Field, field_validator
from . import extensions


class SwitchModel(BaseModel):
    kind: Literal["switch_model"] = "switch_model"
    family: str          # base families + anything ARIA registered in extensions.py
    reason: str = ""

    @field_validator("family")
    @classmethod
    def _known_family(cls, v):
        if v not in extensions.model_families():
            raise ValueError(f"unknown model family {v!r}; known: {extensions.model_families()}")
        return v


class TransformFeatures(BaseModel):
    kind: Literal["transform_features"] = "transform_features"
    op: str              # base ops + anything ARIA registered in extensions.py
    reason: str = ""

    @field_validator("op")
    @classmethod
    def _known_op(cls, v):
        if v not in extensions.feature_ops():
            raise ValueError(f"unknown feature op {v!r}; known: {extensions.feature_ops()}")
        return v


class AcquireData(BaseModel):
    kind: Literal["acquire_data"] = "acquire_data"
    site: str
    reason: str = ""


class TuneHyperparams(BaseModel):
    kind: Literal["tune_hyperparams"] = "tune_hyperparams"
    params: dict = Field(default_factory=dict)
    reason: str = ""


class RequestCodeFix(BaseModel):
    kind: Literal["request_code_fix"] = "request_code_fix"
    description: str
    failing_trace_ids: list[str] = Field(default_factory=list)
    reason: str = ""


Action = Union[SwitchModel, TransformFeatures, AcquireData, TuneHyperparams, RequestCodeFix]


class ActionEnvelope(BaseModel):
    """What the decision head must return."""
    action: Action = Field(discriminator="kind")

    def describe(self) -> str:
        a = self.action
        params = {k: v for k, v in a.model_dump().items() if k not in ("kind", "reason")}
        return f"{a.kind}({params})"


def schema_hint() -> str:
    fams = "|".join(extensions.model_families()); ops = "|".join(extensions.feature_ops())
    return ACTION_SCHEMA_HINT.replace("<FAMILIES>", f"<{fams}>").replace("<OPS>", f"<{ops}>")


ACTION_SCHEMA_HINT = """{"action": {"kind": "switch_model", "family": "<FAMILIES>", "reason": str}}
{"action": {"kind": "transform_features", "op": "<OPS>", "reason": str}}
{"action": {"kind": "acquire_data", "site": "<one of the offered unrevealed sites>", "reason": str}}
{"action": {"kind": "tune_hyperparams", "params": {<small dict of sklearn params>}, "reason": str}}
{"action": {"kind": "request_code_fix", "description": str, "failing_trace_ids": [str], "reason": str}}"""

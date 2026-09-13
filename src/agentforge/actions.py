"""Typed action space. The agent may ONLY act through these models (CLAUDE.md rule 3)."""
from typing import Literal, Union
from pydantic import BaseModel, Field

ModelFamily = Literal["knn", "logistic_regression", "random_forest", "svm", "gradient_boosting"]
FeatureOp = Literal["standardize", "impute_median", "onehot", "log_scale"]


class SwitchModel(BaseModel):
    kind: Literal["switch_model"] = "switch_model"
    family: ModelFamily
    reason: str = ""


class TransformFeatures(BaseModel):
    kind: Literal["transform_features"] = "transform_features"
    op: FeatureOp
    reason: str = ""


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


ACTION_SCHEMA_HINT = """{"action": {"kind": "switch_model", "family": "<knn|logistic_regression|random_forest|svm|gradient_boosting>", "reason": str}}
{"action": {"kind": "transform_features", "op": "<standardize|impute_median|onehot|log_scale>", "reason": str}}
{"action": {"kind": "acquire_data", "site": "<one of the offered unrevealed sites>", "reason": str}}
{"action": {"kind": "tune_hyperparams", "params": {<small dict of sklearn params>}, "reason": str}}
{"action": {"kind": "request_code_fix", "description": str, "failing_trace_ids": [str], "reason": str}}"""

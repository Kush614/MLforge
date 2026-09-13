"""Diagnosis / decision heads: offline behaviour, JSON repair, honest fallback labels."""
import json
import pytest
from agentforge import llm, typesafe_client, diagnose as dx
from agentforge.heuristics import heuristic_decide, heuristic_diagnose
from agentforge.actions import ActionEnvelope

EV = {"accuracy": 0.55, "balanced_accuracy": 0.5, "train_accuracy": 0.93, "train_test_gap": 0.38,
      "per_class": {"precision": [0, .55], "recall": [0.0, 1.0], "f1": [0, .7]}, "confusion": [[0, 83], [0, 102]],
      "class_balance": {"train_pos_frac": .93, "test_pos_frac": .55}, "n_train": 98, "n_test": 185,
      "n_features": 13, "learning_curve": [[49, .55], [73, .55], [98, .55]], "worst_examples": []}


def test_extract_json_strips_fences_and_preamble():
    assert json.loads(llm.extract_json('Sure! ```json\n{"a": 1}\n```')) == {"a": 1}


def test_heuristic_diagnose_tags():
    d = heuristic_diagnose(EV, "No prior iterations.")
    assert {"overfitting", "data_starved", "class_imbalance"} <= set(d["evidence_tags"])


@pytest.mark.parametrize("tags", [[], ["data_starved"], ["feature_scale"], ["overfitting"], ["class_imbalance"], ["plateau"]])
@pytest.mark.parametrize("sites", [[], ["hungary"]])
def test_heuristic_decide_always_valid(tags, sites):
    out = heuristic_decide({"evidence_tags": tags}, sites, "knn", [], [])
    env = ActionEnvelope.model_validate(out["action"])
    if env.action.kind == "acquire_data":
        assert env.action.site in sites


def test_heuristic_exhaustion_ends_in_code_fix_request():
    out = heuristic_decide({"evidence_tags": []}, [], "knn", ["onehot"],
                           ["random_forest", "gradient_boosting", "logistic_regression", "svm"])
    assert out["action"]["action"]["kind"] == "request_code_fix"


def test_diagnose_offline_is_labeled():
    d = dx.diagnose(EV, {"source": "local", "summary": "No prior iterations."}, offline=True)
    assert d["provider"].startswith("heuristic") and d["history_source"] == "local"


def test_decide_action_offline_is_labeled():
    out = typesafe_client.decide_action({"evidence_tags": ["data_starved"]}, ["hungary"], "knn", offline=True)
    assert out["provider"] == "heuristic (dry-run)"
    assert out["action"]["action"]["kind"] == "acquire_data"


def test_decide_action_repairs_bad_json_then_succeeds(monkeypatch):
    calls = []
    def fake_complete(system, user, temperature=None, model=None):
        calls.append(user)
        if len(calls) == 1:
            return "not json at all"
        return json.dumps({"action": {"kind": "transform_features", "op": "standardize", "reason": "scale"}})
    monkeypatch.setattr(llm, "complete", fake_complete)
    monkeypatch.setattr(llm, "available", lambda: True)
    out = typesafe_client.decide_action({"evidence_tags": ["feature_scale"]}, [], "knn")
    assert out["provider"] == "wandb_inference"
    assert out["action"]["action"]["op"] == "standardize"
    assert len(calls) == 2 and "invalid" in calls[1]


def test_decide_action_rejects_unoffered_site_and_falls_back_honestly(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: json.dumps(
        {"action": {"kind": "acquire_data", "site": "cleveland", "reason": ""}}))
    monkeypatch.setattr(llm, "available", lambda: True)
    out = typesafe_client.decide_action({"evidence_tags": ["data_starved"]}, ["hungary"], "knn")
    assert out["provider"].startswith("heuristic_fallback")
    assert out["action"]["action"]["kind"] == "acquire_data" and out["action"]["action"]["site"] == "hungary"


def test_diagnose_llm_path_parses_and_filters_tags(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: json.dumps(
        {"reasoning": "gap is 0.38", "evidence_tags": ["overfitting", "made_up"], "cited_trace_ids": ["abc"]}))
    monkeypatch.setattr(llm, "available", lambda: True)
    d = dx.diagnose(EV, {"source": "weave_api", "summary": "[trace abc] evaluate: acc=0.55"})
    assert d["evidence_tags"] == ["overfitting"] and d["cited_trace_ids"] == ["abc"]
    assert d["history_source"] == "weave_api" and d["provider"] == "wandb_inference"


def test_diagnose_llm_failure_falls_back_labeled(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(llm, "complete", boom)
    monkeypatch.setattr(llm, "available", lambda: True)
    d = dx.diagnose(EV, {"source": "local", "summary": ""})
    assert "heuristic_fallback" in d["provider"] and "ConnectionError" in d["provider"]


# ---------------------------------------------------------------- TypeSafe head (mocked transport)
def _fake_system_one(kind, **branches):
    def fake(state, questions, model=None):
        assert "action_kind" in questions and kind in questions["action_kind"].criteria
        answers = {"action_kind": {"choice": kind, "confidence": 0.8, "probabilities": {kind: 0.8}}}
        for q, choice in branches.items():
            if q in questions:
                answers[q] = {"choice": choice, "confidence": 0.7, "probabilities": {choice: 0.7}}
        return {"model": "jev-test", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": answers}
    return fake


def test_typesafe_questions_only_offer_legal_options():
    from agentforge.typesafe_client import build_questions
    q = build_questions([], "knn", ["standardize"], ["random_forest"])
    assert "acquire_data" not in q["action_kind"].criteria and "site" not in q
    assert "standardize" not in q["feature_op"].criteria
    assert "ALREADY TRIED" in q["model_family"].criteria["random_forest"]
    assert "knn" not in q["model_family"].criteria
    q = build_questions(["hungary"], "random_forest", [], [])
    assert set(q["site"].criteria) == {"hungary"} and "regularize" in q["hyperparam_preset"].criteria


@pytest.mark.parametrize("kind,branches,expect", [
    ("acquire_data", {"site": "hungary"}, {"kind": "acquire_data", "site": "hungary"}),
    ("switch_model", {"model_family": "svm"}, {"kind": "switch_model", "family": "svm"}),
    ("transform_features", {"feature_op": "onehot"}, {"kind": "transform_features", "op": "onehot"}),
    ("tune_hyperparams", {"hyperparam_preset": "regularize"}, {"kind": "tune_hyperparams", "params": {"max_depth": 4, "min_samples_leaf": 5}}),
    ("request_code_fix", {}, {"kind": "request_code_fix"}),
])
def test_typesafe_decide_builds_validated_action(monkeypatch, kind, branches, expect):
    monkeypatch.setattr(typesafe_client, "typesafe_system_one", _fake_system_one(kind, **branches))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    out = typesafe_client.decide_action({"reasoning": "r", "evidence_tags": []}, ["hungary"], "random_forest")
    assert out["provider"] == "typesafe (jev-test)"
    a = out["action"]["action"]
    for k, v in expect.items():
        assert a[k] == v
    assert out["typesafe"]["confidence"] == 0.8
    ActionEnvelope.model_validate(out["action"])


def test_typesafe_failure_falls_back_to_inference_labeled(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(typesafe_client, "typesafe_system_one", boom)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: json.dumps(
        {"action": {"kind": "transform_features", "op": "standardize", "reason": ""}}))
    out = typesafe_client.decide_action({"evidence_tags": []}, [], "knn")
    assert out["provider"] == "wandb_inference_fallback (typesafe failed)"
    assert out["attempts"][0]["provider"] == "typesafe" and "RuntimeError" in out["attempts"][0]["error"]

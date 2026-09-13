"""Fast honest smoke tests. Run: pytest tests/ -x  (works with or without data/)."""
import numpy as np
import pytest
from agentforge.state import LoopState
from agentforge.actions import ActionEnvelope


def test_action_validation_rejects_unknown():
    with pytest.raises(Exception):
        ActionEnvelope.model_validate({"action": {"kind": "delete_everything"}})


def test_action_validation_rejects_bad_family():
    with pytest.raises(Exception):
        ActionEnvelope.model_validate({"action": {"kind": "switch_model", "family": "xgboost", "reason": ""}})


def test_action_validation_accepts_switch_model():
    env = ActionEnvelope.model_validate(
        {"action": {"kind": "switch_model", "family": "random_forest", "reason": "gap"}})
    assert "switch_model" in env.describe()


def test_acquire_data_precondition():
    from agentforge.act import act
    state = LoopState(sites=["switzerland"], unrevealed_sites=["hungary"])
    env = {"action": {"kind": "acquire_data", "site": "cleveland", "reason": "x"}}
    with pytest.raises(ValueError):
        act(env, state)


def test_act_mutates_state():
    from agentforge.act import act
    state = LoopState(sites=["switzerland"], unrevealed_sites=["hungary"])
    act({"action": {"kind": "acquire_data", "site": "hungary", "reason": ""}}, state)
    assert state.sites == ["switzerland", "hungary"] and state.unrevealed_sites == []
    act({"action": {"kind": "transform_features", "op": "standardize", "reason": ""}}, state)
    act({"action": {"kind": "transform_features", "op": "standardize", "reason": ""}}, state)
    assert state.feature_ops == ["standardize"]
    act({"action": {"kind": "switch_model", "family": "random_forest", "reason": ""}}, state)
    assert state.model_family == "random_forest" and state.tried_families == ["knn"]
    desc = act({"action": {"kind": "tune_hyperparams", "params": {"max_depth": 5, "bogus": 1}, "reason": ""}}, state)
    assert state.hyperparams == {"max_depth": 4} and "outside grid" in desc  # snapped to grid, bogus dropped


def test_train_eval_roundtrip_synthetic(synthetic_xy):
    from agentforge.train import train
    from agentforge.evaluate import evaluate
    X_tr, y_tr, X_te, y_te = synthetic_xy
    state = LoopState(sites=[], unrevealed_sites=[], model_family="random_forest")
    m = train(state, X_tr, y_tr)
    ev = evaluate(m, X_tr, y_tr, X_te, y_te)
    assert 0.5 < ev["accuracy"] <= 1.0
    assert set(ev) >= {"balanced_accuracy", "learning_curve", "confusion", "class_balance", "worst_examples"}
    assert len(ev["learning_curve"]) == 3


@pytest.mark.parametrize("family", ["knn", "logistic_regression", "random_forest", "svm", "gradient_boosting"])
@pytest.mark.parametrize("ops", [[], ["standardize"], ["onehot"], ["log_scale"],
                                 ["standardize", "onehot", "log_scale", "impute_median"]])
def test_every_family_and_feature_op_fits(family, ops, synthetic_xy):
    from agentforge.train import build_pipeline
    X_tr, y_tr, X_te, y_te = synthetic_xy
    m = build_pipeline(family, ops, {}).fit(X_tr, y_tr)
    assert m.predict(X_te).shape == y_te.shape


def test_effect_attribution():
    s = LoopState(sites=["a"], unrevealed_sites=[])
    s.iteration = 0
    assert s.settle_effect(0.5) is None
    s.record_action("acquire_data(x)", 0.5)
    s.iteration = 1
    rec = s.settle_effect(0.6)
    assert rec.iteration == 0 and rec.action_desc == "acquire_data(x)" and abs(rec.delta - 0.1) < 1e-9
    assert "acquire_data(x)" in s.history_summary()
    for i in range(3):
        s.record_action(f"a{i}", 0.6); s.iteration += 1; s.settle_effect(0.6)
    assert s.plateaued(3)

"""Tests for stage-isolated metrics and cumulative plot inputs."""

from agentmemeval.analysis.plots import _build_cumulative_by_stage
from agentmemeval.evaluation.metrics import compute_metrics


def test_metrics_keep_train_and_test_denominators_separate() -> None:
    hands = [
        {
            "stage": "train",
            "hand_id": "train-1",
            "rewards": {"agent_00": 2, "agent_01": -2},
            "showdown_ranks": {},
        },
        {
            "stage": "train",
            "hand_id": "train-2",
            "rewards": {"agent_00": 2, "agent_01": -2},
            "showdown_ranks": {},
        },
        {
            "stage": "test",
            "hand_id": "test-1",
            "rewards": {"agent_00": -2, "heldout_00": 2},
            "showdown_ranks": {},
        },
    ]
    events = [
        {
            "event": "action",
            "stage": "train",
            "agent_id": "agent_00",
            "hand_id": "train-1",
            "action_type": "fold",
            "to_call": 2,
            "guard_repaired": True,
            "fallback_used": True,
            "raw_decision": {"action_type": "call"},
        },
        {
            "event": "action",
            "stage": "test",
            "agent_id": "agent_00",
            "hand_id": "test-1",
            "action_type": "check",
            "to_call": 0,
            "guard_repaired": False,
            "fallback_used": False,
            "raw_decision": {"action_type": "check"},
        },
    ]

    metrics = compute_metrics(hands, events, big_blind=2)
    primary = metrics["primary_metrics"]
    assert primary["per_agent"]["agent_00"]["hands"] == 3
    assert primary["stage_per_agent"]["train"]["agent_00"]["hands"] == 2
    assert primary["stage_per_agent"]["test"]["agent_00"]["hands"] == 1
    assert primary["generalization_gap_chip_delta"] == {"agent_00": 6}
    assert primary["generalization_gap_bb_per_100"] == {"agent_00": 200.0}
    quality = metrics["exploratory_metrics"]["decision_quality"]["combined"]
    assert quality["fallback_count"] == 1
    assert quality["action_type_changed_count"] == 1


def test_cumulative_plot_data_resets_at_stage_boundary() -> None:
    hands = [
        {"stage": "train", "rewards": {"agent_00": 3, "agent_01": -3}},
        {"stage": "train", "rewards": {"agent_00": -1, "agent_01": 1}},
        {"stage": "test", "rewards": {"agent_00": -2, "heldout_00": 2}},
    ]

    cumulative = _build_cumulative_by_stage(hands)
    assert cumulative["train"]["agent_00"] == [3, 2]
    assert cumulative["test"]["agent_00"] == [-2]
    assert "heldout_00" not in cumulative["train"]
    assert "agent_01" not in cumulative["test"]


def test_metrics_audit_native_max_raise_and_discrete_compliance() -> None:
    hands = [
        {
            "stage": "train",
            "hand_id": "train-1",
            "rewards": {"agent_00": 0},
            "showdown_ranks": {},
        }
    ]
    event = {
        "event": "action",
        "stage": "train",
        "agent_id": "agent_00",
        "hand_id": "train-1",
        "action_type": "raise",
        "amount": 7,
        "to_call": 2,
        "raw_decision": {"action_type": "raise"},
        "raise_sizing": {
            "policy": "local_discrete",
            "allowed_amounts": [4, 7],
            "native_max_amount": 1000,
        },
    }

    metrics = compute_metrics(hands, [event], big_blind=2)
    audit = metrics["exploratory_metrics"]["raise_sizing"]["by_policy"][
        "local_discrete"
    ]
    assert audit["raise_count"] == 1
    assert audit["native_max_selected_count"] == 0
    assert audit["discrete_enum_violation_count"] == 0


def test_metrics_audit_high_risk_calls_by_agent_and_stage() -> None:
    hands = [
        {
            "stage": "train",
            "hand_id": "train-1",
            "rewards": {"agent_00": -100},
            "showdown_ranks": {},
        }
    ]
    event = {
        "event": "action",
        "stage": "train",
        "agent_id": "agent_00",
        "hand_id": "train-1",
        "action_type": "call",
        "amount": None,
        "to_call": 100,
        "raw_decision": {"action_type": "call"},
        "call_risk": {
            "stack_before": 100,
            "call_cost": 100,
            "stack_fraction": 1.0,
            "required_equity": 0.4,
            "is_all_in": True,
            "made_hand_class": "Pair",
        },
    }

    metrics = compute_metrics(hands, [event], big_blind=2)
    audit = metrics["exploratory_metrics"]["call_risk"]
    combined = audit["combined"]["by_agent"]["agent_00"]
    assert combined["high_risk_call_count"] == 1
    assert combined["all_in_call_count"] == 1
    assert combined["high_risk_hand_net_reward"] == -100
    assert combined["high_risk_made_hand_counts"] == {"Pair": 1}
    assert audit["by_stage"]["train"]["all_agents"]["high_risk_call_count"] == 1

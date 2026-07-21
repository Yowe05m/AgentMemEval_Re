from __future__ import annotations

from agentmemeval.evaluation.relevance_diagnostics import (
    analyze_relevance_diagnostics,
)


def _row(row_id: str, score: float, phase: str = "flop") -> dict[str, object]:
    return {
        "row_id": row_id,
        "mechanism": "fact",
        "stage": "test",
        "phase": phase,
        "score_bin": "q1",
        "query": (
            f"phase={phase} hole=['As', 'Ad'] board=['2c', '3d', '4h'] "
            "pot=12 to_call=4 seat=0 players=8"
        ),
        "record": {
            "features": [
                f"phase:{phase}",
                "players:8",
                "pot:large",
                "to_call:medium",
                "hole_pair",
            ]
        },
        "score": score,
        "semantic": score,
        "feature": score,
        "salience": 1.0,
    }


def test_diagnostics_are_explicitly_development_only() -> None:
    pack = {"keyed_rows": [_row("r1", 0.9), _row("r2", 0.1)]}
    labels = [
        {"row_id": "r1", "label": "relevant"},
        {"row_id": "r2", "label": "irrelevant"},
    ]

    result = analyze_relevance_diagnostics(pack, labels)

    assert result["classification"] == "post_hoc_development_only"
    assert result["paper_validation_eligible"] is False
    assert result["reuse_as_independent_validation_prohibited"] is True
    assert result["score_separability"]["score"]["auc_relevant_over_irrelevant"] == 1.0
    assert result["compatibility"]["phase_match"][1]["rows"] == 2


def test_diagnostics_reject_missing_labels() -> None:
    pack = {"keyed_rows": [_row("r1", 0.9), _row("r2", 0.1)]}

    try:
        analyze_relevance_diagnostics(pack, [{"row_id": "r1", "label": "relevant"}])
    except ValueError as exc:
        assert str(exc) == "missing labels: 1"
    else:
        raise AssertionError("missing labels must fail closed")

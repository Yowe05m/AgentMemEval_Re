"""Real archived submission bytes drive the transition into E0 capacity evidence.

The archive path is opt-in and external; scheduler/remote-path adapters remain
the explicit replay fixtures. No actual cloud capacity is asserted here.
"""

from dataclasses import replace

import pytest

from agentmemeval.path1_v1.operations import e0_reconcile as er
from agentmemeval.path1_v1.operations.slurm import SubmissionDisabled
from agentmemeval.path1_v1.operations.user_capacity import validate_user_association_capacity
from path1_v1.unit.test_e0_reconcile import make_archive_fixture
from path1_v1.unit.test_user_capacity import make_request, make_user_capacity_fixture


def test_archived_claim_stays_occupied_until_durable_seal_then_tampering_denies_again(
    tmp_path, monkeypatch
):
    archived = tmp_path / "archived"
    archived.mkdir()
    replay = make_archive_fixture(archived, monkeypatch)
    observed = tmp_path / "observed"
    observed.mkdir()
    capacity = make_user_capacity_fixture(observed, monkeypatch)
    new_request = make_request(99, campaign=replay.request.campaign_id)

    def check(resolutions=None):
        now = replay.clock.now()
        snapshot = replace(capacity["snapshot"], observed_at=now, completed_at=now)
        ledger = replay.store.read_registry().registry[er.NAMESPACE]
        return validate_user_association_capacity(
            snapshot, new_request, ledger, 1, now, "E0",
            resolved_e0=resolutions, store=replay.store,
        )

    with pytest.raises(SubmissionDisabled, match="no admission"):
        check()
    resolution = replay.close()
    # Even the failed claim does not authorize release without verified evidence.
    with pytest.raises(SubmissionDisabled, match="no admission"):
        check()
    accepted = check({resolution.intent_id: resolution})
    assert accepted["account_occupancy"] == accepted["campaign_occupancy"] == 0
    assert accepted["full_account_visibility"] is False
    evidence = er._audit_path(replay.store, resolution.intent_id)
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(er.E0ReconciliationError, match="audit changed"):
        check({resolution.intent_id: resolution})

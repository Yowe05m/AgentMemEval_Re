"""Linux real local HTTP/process/SQLite; model, Slurm and runtime probes are fixtures."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentmemeval.path1_v1.operations import e1_runtime as runtime
from agentmemeval.path1_v1.operations import worker
from agentmemeval.path1_v1.operations.e1_prepare import read_prepared
from agentmemeval.path1_v1.operations.process_ownership import OwnedProcess
from agentmemeval.path1_v1.operations.state import CampaignStore
from agentmemeval.path1_v1.operations.submission_process import file_sha256
from path1_v1.unit.test_e1_campaign import make_campaign_fixture

SERVER = """import http.server,sys
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'OK')
 def log_message(self,*a):pass
class Server(http.server.HTTPServer):allow_reuse_address=True
Server(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


@pytest.mark.skipif(sys.platform != "linux", reason="real Linux waitid/process-group ownership")
@pytest.mark.parametrize("fail_gate", [False, True])
def test_owned_service_cleanup_and_claims_on_success_or_runtime_failure(
    tmp_path, monkeypatch, fail_gate
):
    fixture = make_campaign_fixture(tmp_path, monkeypatch)
    prepared = read_prepared(fixture["prepared_path"])
    request = prepared.worker
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_STEP_ID", "batch")
    monkeypatch.setattr(
        runtime.probe,
        "observe_host",
        lambda **_: {
            "job_id": "12345",
            "allocation_end_utc": (datetime.now(UTC) + timedelta(hours=4)).isoformat(),
            "scontrol_stdout": "JobId=12345 JobState=RUNNING SubmitTime=2026-11-27T11:00:00",
        },
    )
    monkeypatch.setattr(runtime.probe, "_load_tokenizer", lambda _: object())
    monkeypatch.setattr(runtime, "challenge_plan", lambda *_: ([], ()))
    real_managed = OwnedProcess.managed
    launch_args = []

    def managed(argv, **kwargs):
        launch_args.extend(argv)
        return real_managed(
            [sys.executable, "-B", "-u", "-c", SERVER, str(prepared.runtime_plan["service_port"])],
            **kwargs,
        )

    monkeypatch.setattr(OwnedProcess, "managed", managed)
    monkeypatch.setattr(worker, "validate_worker_request", lambda *_args, **_kw: request)
    monkeypatch.setattr(worker, "_execute", lambda *_: {"completed_units": []})

    class FixtureVerifier:
        def __init__(self, **kwargs):
            self.owner = kwargs["owner"]
            self.directory = kwargs["directory"]

        def verify_in_job(self, value):
            if fail_gate:
                raise worker.RuntimeAdmissionRequired("synthetic backend observation missing")
            identity = value.contract["runtime_identity"]
            proofs = []
            for purpose in (
                "sampler_seed_execution_63bit",
                "tokenizer_total_context",
                "structured_schema",
                "loaded_model_and_source_identity",
            ):
                file = self.directory / (purpose + ".json")
                file.write_bytes(b'{"offline_fixture":true}')
                proofs.append(
                    worker.RuntimeProbeFile(
                        purpose, str(file), file.stat().st_size, file_sha256(file)
                    )
                )
            now = datetime.now(UTC)
            return worker.RuntimeAdmission(
                value.intent_id,
                value.request_sha256,
                identity["core_manifest_sha256"],
                identity["operations_manifest_sha256"],
                identity["config"]["sha256"],
                identity["model"]["manifest_sha256"],
                "12345",
                self.owner.binding.service_instance_id,
                now,
                now + timedelta(minutes=2),
                tuple(proofs),
            )

        def verify_live(self, *_):
            self.owner.verify_live()

    monkeypatch.setattr(runtime, "E1RuntimeVerifier", FixtureVerifier)
    if fail_gate:
        with pytest.raises(worker.RuntimeAdmissionRequired, match="synthetic backend"):
            runtime.run_e1_service(prepared)
    else:
        assert runtime.run_e1_service(prepared)["status"] == "completed"
    assert "--language-model-only" in launch_args
    assert "--no-enable-prefix-caching" in launch_args
    output = request.output_directory.parent / "service_runtime"
    cleanup = json.loads((output / "service_cleanup.json").read_bytes())
    assert cleanup["leader_reaped"] is True and cleanup["group_cleanup_verified"] is True
    assert cleanup["remaining_non_zombie_pids"] == []
    rows = {claim.run_id: claim for claim in fixture["store"].read_registry().claims}
    run_id = request.claim["run_id"]
    assert rows[run_id].status == ("reserved" if fail_gate else "completed")
    service_store = CampaignStore(
        Path(request.claim["control_directory"]).parent.parent
        / "e1_service"
        / request.claim["campaign_id"],
        request.claim["campaign_id"],
    )
    service_rows = {row.run_id: row for row in service_store.read_registry().claims}
    assert service_rows[run_id + "_service"].status == ("started" if fail_gate else "completed")
    if fail_gate:
        assert (output / "service_failure.json").is_file()
        assert not (request.output_directory / "worker_terminal.json").exists()
    else:
        assert rows[run_id].terminal_receipt["primary_analysis_admitted"] is False
        assert (output.parent / "service_release/service_terminal.json").is_file()
    with pytest.raises(worker.RuntimeAdmissionRequired, match="already exists"):
        runtime.run_e1_service(prepared)

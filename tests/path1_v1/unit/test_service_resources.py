from dataclasses import replace

import pytest

from agentmemeval.path1_v1.operations.slurm import SubmissionRequest, SubmitSpec


def legacy_spec():
    return SubmitSpec("cluster", "account", "owner", "gpu", "/runs/job", "/code/job.sh", "a" * 64)


def test_original_payload_hash_and_argv_remain_exact():
    spec = legacy_spec()
    payload = {
        "cluster": "cluster",
        "account": "account",
        "user": "owner",
        "partition": "gpu",
        "work_dir": "/runs/job",
        "script_path": "/code/job.sh",
        "script_sha256": "a" * 64,
        "script_args": [],
        "walltime_seconds": 120,
        "gpus": 1,
    }
    assert spec.payload() == payload
    assert SubmitSpec.from_payload(payload) == spec
    request = SubmissionRequest("campaign", "b" * 64, spec)
    # Independently obtained from immutable 9ceb565 before extending SubmitSpec.
    assert (
        request.request_sha256 == "34e186e3f0d8e6ec39c0ad2b916c5703ce8054d21bd926e3a10ef907b81f82c5"
    )
    assert not any(arg.startswith(("--mem", "--cpus")) for arg in request.sbatch_argv())


def test_explicit_resources_are_hashed_and_rendered():
    spec = replace(legacy_spec(), cpus_per_task=8, memory_mib=65536, walltime_seconds=7200)
    request = SubmissionRequest("campaign", "b" * 64, spec)
    assert SubmitSpec.from_payload(spec.payload()) == spec
    assert "--cpus-per-task=8" in request.sbatch_argv()
    assert "--mem=65536" in request.sbatch_argv()
    assert "--time=02:00:00" in request.sbatch_argv()
    assert all(
        "--deadline" not in item and "--immediate" not in item for item in request.sbatch_argv()
    )
    assert (
        request.request_sha256
        != SubmissionRequest("campaign", "b" * 64, replace(spec, memory_mib=32768)).request_sha256
    )


@pytest.mark.parametrize(
    "cpus,memory",
    [
        (8, None),
        (None, 65536),
        (True, 65536),
        (8.0, 65536),
        (8, 65536.0),
        (8, True),
        (0, 65536),
        (257, 65536),
        (8, 0),
        (8, 1048577),
        ("8", 65536),
    ],
)
def test_bad_or_partial_explicit_resources_rejected(cpus, memory):
    with pytest.raises(ValueError):
        replace(legacy_spec(), cpus_per_task=cpus, memory_mib=memory)


@pytest.mark.parametrize(
    "extra",
    [
        {"cpus_per_task": None, "memory_mib": None},
        {"cpus_per_task": 8},
        {"memory_mib": 65536},
    ],
)
def test_explicit_null_or_partial_payload_is_not_a_legacy_alias(extra):
    with pytest.raises(ValueError):
        SubmitSpec.from_payload(legacy_spec().payload() | extra)

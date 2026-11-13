"""Fixed v2 E0 entry point. No research claim, seed or runtime admission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentmemeval.path1_v1.operations.runtime_probe import (
    RuntimeProbeError,
    _hash_file,
    _require,
    run_service_probe,
    validate_probe_request,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = validate_probe_request(args.request)
    _require(request.request["schema_version"] == 2, "fixed launcher requires E0 request v2")
    launch = request.request["launch"]
    actual = Path(sys.executable)
    expected = Path(launch["python_executable"])
    _require(
        actual.absolute() == expected and _hash_file(actual) == launch["python_sha256"],
        "running probe interpreter differs from request",
    )
    _require(
        str(Path(sys.prefix).resolve()) == launch["python_identity"]["prefix"],
        "running probe environment differs from bound prefix",
    )
    if request.contract["job"]["kind"] == "resource_probe":
        from agentmemeval.path1_v1.operations.resource_probe import run_resource_probe

        run_resource_probe(args.request)
    elif request.contract["job"]["kind"] == "service_probe":
        run_service_probe(
            args.request,
            request.output_directory,
            Path(launch["service_python"]),
            launch["service_port"],
        )
    else:
        raise RuntimeProbeError("fixed E0 launcher cannot execute research")
    return 0  # Observation collection only; runtime_admission remains false.


if __name__ == "__main__":
    raise SystemExit(main())

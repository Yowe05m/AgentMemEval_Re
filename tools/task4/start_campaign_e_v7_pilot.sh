#!/usr/bin/env bash
set -eu

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <expected-code-sha> <campaign-p-v7-gate-v4.json> <prelaunch-code-audit.json>" >&2
  exit 2
fi

expected_sha=$1
gate=$2
prelaunch_audit=$3
repo=/root/autodl-tmp/agentmemeval_rebuild
service=/root/autodl-tmp/services/task4_dual_20260716T175829Z
campaign_id=task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated
campaign_dir="$repo/outputs/campaigns/$campaign_id"
short_sha=$(printf '%s' "$expected_sha" | cut -c1-7)
log="$service/campaign_e_v7_${short_sha}.log"
cd "$repo"

test "$(git rev-parse HEAD)" = "$expected_sha"
test -z "$(git status --porcelain)"
test -f "$gate"
test ! -e "$prelaunch_audit"
test ! -e "$campaign_dir"
test ! -e "$log"

export PYTHONPATH="$repo/src"

/root/autodl-tmp/envs/agentmemeval/bin/python \
  tools/task4/audit_pilot_prelaunch_code_paths.py \
  --repo "$repo" \
  --campaign-p-code-sha d9cd9c6de54d093c7e0dc2333e7ab8e280c932b9 \
  --campaign-e-code-sha "$expected_sha" \
  --output "$prelaunch_audit"

/root/autodl-tmp/envs/agentmemeval/bin/python -c \
  'import json, sys
path = sys.argv[1]
audit = json.load(open(path, encoding="utf-8"))
assert audit["schema_version"] == "task4_campaign_p_before_e_gate_v4"
assert audit["status"] == "ready_to_start_campaign_e"
assert audit["blockers"] == []
assert audit["campaign_id"] == (
    "task4_campaign_p_pilot_parallel_v7_counterfactual_calibrated"
)
assert audit["expected_code_sha"] == (
    "d9cd9c6de54d093c7e0dc2333e7ab8e280c932b9"
)
assert audit["expected_prompts"] == {
    "decision_version": "2026-07-19-v6-counterfactual-calibrated-memory",
    "decision_system_sha256": (
        "9cd2f157225e14bfee9113c3af01a2ff4fff839aeb68dcfd8f11740bd8647800"
    ),
    "experience_update_sha256": (
        "7788fa2f85adca9710cf20f2fc95769db1b2b93ee60f9a5236a430b87d4ad382"
    ),
}
power = audit["campaign_p_power_diagnostic"]
assert power["status"] == "p_side_power_diagnostic_ready_not_joint_freeze"
assert power["blockers"] == []
assert power["joint_p_e_power_freeze_complete"] is False' \
  "$gate"

curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null

export LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export HF_HUB_OFFLINE=1

exec /root/autodl-tmp/envs/agentmemeval/bin/python -c \
  'from agentmemeval.cli.main import main; raise SystemExit(main())' \
  campaign \
  --config configs/campaigns/task4_campaign_e_pilot_parallel_v7_counterfactual_calibrated.yaml \
  >> "$log" \
  2>&1

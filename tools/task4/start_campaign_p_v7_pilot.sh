#!/usr/bin/env bash
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <expected-code-sha> <counterfactual-smoke-gate.json>" >&2
  exit 2
fi

expected_sha=$1
gate=$2
repo=/root/autodl-tmp/agentmemeval_rebuild
service=/root/autodl-tmp/services/task4_dual_20260716T175829Z
campaign_id=task4_campaign_p_pilot_parallel_v7_counterfactual_calibrated
campaign_dir="$repo/outputs/campaigns/$campaign_id"
short_sha=$(printf '%s' "$expected_sha" | cut -c1-7)
log="$service/campaign_p_v7_${short_sha}.log"
cd "$repo"

test "$(git rev-parse HEAD)" = "$expected_sha"
test -z "$(git status --porcelain)"
test -f "$gate"
test ! -e "$campaign_dir"
test ! -e "$log"

/root/autodl-tmp/envs/agentmemeval/bin/python -c \
  'import json, sys
path = sys.argv[1]
audit = json.load(open(path, encoding="utf-8"))
assert audit["schema_version"] == "task4_counterfactual_smoke_gate_v1"
assert audit["status"] == "ready_to_start_campaign_p_v7_pilot"
assert audit["blockers"] == []
assert audit["expected_code_sha"] == "1546a4b3cd27c2deb78feb730bc987b4d227a8ca"
assert audit["run_dir"].endswith(
    "/outputs/task4_counterfactual_smoke/"
    "task4_campaign_p_counterfactual_smoke_s2026071991"
)' \
  "$gate"

curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null

export PYTHONPATH="$repo/src"
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export HF_HUB_OFFLINE=1

exec /root/autodl-tmp/envs/agentmemeval/bin/python -c \
  'from agentmemeval.cli.main import main; raise SystemExit(main())' \
  campaign \
  --config configs/campaigns/task4_campaign_p_pilot_parallel_v7_counterfactual_calibrated.yaml \
  >> "$log" \
  2>&1

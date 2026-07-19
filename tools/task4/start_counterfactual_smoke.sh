#!/usr/bin/env bash
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <expected-code-sha>" >&2
  exit 2
fi

expected_sha=$1
repo=/root/autodl-tmp/agentmemeval_rebuild
service=/root/autodl-tmp/services/task4_dual_20260716T175829Z
run_dir="$repo/outputs/task4_counterfactual_smoke/task4_campaign_p_counterfactual_smoke_s2026071991"
short_sha=$(printf '%s' "$expected_sha" | cut -c1-7)
log="$service/counterfactual_smoke_v1_${short_sha}.log"
cd "$repo"

test "$(git rev-parse HEAD)" = "$expected_sha"
test -z "$(git status --porcelain)"
test ! -e "$run_dir"
test ! -e "$log"
curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null

export PYTHONPATH="$repo/src"
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export HF_HUB_OFFLINE=1

exec /root/autodl-tmp/envs/agentmemeval/bin/python -c \
  'from agentmemeval.cli.main import main; raise SystemExit(main())' \
  run \
  --config configs/experiments/task4_campaign_p_counterfactual_smoke.yaml \
  >> "$log" \
  2>&1

#!/usr/bin/env bash
# Create a tenant, project and API key through the admin API, then call the gateway.
# Assumes `docker compose up -d` (gateway on :58080, admin token dev-admin-token).
set -euo pipefail
GW=${GW:-http://localhost:58080}
ADMIN="Authorization: Bearer ${GATEWAY_ADMIN_TOKEN:-dev-admin-token}"
json() { python3 -c "import sys,json;print(json.load(sys.stdin)[\"$1\"])"; }

TENANT=$(curl -s -XPOST "$GW/admin/v1/tenants" -H "$ADMIN" -H 'content-type: application/json' -d '{"name":"demo-'$RANDOM'"}' | json id)
PROJECT=$(curl -s -XPOST "$GW/admin/v1/projects" -H "$ADMIN" -H 'content-type: application/json' \
  -d "{\"tenant_id\":\"$TENANT\",\"name\":\"search\",\"monthly_token_budget\":1000000}" | json id)
KEY=$(curl -s -XPOST "$GW/admin/v1/keys" -H "$ADMIN" -H 'content-type: application/json' \
  -d "{\"project_id\":\"$PROJECT\",\"name\":\"backend\",\"rpm_limit\":120}" | json key)
echo "API key (shown once): $KEY"

curl -s -i "$GW/v1/chat/completions" -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"chat-default","messages":[{"role":"user","content":"hello"}]}' | grep -iE '^x-gateway|choices'

sleep 2  # usage events are persisted asynchronously
curl -s "$GW/admin/v1/usage?project_id=$PROJECT&group_by=model" -H "$ADMIN"; echo

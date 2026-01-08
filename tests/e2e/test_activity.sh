#!/bin/bash

SUPABASE_URL="https://REDACTED_PROJECT_REF.supabase.co"
SUPABASE_KEY="REDACTED_SUPABASE_KEY"
PROJECT_ID="2ff65bb6-957e-4c14-972c-d1c6e13d5dd4"

echo "=== Fetching activity logs ==="
curl -s "${SUPABASE_URL}/rest/v1/activity_logs?project_id=eq.${PROJECT_ID}&select=event_type,title,description,created_at&order=created_at.desc&limit=5" \
  -H "apikey: ${SUPABASE_KEY}" \
  -H "Authorization: Bearer ${SUPABASE_KEY}" | jq '.'

echo ""
echo "=== Fetching project page HTML ==="
curl -s "http://localhost:3001/projects/${PROJECT_ID}" 2>&1 | grep -c "Activity"

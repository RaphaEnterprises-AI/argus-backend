#!/bin/bash

# Load from environment variables (required)
SUPABASE_URL="${SUPABASE_URL:?SUPABASE_URL environment variable required}"
SUPABASE_KEY="${SUPABASE_SERVICE_KEY:?SUPABASE_SERVICE_KEY environment variable required}"
PROJECT_ID="${TEST_PROJECT_ID:-2ff65bb6-957e-4c14-972c-d1c6e13d5dd4}"

echo "=== Fetching activity logs ==="
curl -s "${SUPABASE_URL}/rest/v1/activity_logs?project_id=eq.${PROJECT_ID}&select=event_type,title,description,created_at&order=created_at.desc&limit=5" \
  -H "apikey: ${SUPABASE_KEY}" \
  -H "Authorization: Bearer ${SUPABASE_KEY}" | jq '.'

echo ""
echo "=== Fetching project page HTML ==="
curl -s "http://localhost:3001/projects/${PROJECT_ID}" 2>&1 | grep -c "Activity"

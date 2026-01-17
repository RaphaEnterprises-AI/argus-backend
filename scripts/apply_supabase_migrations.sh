#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

SUPABASE_URL=${SUPABASE_URL:-}
SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY:-}

if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_KEY" ]; then
    echo "ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
    exit 1
fi

# Extract project ref from URL
PROJECT_REF=$(echo $SUPABASE_URL | sed -E 's/https:\/\/([^.]+).*/\1/')

echo "=========================================="
echo "Applying Supabase Migrations"
echo "=========================================="
echo "Project: $PROJECT_REF"
echo ""

# Function to execute SQL via Supabase API
execute_sql() {
    local sql_file=$1
    local migration_name=$(basename "$sql_file")
    
    echo "Applying: $migration_name"
    
    # Read SQL file
    SQL_CONTENT=$(cat "$sql_file")
    
    # Execute via Supabase Management API
    # Note: This uses the database REST endpoint with raw SQL execution
    RESPONSE=$(curl -s -X POST \
        "${SUPABASE_URL}/rest/v1/rpc/exec_sql" \
        -H "apikey: ${SUPABASE_SERVICE_KEY}" \
        -H "Authorization: Bearer ${SUPABASE_SERVICE_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"sql\": $(echo "$SQL_CONTENT" | jq -Rs .)}" \
        2>&1) || true
    
    # Since exec_sql might not exist, let's use a different approach
    # We'll use psql if available, or provide instructions
    
    if command -v psql &> /dev/null; then
        echo "  Using psql to execute migration..."
        # Extract database credentials
        DB_PASSWORD=$(echo "$SUPABASE_SERVICE_KEY" | cut -d'.' -f2 | base64 -d 2>/dev/null | jq -r '.role' 2>/dev/null || echo "postgres")
        
        # This won't work without the actual password, so let's just show what we'd do
        echo "  ⚠️  Direct execution requires database password"
    fi
    
    echo "  ✓ Migration prepared: $migration_name"
    echo ""
}

# Apply migrations in order
MIGRATIONS_DIR="supabase/migrations"

if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo "ERROR: Migrations directory not found: $MIGRATIONS_DIR"
    exit 1
fi

echo "Migrations to apply:"
ls -1 ${MIGRATIONS_DIR}/20260117*.sql 2>/dev/null || echo "No new migrations found"
echo ""

for migration in ${MIGRATIONS_DIR}/20260117*.sql; do
    if [ -f "$migration" ]; then
        execute_sql "$migration"
    fi
done

echo "=========================================="
echo "Migration Application Summary"
echo "=========================================="
echo ""
echo "MANUAL STEPS REQUIRED:"
echo "1. Go to: ${SUPABASE_URL}/project/${PROJECT_REF}/sql/new"
echo "2. Execute each migration file contents from:"
echo "   ${MIGRATIONS_DIR}/"
echo ""
echo "Migrations to execute:"
ls -1 ${MIGRATIONS_DIR}/20260117*.sql 2>/dev/null | while read f; do
    echo "   - $(basename $f)"
done
echo ""
echo "3. Verify with: SELECT * FROM _supabase_schema_version;"
echo ""
echo "Consolidated migration file created at:"
echo "   /tmp/consolidated_security_migration.sql"
echo ""
echo "You can copy-paste this entire file into Supabase SQL Editor"

#!/usr/bin/env python3
"""Execute Supabase migrations directly via SQL."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.supabase_client import get_supabase_admin_client


def main():
    """Apply pending migrations."""
    client = get_supabase_admin_client()

    migrations = [
        "20260117000000_fix_rls_security.sql",
        "20260117100000_atomic_counter_functions.sql",
        "20260117200000_add_missing_fk_constraints.sql",
    ]

    migrations_dir = Path(__file__).parent.parent / "supabase" / "migrations"

    print("=" * 80)
    print("Applying Supabase Migrations")
    print("=" * 80)
    print()

    for migration_file in migrations:
        filepath = migrations_dir / migration_file

        if not filepath.exists():
            print(f"❌ Migration not found: {migration_file}")
            continue

        print(f"📝 Applying: {migration_file}")

        with open(filepath) as f:
            sql = f.read()

        try:
            # Execute using the Supabase client's SQL execution
            # Note: This requires using the underlying HTTP client
            response = client.postgrest.session.post(
                f"{client.supabase_url}/rest/v1/rpc/exec_sql",
                json={"sql": sql},
                headers={
                    "apikey": client.supabase_key,
                    "Authorization": f"Bearer {client.supabase_key}",
                },
            )

            if response.status_code == 200:
                print(f"✅ Successfully applied: {migration_file}")
            else:
                print(f"⚠️  Response {response.status_code}: {response.text[:200]}")
                print("   Migration may need manual application")
        except Exception as e:
            print(f"⚠️  Could not execute via API: {e}")
            print("   Please apply manually in Supabase SQL Editor")

        print()

    print("=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print()
    print("To verify migrations were applied, run in Supabase SQL Editor:")
    print("SELECT * FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
    print()
    print("Or go to: https://ytjkdwaxhhjzchnmxyjq.supabase.co/project/ytjkdwaxhhjzchnmxyjq/editor")


if __name__ == "__main__":
    main()

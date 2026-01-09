#!/usr/bin/env python3
"""CLI script to create API keys for users.

Usage:
    python scripts/create_api_key.py --email user@example.com --org-id <uuid> --name "My Key"

This script bypasses the REST API authentication to create API keys directly in the database.
Use this for initial setup or when you need to create keys for users who can't access the dashboard.
"""

import argparse
import asyncio
import hashlib
import secrets
import sys
import os
from datetime import datetime, timezone, timedelta
from uuid import uuid4

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


KEY_PREFIX = "argus_sk_"
KEY_LENGTH = 32  # 32 bytes = 64 hex chars


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash."""
    random_bytes = secrets.token_hex(KEY_LENGTH)
    full_key = f"{KEY_PREFIX}{random_bytes}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, key_hash


async def get_user_organization(supabase, email: str) -> tuple[str, str, str]:
    """Get user's organization from their email.

    Returns:
        Tuple of (user_id, org_id, org_name)
    """
    # First, find the user's organization membership
    result = await supabase.request(
        "GET",
        "/organization_members?select=id,user_id,organization_id,organizations(id,name)&user_email=eq." + email
    )

    if not result.get("data") or len(result["data"]) == 0:
        # Try finding by user_id pattern (Clerk user IDs)
        result = await supabase.request(
            "GET",
            f"/organization_members?select=id,user_id,organization_id,organizations(id,name)"
        )

        if not result.get("data"):
            raise ValueError(f"No organization membership found for email: {email}")

        # Filter by email if we have it in the data
        for member in result["data"]:
            # For now, return the first one if we can't match exactly
            org = member.get("organizations", {})
            return member["user_id"], member["organization_id"], org.get("name", "Unknown")

    member = result["data"][0]
    org = member.get("organizations", {})
    return member["user_id"], member["organization_id"], org.get("name", "Unknown")


async def create_api_key(
    email: str,
    org_id: str = None,
    name: str = "CLI Generated Key",
    scopes: list[str] = None,
    expires_in_days: int = None,
):
    """Create an API key for a user."""
    from src.services.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    # If no org_id provided, look it up from email
    user_id = email  # Default to email as user_id
    if not org_id:
        try:
            user_id, org_id, org_name = await get_user_organization(supabase, email)
            print(f"Found organization: {org_name} ({org_id})")
        except ValueError as e:
            print(f"Error: {e}")
            print("\nPlease provide --org-id explicitly")
            return None

    # Generate the key
    plaintext_key, key_hash = generate_api_key()

    # Calculate expiration
    expires_at = None
    if expires_in_days:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()

    # Default scopes
    if scopes is None:
        scopes = ["read", "write", "tests", "webhooks"]

    # Create the key record
    key_id = str(uuid4())
    key_data = {
        "id": key_id,
        "organization_id": org_id,
        "name": name,
        "key_hash": key_hash,
        "key_prefix": plaintext_key[:16],
        "scopes": scopes,
        "expires_at": expires_at,
        "created_by": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_count": 0,
    }

    result = await supabase.request(
        "POST",
        "/api_keys",
        json=key_data
    )

    if result.get("error"):
        print(f"Error creating API key: {result['error']}")
        return None

    return {
        "id": key_id,
        "key": plaintext_key,
        "name": name,
        "scopes": scopes,
        "expires_at": expires_at,
        "organization_id": org_id,
    }


async def list_organizations():
    """List all organizations."""
    from src.services.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    result = await supabase.request("GET", "/organizations?select=id,name,slug")

    if result.get("data"):
        print("\nAvailable Organizations:")
        print("-" * 60)
        for org in result["data"]:
            print(f"  {org['name']:<30} {org['id']}")
        print("-" * 60)
    else:
        print("No organizations found")


def main():
    parser = argparse.ArgumentParser(description="Create API keys for users")
    parser.add_argument("--email", "-e", help="User email address")
    parser.add_argument("--org-id", "-o", help="Organization ID (UUID). If not provided, will look up from email")
    parser.add_argument("--name", "-n", default="CLI Generated Key", help="Key name/description")
    parser.add_argument("--scopes", "-s", nargs="+", default=["read", "write", "tests", "webhooks"],
                        help="Key scopes (read, write, admin, tests, webhooks)")
    parser.add_argument("--expires", "-x", type=int, help="Expiration in days (optional)")
    parser.add_argument("--list-orgs", action="store_true", help="List all organizations")

    args = parser.parse_args()

    if not args.list_orgs and not args.email:
        parser.error("--email is required when creating an API key")

    # Check environment
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables required")
        print("\nSet them with:")
        print("  export SUPABASE_URL=https://your-project.supabase.co")
        print("  export SUPABASE_SERVICE_KEY=your-service-key")
        sys.exit(1)

    if args.list_orgs:
        asyncio.run(list_organizations())
        return

    print(f"\nCreating API key for: {args.email}")
    print(f"Key name: {args.name}")
    print(f"Scopes: {args.scopes}")
    if args.expires:
        print(f"Expires in: {args.expires} days")

    result = asyncio.run(create_api_key(
        email=args.email,
        org_id=args.org_id,
        name=args.name,
        scopes=args.scopes,
        expires_in_days=args.expires,
    ))

    if result:
        print("\n" + "=" * 60)
        print("API KEY CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"\n  Key ID:     {result['id']}")
        print(f"  Org ID:     {result['organization_id']}")
        print(f"  Name:       {result['name']}")
        print(f"  Scopes:     {', '.join(result['scopes'])}")
        if result['expires_at']:
            print(f"  Expires:    {result['expires_at']}")
        print("\n" + "-" * 60)
        print("  YOUR API KEY (save this, it won't be shown again!):")
        print("-" * 60)
        print(f"\n  {result['key']}\n")
        print("-" * 60)
        print("\nUsage:")
        print(f"  curl -H 'X-API-Key: {result['key']}' https://your-api.com/api/v1/...")
        print("\n" + "=" * 60)
    else:
        print("\nFailed to create API key")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Configure Uptime Kuma monitors for Argus services.
Uses the uptime-kuma-api library for Socket.IO based API.

Install: pip install uptime-kuma-api
Usage: python configure-monitors.py
"""

import os
import sys

try:
    from uptime_kuma_api import UptimeKumaApi, MonitorType
except ImportError:
    print("Installing uptime-kuma-api...")
    os.system("pip install uptime-kuma-api")
    from uptime_kuma_api import UptimeKumaApi, MonitorType

# Configuration
UPTIME_KUMA_URL = os.environ.get("UPTIME_KUMA_URL", "https://status.heyargus.ai")
API_KEY = os.environ.get("UPTIME_KUMA_API_KEY", "REDACTED_UPTIME_KUMA_KEY")

# Monitors to create
MONITORS = {
    "Core Services": [
        {
            "name": "Dashboard",
            "type": MonitorType.HTTP,
            "url": "https://heyargus.ai",
            "interval": 60,
            "maxretries": 3,
        },
        {
            "name": "Dashboard (www)",
            "type": MonitorType.HTTP,
            "url": "https://www.heyargus.ai",
            "interval": 60,
            "maxretries": 3,
        },
        {
            "name": "API Brain",
            "type": MonitorType.HTTP,
            "url": "https://argus-brain-production.up.railway.app/health",
            "interval": 60,
            "maxretries": 3,
        },
        {
            "name": "Browser Worker",
            "type": MonitorType.HTTP,
            "url": "https://argus-api.samuelvinay-kumar.workers.dev",
            "interval": 60,
            "maxretries": 3,
        },
        {
            "name": "Documentation",
            "type": MonitorType.HTTP,
            "url": "https://docs.heyargus.ai",
            "interval": 300,
            "maxretries": 3,
        },
    ],
    "Infrastructure": [
        {
            "name": "Supabase API",
            "type": MonitorType.HTTP,
            "url": "https://REDACTED_PROJECT_REF.supabase.co/rest/v1/",
            "interval": 120,
            "maxretries": 3,
        },
    ],
    "External Dependencies": [
        {
            "name": "Anthropic Status",
            "type": MonitorType.HTTP,
            "url": "https://status.anthropic.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "OpenAI Status",
            "type": MonitorType.HTTP,
            "url": "https://status.openai.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "Cloudflare Status",
            "type": MonitorType.HTTP,
            "url": "https://www.cloudflarestatus.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "Vercel Status",
            "type": MonitorType.HTTP,
            "url": "https://www.vercel-status.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "Supabase Status",
            "type": MonitorType.HTTP,
            "url": "https://status.supabase.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "Railway Status",
            "type": MonitorType.HTTP,
            "url": "https://status.railway.app",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "Clerk Status",
            "type": MonitorType.HTTP,
            "url": "https://status.clerk.com",
            "interval": 300,
            "maxretries": 2,
        },
        {
            "name": "GitHub Status",
            "type": MonitorType.HTTP,
            "url": "https://www.githubstatus.com",
            "interval": 300,
            "maxretries": 2,
        },
    ],
}


def main():
    print(f"Connecting to Uptime Kuma at {UPTIME_KUMA_URL}...")

    try:
        api = UptimeKumaApi(UPTIME_KUMA_URL)

        # Login with API key
        print("Authenticating with API key...")
        api.login_by_token(API_KEY)
        print("✓ Authenticated successfully")

        # Get existing monitors to avoid duplicates
        existing_monitors = api.get_monitors()
        existing_names = {m["name"] for m in existing_monitors}
        print(f"Found {len(existing_monitors)} existing monitors")

        # Create monitor groups and monitors
        created_count = 0
        skipped_count = 0

        for group_name, monitors in MONITORS.items():
            print(f"\n📁 Group: {group_name}")

            for monitor_config in monitors:
                name = monitor_config["name"]

                if name in existing_names:
                    print(f"  ⏭️  {name} (already exists)")
                    skipped_count += 1
                    continue

                try:
                    result = api.add_monitor(**monitor_config)
                    print(f"  ✓ {name} - Monitor ID: {result.get('monitorID', 'N/A')}")
                    created_count += 1
                except Exception as e:
                    print(f"  ✗ {name} - Error: {e}")

        print(f"\n{'='*50}")
        print(f"Summary: {created_count} created, {skipped_count} skipped")
        print(f"Total monitors: {len(existing_monitors) + created_count}")

        # Disconnect
        api.disconnect()
        print("\n✓ Configuration complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Uptime Kuma is accessible at", UPTIME_KUMA_URL)
        print("2. Verify the API key is correct")
        print("3. Ensure you've enabled API access in Settings → API Keys")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export OpenAPI specification to JSON file.

Usage:
    python scripts/export_openapi.py [--output docs/openapi.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def export_openapi(output_path: str = "docs/openapi.json") -> None:
    """Export the OpenAPI specification from the FastAPI app."""
    from src.api.server import app

    # Generate OpenAPI schema
    openapi_schema = app.openapi()

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_file, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    # Print summary
    paths = openapi_schema.get("paths", {})
    total_endpoints = sum(
        len([m for m in methods.keys() if m in ("get", "post", "put", "patch", "delete")])
        for methods in paths.values()
    )

    print(f"✅ OpenAPI specification exported to: {output_file}")
    print(f"   Version: {openapi_schema.get('info', {}).get('version', 'unknown')}")
    print(f"   Title: {openapi_schema.get('info', {}).get('title', 'unknown')}")
    print(f"   Paths: {len(paths)}")
    print(f"   Total Endpoints: {total_endpoints}")
    print(f"   Tags: {len(openapi_schema.get('tags', []))}")

    # List tags
    tags = openapi_schema.get("tags", [])
    if tags:
        print("\n   Available Tags:")
        for tag in tags:
            print(f"      - {tag.get('name')}: {tag.get('description', '')[:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Export OpenAPI specification")
    parser.add_argument(
        "--output", "-o",
        default="docs/openapi.json",
        help="Output file path (default: docs/openapi.json)"
    )
    args = parser.parse_args()

    try:
        export_openapi(args.output)
    except Exception as e:
        print(f"❌ Error exporting OpenAPI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

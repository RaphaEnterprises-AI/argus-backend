#!/usr/bin/env python3
"""CLI script to migrate existing data to the knowledge graph.

Usage:
    python scripts/migrate_knowledge_graph.py [--project-id UUID] [--batch-size N]

Examples:
    # Migrate all projects
    python scripts/migrate_knowledge_graph.py

    # Migrate specific project
    python scripts/migrate_knowledge_graph.py --project-id abc123...

    # Use custom batch size
    python scripts/migrate_knowledge_graph.py --batch-size 500
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph.migrations import run_migration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    """Run the migration CLI."""
    parser = argparse.ArgumentParser(
        description="Migrate existing data to Apache AGE knowledge graph"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Optional project UUID to migrate (migrates all if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to process per batch (default: 100)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL database URL (default: DATABASE_URL env var)",
    )

    args = parser.parse_args()

    if not args.database_url:
        logger.error("DATABASE_URL environment variable or --database-url required")
        sys.exit(1)

    logger.info("Starting knowledge graph migration")
    logger.info(f"Database URL: {args.database_url[:30]}...")
    if args.project_id:
        logger.info(f"Project ID: {args.project_id}")
    logger.info(f"Batch size: {args.batch_size}")

    try:
        stats = await run_migration(
            database_url=args.database_url,
            project_id=args.project_id,
        )

        logger.info("Migration completed successfully!")
        logger.info("Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

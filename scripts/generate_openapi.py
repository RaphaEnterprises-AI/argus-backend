#!/usr/bin/env python3
"""Generate OpenAPI specification without requiring all runtime dependencies.

This script mocks heavy dependencies (cognee, langchain, etc.) to allow
generating the OpenAPI spec from the FastAPI route definitions.
"""

import json
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing the app
MOCK_MODULES = [
    'cognee',
    'cognee.api',
    'cognee.api.v1',
    'cognee.api.v1.search',
    'cognee.api.v1.add',
    'cognee.api.v1.cognify',
    'cognee.api.v1.datasets',
    'cognee.modules',
    'cognee.modules.ingestion',
    'cognee.exceptions',
    'langchain_core',
    'langchain_core.messages',
    'langchain_core.tools',
    'langchain_core.runnables',
    'langchain_anthropic',
    'langgraph',
    'langgraph.graph',
    'langgraph.graph.message',
    'langgraph.graph.state',
    'langgraph.checkpoint',
    'langgraph.checkpoint.memory',
    'langgraph.checkpoint.postgres',
    'langgraph.checkpoint.postgres.aio',
    'langgraph.prebuilt',
    'langgraph.types',
    'langgraph_checkpoint',
    'langgraph_checkpoint_postgres',
    'psycopg',
    'psycopg.rows',
    'psycopg_pool',
    'anthropic',
    'anthropic.types',
    'google.genai',
    'google.generativeai',
    'playwright',
    'playwright.async_api',
    'selenium',
    'selenium.webdriver',
    'docker',
    'aiokafka',
    'sentence_transformers',
    'falkordb',
    'redis',
    'tiktoken',
    'PIL',
    'PIL.Image',
    'imagehash',
    'skimage',
    'skimage.metrics',
    'numpy',
    'boto3',
    'sentry_sdk',
    'sentry_sdk.integrations',
    'sentry_sdk.integrations.fastapi',
    'sentry_sdk.integrations.logging',
    'sentry_sdk.integrations.starlette',
    'croniter',
    'mcp',
    'langchain_mcp_adapters',
    'openai',
]

# Create mock modules
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# Specific mocks for commonly used classes
sys.modules['langgraph.graph.message'].add_messages = lambda x: x
sys.modules['langchain_core.messages'].BaseMessage = MagicMock

# Now import the app components
sys.path.insert(0, '.')

def generate_openapi():
    """Generate OpenAPI spec from the FastAPI app."""
    try:
        from src.api.server import app
        openapi_schema = app.openapi()
        return openapi_schema
    except Exception as e:
        print(f"Error generating OpenAPI: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    schema = generate_openapi()
    print(json.dumps(schema, indent=2))

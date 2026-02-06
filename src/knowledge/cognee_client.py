"""
Cognee Knowledge Client - Unified knowledge layer for Argus.

This module provides a unified interface to Cognee for:
- Storing and retrieving knowledge (key-value with embeddings)
- Semantic search across all knowledge types
- Failure pattern storage and retrieval
- Multi-hop graph reasoning

This replaces the custom MemoryStore and GraphStore with a single
Cognee-powered layer, reducing code complexity and gaining:
- ECL (Extract, Cognify, Load) pipeline
- 30+ data connectors
- Incremental learning
- Graph + vector hybrid search
- Multi-hop reasoning

IMPORTANT: Cognee is a required dependency. The MemoryStore and GraphStore
components were deprecated in favor of this unified Cognee layer. If Cognee
is not installed, the application will fail with a clear error message.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from typing import Any, Optional

import structlog

# =============================================================================
# CRITICAL: Configure embedding environment BEFORE importing Cognee!
# =============================================================================
# Cognee's EmbeddingConfig uses pydantic_settings with @lru_cache, which means
# the config is read from environment variables at first access and then cached.
# If we set env vars AFTER importing cognee, the cache already has wrong values.
# We MUST set embedding env vars here, BEFORE the cognee import below.

_early_logger = structlog.get_logger("cognee_early_config")


def _configure_cognee_early() -> None:
    """Configure ALL Cognee settings BEFORE cognee import.

    This must run before 'import cognee' because Cognee's configs use
    @lru_cache and read env vars on first access. If we set env vars
    AFTER importing cognee, the caches already have wrong values.

    Configures:
    - Embedding provider (Cohere/OpenAI)
    - Database provider (PostgreSQL/SQLite)
    - Vector DB provider (pgvector/lancedb)
    """
    # =========================================================================
    # 1. EMBEDDING CONFIGURATION
    # =========================================================================
    cohere_key = os.environ.get("COHERE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if cohere_key:
        os.environ["EMBEDDING_PROVIDER"] = "cohere"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("EMBEDDING_MODEL", "embed-multilingual-v3.0")
        os.environ["EMBEDDING_API_KEY"] = cohere_key
        os.environ["EMBEDDING_DIMENSIONS"] = os.environ.get("EMBEDDING_DIMENSIONS", "1024")
        _early_logger.info(
            "Pre-import: Embedding configured for Cohere",
            model=os.environ["EMBEDDING_MODEL"],
            dimensions=os.environ["EMBEDDING_DIMENSIONS"],
        )
    elif openai_key:
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["EMBEDDING_MODEL"] = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        os.environ["EMBEDDING_API_KEY"] = openai_key
        os.environ["EMBEDDING_DIMENSIONS"] = os.environ.get("EMBEDDING_DIMENSIONS", "1536")
        _early_logger.info(
            "Pre-import: Embedding configured for OpenAI",
            model=os.environ["EMBEDDING_MODEL"],
        )
    elif openrouter_key:
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
        os.environ["EMBEDDING_ENDPOINT"] = "https://openrouter.ai/api/v1"
        os.environ["EMBEDDING_API_KEY"] = openrouter_key
        os.environ["EMBEDDING_DIMENSIONS"] = "1536"
        _early_logger.info(
            "Pre-import: Embedding configured for OpenRouter (OpenAI-compatible)",
        )
    else:
        _early_logger.error(
            "CRITICAL: No embedding API key found! "
            "Set COHERE_API_KEY or OPENAI_API_KEY. "
            "Cognee storage will FAIL without embeddings."
        )

    # =========================================================================
    # 2. DATABASE CONFIGURATION (RelationalConfig + VectorConfig)
    # =========================================================================
    # Cognee's RelationalConfig also uses @lru_cache - must set env vars early!
    database_url = os.environ.get("DATABASE_URL")

    if database_url and "postgresql" in database_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)

            # Set env vars that Cognee's RelationalConfig reads
            os.environ["DB_PROVIDER"] = "postgres"
            os.environ["DB_HOST"] = parsed.hostname or "localhost"
            os.environ["DB_PORT"] = str(parsed.port) if parsed.port else "5432"
            os.environ["DB_NAME"] = parsed.path.lstrip("/") if parsed.path else "cognee"
            os.environ["DB_USERNAME"] = parsed.username or ""
            os.environ["DB_PASSWORD"] = parsed.password or ""

            # Use pgvector for vector storage (not lancedb)
            os.environ["VECTOR_DB_PROVIDER"] = "pgvector"

            _early_logger.info(
                "Pre-import: Database configured for PostgreSQL + pgvector",
                host=parsed.hostname,
                port=os.environ["DB_PORT"],
                db=os.environ["DB_NAME"],
            )
        except Exception as e:
            _early_logger.error(f"Pre-import: Failed to parse DATABASE_URL: {e}")
            # Fall back to in-memory/local storage
            os.environ["VECTOR_DB_PROVIDER"] = "lancedb"
            os.environ["DB_PROVIDER"] = "sqlite"
    else:
        # Local development - use embedded databases
        os.environ["VECTOR_DB_PROVIDER"] = "lancedb"
        os.environ["DB_PROVIDER"] = "sqlite"
        _early_logger.info("Pre-import: Database configured for SQLite + LanceDB (local mode)")

    # =========================================================================
    # 3. GRAPH DATABASE CONFIGURATION
    # =========================================================================
    # CRITICAL: FalkorDB is NOT supported by Cognee!
    # - Backend access control supports: ['neo4j_aura_dev', 'lancedb', 'pgvector', 'kuzu']
    # - ECL graph pipeline supports: ['neo4j', 'kuzu', 'kuzu-remote', 'neptune']
    #
    # We use Neo4j Aura (cloud) when NEO4J_URI is configured, otherwise kuzu (local).
    #
    # IMPORTANT: Remove FALKORDB_* env vars so Cognee doesn't detect FalkorDB
    # and try to use it (Cognee reads these before our GRAPH_DATABASE_PROVIDER).
    import sys
    falkordb_vars = [k for k in os.environ.keys() if "FALKOR" in k.upper()]
    print(f"[COGNEE DEBUG] Found FALKOR env vars: {falkordb_vars}", file=sys.stderr)
    for key in list(os.environ.keys()):
        if "FALKOR" in key.upper():
            del os.environ[key]
            print(f"[COGNEE DEBUG] Removed {key}", file=sys.stderr)

    # Check graph database provider - FalkorDB is the primary choice
    explicit_provider = os.environ.get("GRAPH_DATABASE_PROVIDER", "").lower()
    falkordb_host = os.environ.get("FALKORDB_HOST")
    falkordb_port = os.environ.get("FALKORDB_PORT", "6379")
    falkordb_password = os.environ.get("FALKORDB_PASSWORD", "")

    if explicit_provider == "falkor" or falkordb_host:
        # FalkorDB configured - use it for both graph and vector
        os.environ["GRAPH_DATABASE_PROVIDER"] = "falkor"
        os.environ["GRAPH_DATABASE_URL"] = falkordb_host or "localhost"
        os.environ["GRAPH_DATABASE_PORT"] = falkordb_port
        if falkordb_password:
            os.environ["GRAPH_DATABASE_PASSWORD"] = falkordb_password
        # Also use FalkorDB for vector storage (hybrid mode)
        os.environ["VECTOR_DB_PROVIDER"] = "falkor"
        os.environ["VECTOR_DB_URL"] = f"redis://:{falkordb_password}@{falkordb_host or 'localhost'}:{falkordb_port}"
        _early_logger.info(
            "Pre-import: Graph + Vector DB configured for FalkorDB",
            host=falkordb_host or "localhost",
            port=falkordb_port,
        )
    elif explicit_provider == "kuzu":
        # Kuzu explicitly requested - use embedded graph DB
        os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
        _early_logger.info("Pre-import: Graph DB configured for Kuzu (embedded)")
    else:
        # Default to FalkorDB if available, else kuzu
        os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
        _early_logger.info("Pre-import: Graph DB configured for Kuzu (default)")


# Execute early configuration BEFORE importing cognee
_configure_cognee_early()

# Cognee is a REQUIRED dependency - fail fast if not installed
try:
    import cognee

    # Register FalkorDB adapter if using FalkorDB
    if os.environ.get("GRAPH_DATABASE_PROVIDER", "").lower() == "falkor":
        try:
            from cognee_community_hybrid_adapter_falkor import register
            register()
            _early_logger.info("FalkorDB adapter registered successfully")
        except ImportError:
            _early_logger.warning(
                "cognee-community-hybrid-adapter-falkor not installed. "
                "Install with: pip install cognee-community-hybrid-adapter-falkor"
            )
    # Clear ALL config caches in case they were already loaded
    # This ensures our env vars are picked up
    try:
        from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
        get_embedding_config.cache_clear()
        _early_logger.info("Cleared Cognee embedding config cache")
    except Exception:
        pass  # Cache clear is best-effort

    try:
        from cognee.infrastructure.databases.relational.config import get_relational_config
        get_relational_config.cache_clear()
        _early_logger.info("Cleared Cognee relational config cache")
    except Exception:
        pass  # Cache clear is best-effort

    try:
        from cognee.infrastructure.databases.vector.config import get_vectordb_config
        get_vectordb_config.cache_clear()
        _early_logger.info("Cleared Cognee vector DB config cache")
    except Exception:
        pass  # Cache clear is best-effort

    # CRITICAL: Clear graph database config cache to prevent FalkorDB detection
    try:
        from cognee.infrastructure.databases.graph.config import get_graph_config
        get_graph_config.cache_clear()
        _early_logger.info("Cleared Cognee graph DB config cache")
    except Exception as e:
        _early_logger.warning(f"Could not clear graph config cache: {e}")

    # Also try the alternative graph config location
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
        if hasattr(get_graph_engine, 'cache_clear'):
            get_graph_engine.cache_clear()
            _early_logger.info("Cleared Cognee graph engine cache")
    except Exception:
        pass  # Cache clear is best-effort

except ImportError as e:
    raise ImportError(
        "The cognee package is required for Argus. "
        "This is a mandatory dependency since MemoryStore and GraphStore were deprecated. "
        "Install it with: pip install cognee"
    ) from e

logger = structlog.get_logger(__name__)


# =============================================================================
# Selector Classification (RAP-355)
# =============================================================================


def classify_selector_type(selector: str) -> str:
    """Classify a selector into an anonymous type for SOC2-compliant storage.

    This function anonymizes selectors by categorizing them into types rather than
    storing actual selector values. This is crucial for:
    - SOC2 compliance (no customer-specific data in patterns)
    - Cross-project learning (patterns transfer between projects)
    - Privacy (no PII or sensitive data in the learning loop)

    Args:
        selector: The CSS selector, XPath, or Playwright locator string

    Returns:
        Anonymized selector type string. One of:
        - "data_testid": data-testid attribute selectors
        - "data_cy": Cypress-style data-cy attributes
        - "data_test": Generic data-test attributes
        - "id_attribute": ID-based selectors (#id or [id="..."])
        - "aria_label": aria-label attribute selectors
        - "aria_labelledby": aria-labelledby attribute selectors
        - "role": role attribute selectors
        - "class_single": Single class selectors (.class)
        - "class_multiple": Multiple class selectors (.class1.class2)
        - "xpath_id": XPath with ID
        - "xpath_text": XPath with text()
        - "xpath_attribute": XPath with other attributes
        - "xpath_positional": XPath with positional indices [n]
        - "css_attribute": CSS attribute selectors [attr="value"]
        - "css_pseudo": CSS pseudo-selectors (:nth-child, etc.)
        - "text_exact": Exact text match (Playwright text=)
        - "text_partial": Partial text match (Playwright *text=)
        - "name_attribute": name attribute selectors
        - "placeholder": placeholder attribute selectors
        - "tag_only": Tag-only selectors (button, input, etc.)
        - "unknown": Could not classify

    Examples:
        >>> classify_selector_type('[data-testid="login-button"]')
        'data_testid'
        >>> classify_selector_type('#submit-btn')
        'id_attribute'
        >>> classify_selector_type('.btn.btn-primary')
        'class_multiple'
        >>> classify_selector_type('//button[text()="Submit"]')
        'xpath_text'
    """
    if not selector or not isinstance(selector, str):
        return "unknown"

    selector = selector.strip()

    # -------------------------------------------------------------------------
    # 1. XPath selectors (check first as they're distinctive)
    # -------------------------------------------------------------------------
    if selector.startswith("//") or selector.startswith("(//"):
        if "@id" in selector or "[@id=" in selector:
            return "xpath_id"
        if "text()" in selector or "contains(text()" in selector:
            return "xpath_text"
        if re.search(r"\[\d+\]", selector):
            return "xpath_positional"
        if "@" in selector:
            return "xpath_attribute"
        return "xpath_positional"  # Generic XPath

    # -------------------------------------------------------------------------
    # 2. Playwright-specific locators
    # -------------------------------------------------------------------------
    # Playwright text locators
    if selector.startswith("text=") or selector.startswith('"') or selector.startswith("'"):
        return "text_exact"
    if selector.startswith("*text=") or selector.startswith("has-text="):
        return "text_partial"

    # Playwright getBy* style (role, label, etc.)
    if "role=" in selector.lower() or selector.startswith("role="):
        return "role"

    # -------------------------------------------------------------------------
    # 3. Data attributes (most stable for testing)
    # -------------------------------------------------------------------------
    data_testid_patterns = [
        r"\[data-testid[=~\|\^\$\*]",  # CSS attribute selector
        r'data-testid\s*=\s*["\']',  # HTML attribute with quotes
        r'^data-testid\s*=',  # Bare data-testid= at start (e.g., "data-testid=submit")
        r"getByTestId",  # Playwright/RTL
        r"findByTestId",  # RTL
        r"queryByTestId",  # RTL
    ]
    for pattern in data_testid_patterns:
        if re.search(pattern, selector, re.IGNORECASE):
            return "data_testid"

    # Cypress data-cy
    if re.search(r"\[data-cy[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "data_cy"

    # Generic data-test
    if re.search(r"\[data-test[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "data_test"

    # -------------------------------------------------------------------------
    # 4. ARIA attributes (accessible selectors)
    # -------------------------------------------------------------------------
    if re.search(r"\[aria-label[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "aria_label"
    if re.search(r"\[aria-labelledby[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "aria_labelledby"
    if re.search(r"\[role[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "role"
    if "getByRole" in selector or "findByRole" in selector:
        return "role"
    if "getByLabelText" in selector or "findByLabelText" in selector:
        return "aria_label"

    # -------------------------------------------------------------------------
    # 5. ID selectors
    # -------------------------------------------------------------------------
    # CSS ID selector: #id-name
    if re.match(r"^#[a-zA-Z_][a-zA-Z0-9_-]*$", selector):
        return "id_attribute"
    # Attribute selector: [id="..."]
    if re.search(r"\[id[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "id_attribute"
    # getElementById
    if "getElementById" in selector:
        return "id_attribute"

    # -------------------------------------------------------------------------
    # 6. Class selectors
    # -------------------------------------------------------------------------
    # Single class: .class-name
    if re.match(r"^\.[a-zA-Z_][a-zA-Z0-9_-]*$", selector):
        return "class_single"
    # Multiple classes: .class1.class2 or .class1 .class2
    if re.match(r"^\.[a-zA-Z_][a-zA-Z0-9_-]*(\s*\.[a-zA-Z_][a-zA-Z0-9_-]*)+$", selector):
        return "class_multiple"
    # Class attribute: [class="..."] or [class~="..."]
    if re.search(r"\[class[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "class_single"

    # -------------------------------------------------------------------------
    # 7. Name and placeholder attributes
    # -------------------------------------------------------------------------
    if re.search(r"\[name[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "name_attribute"
    if re.search(r"\[placeholder[=~\|\^\$\*]", selector, re.IGNORECASE):
        return "placeholder"
    if "getByPlaceholderText" in selector or "findByPlaceholderText" in selector:
        return "placeholder"

    # -------------------------------------------------------------------------
    # 8. CSS pseudo-selectors
    # -------------------------------------------------------------------------
    if re.search(r":(nth-child|nth-of-type|first-child|last-child|first-of-type|last-of-type)", selector):
        return "css_pseudo"
    if ":has(" in selector or ":is(" in selector or ":where(" in selector:
        return "css_pseudo"

    # -------------------------------------------------------------------------
    # 9. Generic CSS attribute selectors
    # -------------------------------------------------------------------------
    if re.search(r"\[[a-zA-Z_][a-zA-Z0-9_-]*[=~\|\^\$\*]", selector):
        return "css_attribute"

    # -------------------------------------------------------------------------
    # 10. Tag-only selectors
    # -------------------------------------------------------------------------
    html_tags = {
        "button", "input", "a", "div", "span", "form", "select", "textarea",
        "label", "img", "table", "tr", "td", "th", "ul", "ol", "li", "p",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "nav", "main",
        "section", "article", "aside", "dialog", "menu", "summary", "details",
    }
    if selector.lower() in html_tags:
        return "tag_only"

    # -------------------------------------------------------------------------
    # 11. Fallback
    # -------------------------------------------------------------------------
    return "unknown"


# =============================================================================
# Cognee Configuration (runs once at module load)
# =============================================================================

_cognee_configured = False


def _configure_cognee() -> None:
    """Configure Cognee with LLM settings and verify all configs.

    NOTE: Database, embedding, and vector DB configs are now set early
    (before import) in _configure_cognee_early(). This function only
    handles LLM config via Cognee API and verifies all settings.

    Called lazily on first client instantiation.
    """
    global _cognee_configured
    if _cognee_configured:
        return

    # LLM Configuration - prefer Anthropic, fall back to OpenRouter
    # This must use Cognee's API since LLM config isn't just env vars
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    # Configure Graph Database FIRST (before LLM) via Cognee API
    # This is more reliable than just setting env vars
    neo4j_uri = os.environ.get("NEO4J_URI") or os.environ.get("GRAPH_DATABASE_URL")
    if neo4j_uri and "neo4j" in neo4j_uri:
        try:
            cognee.config.set_graph_database_provider("neo4j")
            cognee.config.set_graph_database_config({
                "graph_database_url": neo4j_uri,
                "graph_database_username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                "graph_database_password": os.environ.get("NEO4J_PASSWORD", ""),
            })
            logger.info("Cognee graph DB configured for Neo4j Aura via API", uri=neo4j_uri[:40])
        except Exception as e:
            logger.warning(f"Could not set graph DB via API: {e}, falling back to env vars")
    else:
        try:
            cognee.config.set_graph_database_provider("kuzu")
            logger.info("Cognee graph DB configured for Kuzu via API")
        except Exception as e:
            logger.warning(f"Could not set graph DB via API: {e}")

    if anthropic_key:
        cognee.config.set_llm_provider("anthropic")
        cognee.config.set_llm_api_key(anthropic_key)
        cognee.config.set_llm_model(os.environ.get("LLM_MODEL", "claude-sonnet-4-5"))
        logger.info("Cognee configured with Anthropic LLM")
    elif openrouter_key:
        cognee.config.set_llm_provider("openrouter")
        cognee.config.set_llm_config({
            "llm_api_key": openrouter_key,
            "llm_model": os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4"),
            "llm_endpoint": "https://openrouter.ai/api/v1",
        })
        logger.info("Cognee configured with OpenRouter LLM")
    else:
        logger.warning(
            "No LLM API key found for Cognee. "
            "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY."
        )

    # Verify embedding config (set early in _configure_cognee_early)
    try:
        from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
        embed_config = get_embedding_config()
        logger.info(
            "Embedding configuration verified",
            provider=embed_config.embedding_provider,
            model=embed_config.embedding_model,
            dimensions=embed_config.embedding_dimensions,
            has_api_key=bool(embed_config.embedding_api_key),
        )
        if not embed_config.embedding_api_key:
            logger.error(
                "CRITICAL: Embedding API key not configured! "
                "Set COHERE_API_KEY or OPENAI_API_KEY environment variable. "
                "Cognee storage will FAIL without embeddings."
            )
    except Exception as e:
        logger.warning(f"Could not verify embedding config: {e}")

    # Verify relational DB config (set early in _configure_cognee_early)
    try:
        from cognee.infrastructure.databases.relational.config import get_relational_config
        rel_config = get_relational_config()
        logger.info(
            "Relational DB configuration verified",
            provider=rel_config.db_provider,
            host=rel_config.db_host,
            port=rel_config.db_port,
            db=rel_config.db_name,
        )
    except Exception as e:
        logger.warning(f"Could not verify relational config: {e}")

    # Verify vector DB config (set early in _configure_cognee_early)
    try:
        from cognee.infrastructure.databases.vector.config import get_vectordb_config
        vec_config = get_vectordb_config()
        logger.info(
            "Vector DB configuration verified",
            provider=vec_config.vector_db_provider,
        )
    except Exception as e:
        logger.warning(f"Could not verify vector DB config: {e}")

    _cognee_configured = True
    logger.info("Cognee configuration complete")


# =============================================================================
# Custom Exceptions for Cognee Operations
# =============================================================================


class CogneeError(Exception):
    """Base exception for all Cognee-related errors."""
    pass


class CogneeStorageError(CogneeError):
    """Raised when storing data to Cognee fails."""
    pass


class CogneeRetrievalError(CogneeError):
    """Raised when retrieving data from Cognee fails."""
    pass


class CogneeSearchError(CogneeError):
    """Raised when searching Cognee fails."""
    pass


class CogneeGraphError(CogneeError):
    """Raised when graph operations fail."""
    pass


@dataclass
class SimilarFailure:
    """A similar failure pattern found via semantic search."""

    id: str
    error_message: str
    error_type: str | None
    original_selector: str | None
    healed_selector: str
    healing_method: str
    success_count: int
    failure_count: int
    success_rate: float
    similarity: float
    metadata: dict


class CogneeKnowledgeClient:
    """
    Unified knowledge client powered by Cognee.

    This client provides a compatibility layer that matches the interface
    of the deprecated MemoryStore and GraphStore, while using Cognee's
    superior ECL pipeline and graph capabilities under the hood.

    Key features:
    - Drop-in replacement for MemoryStore.get/put/search
    - Drop-in replacement for MemoryStore failure pattern methods
    - Graph-based reasoning via Cognee's knowledge graph
    - Multi-tenant isolation via dataset namespacing

    Example:
        ```python
        client = CogneeKnowledgeClient(org_id="org123", project_id="proj456")

        # Store a value
        await client.put(
            namespace=["patterns", "selectors"],
            key="login-button",
            value={"selector": "#login-btn", "success_rate": 0.95},
            embed_text="login button selector pattern"
        )

        # Semantic search
        results = await client.search(
            namespace=["patterns"],
            query="button for user login",
            limit=5
        )

        # Find similar failures
        similar = await client.find_similar_failures(
            error_message="Element not found: #submit-btn"
        )
        ```
    """

    def __init__(
        self,
        org_id: str,
        project_id: str,
    ):
        """
        Initialize the Cognee knowledge client.

        Args:
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for multi-tenant isolation
        """
        self.org_id = org_id
        self.project_id = project_id
        self._namespace_prefix = f"{org_id}:{project_id}"
        self._log = logger.bind(
            component="cognee_client",
            org_id=org_id,
            project_id=project_id,
        )

    # Note: _ensure_cognee_available() was removed because Cognee is now
    # a required dependency. The import will fail at module load if not installed.

    def _build_namespace(self, parts: list[str] | tuple[str, ...]) -> str:
        """Build a full namespace string from parts.

        Args:
            parts: Namespace parts like ["patterns", "selectors"]

        Returns:
            Full namespace like "org_id:project_id:patterns:selectors"
        """
        if parts:
            return f"{self._namespace_prefix}:{':'.join(parts)}"
        return self._namespace_prefix

    def _get_dataset_name(self, namespace: list[str] | tuple[str, ...]) -> str:
        """Generate tenant-scoped dataset name from namespace.

        Args:
            namespace: Hierarchical namespace like ["patterns", "selectors"]

        Returns:
            Dataset name like "org_abc_project_xyz_patterns_selectors"
        """
        namespace_str = "_".join(namespace) if namespace else "default"
        return f"org_{self.org_id}_project_{self.project_id}_{namespace_str}"

    def _generate_key_id(self, namespace: list[str] | tuple[str, ...], key: str) -> str:
        """Generate a unique ID for a namespace+key combination.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            Unique hash ID
        """
        full_key = f"{self.org_id}:{self.project_id}:{':'.join(namespace)}:{key}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:32]

    # =========================================================================
    # Key-Value Store Interface (replaces MemoryStore.get/put)
    # =========================================================================

    async def put(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
        value: dict[str, Any],
        embed_text: str | None = None,
    ) -> None:
        """Store a value with optional embedding.

        This provides compatibility with the MemoryStore.put() interface
        while using Cognee's ECL pipeline for storage.

        Args:
            namespace: Hierarchical namespace (e.g., ["patterns", "selectors"])
            key: Unique key within namespace
            value: JSON-serializable value
            embed_text: Optional text for embedding generation

        Raises:
            CogneeStorageError: If storing the value fails
        """
        dataset_name = self._get_dataset_name(namespace)
        key_id = self._generate_key_id(namespace, key)

        # Build content document with metadata
        content = {
            "_id": key_id,
            "_key": key,
            "_namespace": list(namespace),
            "_org_id": self.org_id,
            "_project_id": self.project_id,
            "_created_at": datetime.now(UTC).isoformat(),
            "_embed_text": embed_text,
            **value,
        }

        try:
            # Add to Cognee
            await cognee.add(
                json.dumps(content),
                dataset_name=dataset_name,
            )

            # Run cognify to extract knowledge and create embeddings
            await cognee.cognify(dataset_name=dataset_name)

            self._log.debug(
                "Stored value in Cognee",
                namespace=namespace,
                key=key,
                dataset=dataset_name,
            )
        except Exception as e:
            self._log.error(
                "Failed to store value in Cognee",
                namespace=namespace,
                key=key,
                dataset=dataset_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise CogneeStorageError(
                f"Failed to store value in namespace {namespace}, key '{key}': {e}"
            ) from e

    async def get(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
    ) -> dict[str, Any] | None:
        """Get a value by namespace and key.

        This provides compatibility with the MemoryStore.get() interface.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            Stored value or None if not found

        Raises:
            CogneeRetrievalError: If retrieval fails due to an error (not just missing key)
        """
        dataset_name = self._get_dataset_name(namespace)
        key_id = self._generate_key_id(namespace, key)

        try:
            # Search for the specific key
            results = await cognee.search(
                query_text=f"_id:{key_id}",
                datasets=[dataset_name],
                top_k=1,
            )

            if results and len(results) > 0:
                result = results[0]
                # Parse the result and extract the value
                if isinstance(result, str):
                    return json.loads(result)
                elif isinstance(result, dict):
                    # Remove internal fields
                    return {k: v for k, v in result.items() if not k.startswith("_")}

            # Not found - this is a valid case, return None
            return None

        except json.JSONDecodeError as e:
            self._log.error(
                "Failed to parse Cognee result as JSON",
                namespace=namespace,
                key=key,
                error=str(e),
            )
            raise CogneeRetrievalError(
                f"Failed to parse result for namespace {namespace}, key '{key}': {e}"
            ) from e
        except Exception as e:
            self._log.error(
                "Failed to get value from Cognee",
                namespace=namespace,
                key=key,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise CogneeRetrievalError(
                f"Failed to get value from namespace {namespace}, key '{key}': {e}"
            ) from e

    async def delete(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
    ) -> bool:
        """Delete a value by namespace and key.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            True if deleted, False otherwise
        """
        # Note: Cognee doesn't have a direct delete API in v0.3
        # We mark the item as deleted instead
        await self.put(
            namespace=namespace,
            key=key,
            value={"_deleted": True, "_deleted_at": datetime.now(UTC).isoformat()},
        )
        return True

    async def search(
        self,
        namespace: list[str] | tuple[str, ...],
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Semantic search within a namespace.

        This provides compatibility with the MemoryStore.search() interface
        while leveraging Cognee's hybrid graph+vector search.

        Args:
            namespace: Namespace to search in
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity (0-1)

        Returns:
            List of matching items with similarity scores

        Raises:
            CogneeSearchError: If the search operation fails
        """
        dataset_name = self._get_dataset_name(namespace)

        try:
            results = await cognee.search(
                query_text=query,
                datasets=[dataset_name],
                top_k=limit,
            )

            parsed_results = []
            for i, result in enumerate(results or []):
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        parsed = {"content": result}
                else:
                    parsed = result

                # Add similarity score (Cognee results are ordered by relevance)
                parsed["similarity"] = 1.0 - (i * 0.1)  # Approximate score
                parsed_results.append(parsed)

            return parsed_results

        except Exception as e:
            self._log.error(
                "Search failed",
                namespace=namespace,
                query=query[:50] if query else "",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise CogneeSearchError(
                f"Search failed in namespace {namespace} for query '{query[:50]}...': {e}"
            ) from e

    # =========================================================================
    # Failure Pattern Interface (replaces MemoryStore failure methods)
    # =========================================================================

    async def store_failure_pattern(
        self,
        error_message: str,
        healed_selector: str,
        healing_method: str,
        test_id: str | None = None,
        error_type: str | None = None,
        original_selector: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a failure pattern for future learning.

        This stores a successful healing pattern in Cognee's knowledge graph,
        allowing semantic search to find similar failures and their solutions.

        Args:
            error_message: The error message from the failure
            healed_selector: The selector that fixed the issue
            healing_method: Method used (e.g., "semantic_match", "code_aware")
            test_id: Optional test ID
            error_type: Type of error
            original_selector: The original broken selector
            metadata: Additional metadata

        Returns:
            Pattern ID

        Raises:
            CogneeStorageError: If storing the pattern fails
        """
        # Ensure no None values - Cognee's internal processing fails on None
        # Convert None to empty strings for string fields
        safe_error_message = error_message or ""
        safe_error_type = error_type or "unknown"
        safe_original_selector = original_selector or ""
        safe_healed_selector = healed_selector or ""
        safe_healing_method = healing_method or "unknown"
        safe_test_id = test_id or ""

        pattern_id = hashlib.sha256(
            f"{safe_error_message}:{safe_original_selector}:{safe_healed_selector}".encode()
        ).hexdigest()[:32]

        pattern = {
            "pattern_id": pattern_id,
            "error_message": safe_error_message,
            "error_type": safe_error_type,
            "original_selector": safe_original_selector,
            "healed_selector": safe_healed_selector,
            "healing_method": safe_healing_method,
            "test_id": safe_test_id,
            "success_count": 1,
            "failure_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }

        await self.put(
            namespace=["failure_patterns"],
            key=pattern_id,
            value=pattern,
            embed_text=f"{safe_error_message} {safe_error_type} {safe_original_selector}",
        )

        self._log.info(
            "Stored failure pattern",
            pattern_id=pattern_id,
            error_type=error_type,
            healing_method=healing_method,
        )

        return pattern_id

    async def find_similar_failures(
        self,
        error_message: str,
        limit: int = 5,
        threshold: float = 0.7,
        error_type: str | None = None,
    ) -> list[SimilarFailure]:
        """Find similar past failures with their healing solutions.

        Uses Cognee's semantic search to find failures with similar
        error messages and returns the healing solutions that worked.

        Args:
            error_message: Current error message
            limit: Maximum results
            threshold: Minimum similarity
            error_type: Optional filter

        Returns:
            List of similar failures with solutions
        """
        query = error_message
        if error_type:
            query = f"{error_type}: {error_message}"

        results = await self.search(
            namespace=["failure_patterns"],
            query=query,
            limit=limit,
            threshold=threshold,
        )

        similar_failures = []
        for result in results:
            if result.get("_deleted"):
                continue

            success_count = result.get("success_count", 0)
            failure_count = result.get("failure_count", 0)
            total = success_count + failure_count

            similar_failures.append(SimilarFailure(
                id=result.get("pattern_id", ""),
                error_message=result.get("error_message", ""),
                error_type=result.get("error_type"),
                original_selector=result.get("original_selector"),
                healed_selector=result.get("healed_selector", ""),
                healing_method=result.get("healing_method", ""),
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_count / total if total > 0 else 0,
                similarity=result.get("similarity", 0),
                metadata=result.get("metadata", {}),
            ))

        self._log.debug(
            "Found similar failures",
            query=error_message[:50],
            count=len(similar_failures),
        )

        return similar_failures

    async def record_healing_outcome(
        self,
        pattern_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of applying a healing pattern.

        Updates success/failure counts for continuous learning.

        Args:
            pattern_id: Pattern ID
            success: Whether healing worked
        """
        existing = await self.get(
            namespace=["failure_patterns"],
            key=pattern_id,
        )

        if not existing:
            self._log.warning("Pattern not found", pattern_id=pattern_id)
            return

        if success:
            existing["success_count"] = existing.get("success_count", 0) + 1
        else:
            existing["failure_count"] = existing.get("failure_count", 0) + 1

        existing["last_used_at"] = datetime.now(UTC).isoformat()

        await self.put(
            namespace=["failure_patterns"],
            key=pattern_id,
            value=existing,
        )

        self._log.debug(
            "Recorded healing outcome",
            pattern_id=pattern_id,
            success=success,
        )

    # =========================================================================
    # Cognee Learning Loop Methods (RAP-353, RAP-354)
    # =========================================================================

    async def store_successful_heal(
        self,
        org_id: str,
        test_id: str,
        failure_context: dict,
        fix_applied: dict,
    ) -> str:
        """Store a successful healing pattern with anonymized data for cross-project learning.

        This is the primary entry point for the Cognee Learning Loop. When a test is
        successfully healed, this method stores the ANONYMIZED pattern for future use.

        CRITICAL FOR SOC2 COMPLIANCE:
        - Stores selector_type NOT actual selectors
        - Stores error_category NOT specific error messages
        - No customer-specific data in the patterns

        Args:
            org_id: Organization ID for namespacing
            test_id: Test ID that was healed
            failure_context: Context about the failure, should include:
                - error_type: Type of error (selector_not_found, timeout, etc.)
                - error_message: Original error message (will be anonymized)
                - failed_selector: The selector that failed (will be classified)
                - page_url: URL where failure occurred (domain extracted only)
                - dom_context: Optional DOM context around failure
            fix_applied: Details about the fix, should include:
                - new_selector: The selector that worked (will be classified)
                - fix_type: Type of fix (selector_update, wait_added, etc.)
                - confidence: Confidence score of the fix
                - healing_method: Method used (cached, memory_store, code_aware, llm)

        Returns:
            Pattern ID for the stored pattern

        Raises:
            CogneeStorageError: If storing the pattern fails

        Example:
            ```python
            pattern_id = await client.store_successful_heal(
                org_id="org_123",
                test_id="test_456",
                failure_context={
                    "error_type": "selector_not_found",
                    "error_message": "Element not found: #login-btn",
                    "failed_selector": "#login-btn",
                    "page_url": "https://app.example.com/login",
                },
                fix_applied={
                    "new_selector": "[data-testid='login-button']",
                    "fix_type": "selector_update",
                    "confidence": 0.95,
                    "healing_method": "code_aware",
                },
            )
            ```
        """
        # Extract and anonymize failure context
        error_type = failure_context.get("error_type", "unknown")
        failed_selector = failure_context.get("failed_selector", "")
        page_url = failure_context.get("page_url", "")

        # Classify selectors (anonymization)
        original_selector_type = classify_selector_type(failed_selector)

        # Extract domain from URL (anonymization)
        page_domain = ""
        if page_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(page_url)
                # Only store TLD pattern, not full domain
                domain_parts = (parsed.netloc or "").split(".")
                if len(domain_parts) >= 2:
                    page_domain = f"*.{domain_parts[-1]}"  # e.g., "*.com"
                else:
                    page_domain = "unknown"
            except Exception:
                page_domain = "unknown"

        # Anonymize error message - extract error category, not specifics
        error_message = failure_context.get("error_message", "")
        error_category = self._categorize_error_message(error_message)

        # Extract and anonymize fix details
        new_selector = fix_applied.get("new_selector", "")
        new_selector_type = classify_selector_type(new_selector)
        fix_type = fix_applied.get("fix_type", "unknown")
        confidence = fix_applied.get("confidence", 0.0)
        healing_method = fix_applied.get("healing_method", "unknown")

        # Generate pattern ID based on anonymized components
        pattern_key = f"{error_type}:{original_selector_type}:{new_selector_type}:{fix_type}"
        pattern_id = hashlib.sha256(pattern_key.encode()).hexdigest()[:32]

        # Build anonymized pattern for storage
        anonymized_pattern = {
            "pattern_id": pattern_id,
            # Anonymized failure context
            "error_type": error_type,
            "error_category": error_category,
            "original_selector_type": original_selector_type,
            "page_domain_pattern": page_domain,
            # Anonymized fix details
            "new_selector_type": new_selector_type,
            "fix_type": fix_type,
            "confidence": confidence,
            "healing_method": healing_method,
            # Transition pattern (key for learning)
            "selector_transition": f"{original_selector_type} -> {new_selector_type}",
            # Statistics
            "success_count": 1,
            "failure_count": 0,
            "total_applications": 1,
            # Metadata
            "first_seen_at": datetime.now(UTC).isoformat(),
            "last_success_at": datetime.now(UTC).isoformat(),
            "org_id": org_id,  # For audit trail only
        }

        # Build embedding text for semantic search
        embed_text = (
            f"Error type: {error_type}. "
            f"Error category: {error_category}. "
            f"Selector changed from {original_selector_type} to {new_selector_type}. "
            f"Fix type: {fix_type}. "
            f"Healing method: {healing_method}."
        )

        # Store in Cognee with embeddings
        await self.put(
            namespace=["healing_patterns", "successful"],
            key=pattern_id,
            value=anonymized_pattern,
            embed_text=embed_text,
        )

        self._log.info(
            "Stored successful healing pattern (anonymized)",
            pattern_id=pattern_id,
            error_type=error_type,
            selector_transition=f"{original_selector_type} -> {new_selector_type}",
            healing_method=healing_method,
            test_id=test_id,
        )

        return pattern_id

    async def search_similar_fixes(
        self,
        org_id: str,
        error_type: str,
        selector_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar past fixes using semantic search on anonymized patterns.

        This is the query side of the Cognee Learning Loop. When a test fails,
        this method finds similar past failures and their successful fixes.

        Args:
            org_id: Organization ID (for logging/audit only, search is cross-org)
            error_type: Type of error (selector_not_found, timeout, assertion_failed, etc.)
            selector_type: Optional selector type to filter by (from classify_selector_type)
            top_k: Maximum number of results to return

        Returns:
            List of similar fix patterns, each containing:
                - pattern_id: Unique pattern identifier
                - error_type: Type of error that was fixed
                - error_category: Category of the error
                - original_selector_type: Type of selector that failed
                - new_selector_type: Type of selector that worked
                - selector_transition: "original -> new" transition pattern
                - fix_type: Type of fix applied
                - healing_method: Method used for healing
                - confidence: Original fix confidence
                - success_rate: Rate of successful applications
                - total_applications: Number of times this pattern was applied
                - similarity: Semantic similarity score (0-1)

        Example:
            ```python
            fixes = await client.search_similar_fixes(
                org_id="org_123",
                error_type="selector_not_found",
                selector_type="id_attribute",
                top_k=5,
            )
            for fix in fixes:
                print(f"Pattern: {fix['selector_transition']}")
                print(f"Success rate: {fix['success_rate']:.0%}")
                print(f"Suggested fix type: {fix['fix_type']}")
            ```
        """
        # Build search query
        query_parts = [f"Error type: {error_type}"]

        if selector_type:
            query_parts.append(f"Original selector type: {selector_type}")

        query = ". ".join(query_parts)

        self._log.debug(
            "Searching for similar fixes",
            org_id=org_id,
            error_type=error_type,
            selector_type=selector_type,
            query=query,
        )

        try:
            # Search in the healing patterns namespace
            results = await self.search(
                namespace=["healing_patterns", "successful"],
                query=query,
                limit=top_k * 2,  # Get more to filter
                threshold=0.5,  # Lower threshold for broader search
            )

            # Process and filter results
            fixes = []
            for result in results:
                if not isinstance(result, dict):
                    continue

                # Skip deleted patterns
                if result.get("_deleted"):
                    continue

                # Filter by selector type if specified
                if selector_type and result.get("original_selector_type") != selector_type:
                    # Still include if it's a good match on error type
                    if result.get("error_type") != error_type:
                        continue

                # Calculate success rate
                success_count = result.get("success_count", 0)
                failure_count = result.get("failure_count", 0)
                total = success_count + failure_count
                success_rate = success_count / total if total > 0 else 0.0

                # Only include patterns with reasonable success rate
                if total > 0 and success_rate < 0.3:
                    continue

                fixes.append({
                    "pattern_id": result.get("pattern_id", ""),
                    "error_type": result.get("error_type", ""),
                    "error_category": result.get("error_category", ""),
                    "original_selector_type": result.get("original_selector_type", ""),
                    "new_selector_type": result.get("new_selector_type", ""),
                    "selector_transition": result.get("selector_transition", ""),
                    "fix_type": result.get("fix_type", ""),
                    "healing_method": result.get("healing_method", ""),
                    "confidence": result.get("confidence", 0.0),
                    "success_rate": success_rate,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "total_applications": result.get("total_applications", total),
                    "similarity": result.get("similarity", 0.0),
                    "last_success_at": result.get("last_success_at"),
                })

            # Sort by success rate * similarity (weighted relevance)
            fixes.sort(
                key=lambda x: x["success_rate"] * x["similarity"],
                reverse=True,
            )

            # Return top_k results
            fixes = fixes[:top_k]

            self._log.info(
                "Found similar fixes",
                org_id=org_id,
                error_type=error_type,
                selector_type=selector_type,
                results_count=len(fixes),
            )

            return fixes

        except CogneeSearchError:
            # Re-raise search errors
            raise
        except Exception as e:
            self._log.error(
                "Failed to search similar fixes",
                error_type=error_type,
                selector_type=selector_type,
                error=str(e),
            )
            raise CogneeSearchError(
                f"Failed to search similar fixes for error_type={error_type}: {e}"
            ) from e

    def _categorize_error_message(self, error_message: str) -> str:
        """Categorize an error message into an anonymous category.

        This anonymizes error messages by extracting the general category
        rather than storing specific details that might contain sensitive data.

        Args:
            error_message: The original error message

        Returns:
            Anonymized error category string
        """
        if not error_message:
            return "unknown"

        error_lower = error_message.lower()

        # Selector/element errors
        if any(term in error_lower for term in ["not found", "no element", "cannot find", "unable to locate"]):
            return "element_not_found"
        if any(term in error_lower for term in ["multiple elements", "more than one", "ambiguous"]):
            return "multiple_elements"
        if "stale" in error_lower:
            return "stale_element"
        if "detached" in error_lower:
            return "detached_element"

        # Timeout errors
        if any(term in error_lower for term in ["timeout", "timed out", "exceeded"]):
            if "navigation" in error_lower:
                return "navigation_timeout"
            if "wait" in error_lower:
                return "wait_timeout"
            return "general_timeout"

        # Visibility/interaction errors
        if any(term in error_lower for term in ["not visible", "hidden", "display: none"]):
            return "element_not_visible"
        if any(term in error_lower for term in ["not clickable", "not interactable", "intercepted"]):
            return "element_not_interactable"
        if "disabled" in error_lower:
            return "element_disabled"

        # Navigation errors
        if any(term in error_lower for term in ["navigation", "page load", "net::"]):
            return "navigation_error"
        if "404" in error_lower or "not found" in error_lower:
            return "page_not_found"

        # Assertion errors
        if any(term in error_lower for term in ["assert", "expect", "should", "to be", "to equal"]):
            if "text" in error_lower or "content" in error_lower:
                return "text_assertion_failed"
            if "visible" in error_lower:
                return "visibility_assertion_failed"
            if "count" in error_lower or "length" in error_lower:
                return "count_assertion_failed"
            return "general_assertion_failed"

        # Network errors
        if any(term in error_lower for term in ["network", "connection", "refused", "unreachable"]):
            return "network_error"

        # Frame/context errors
        if any(term in error_lower for term in ["frame", "iframe", "context"]):
            return "frame_context_error"

        return "other"

    # =========================================================================
    # Graph Reasoning Interface (replaces GraphStore)
    # =========================================================================

    async def add_to_knowledge_graph(
        self,
        content: str | dict,
        content_type: str = "general",
    ) -> None:
        """Add content to the knowledge graph via Cognee's ECL pipeline.

        This is the primary way to add knowledge that will be:
        1. Extracted (entities, relationships)
        2. Cognified (embeddings, graph connections)
        3. Loaded (stored in graph + vector store)

        Args:
            content: Text or structured content to add
            content_type: Type hint for better extraction

        Raises:
            CogneeGraphError: If adding to the knowledge graph fails
        """
        dataset_name = self._get_dataset_name([content_type])

        try:
            if isinstance(content, dict):
                content = json.dumps(content)

            await cognee.add(content, dataset_name=dataset_name)
            await cognee.cognify(dataset_name=dataset_name)

            self._log.debug("Added to knowledge graph", content_type=content_type)
        except Exception as e:
            self._log.error(
                "Failed to add to knowledge graph",
                content_type=content_type,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise CogneeGraphError(
                f"Failed to add content to knowledge graph (type: {content_type}): {e}"
            ) from e

    async def query_knowledge_graph(
        self,
        query: str,
        content_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the knowledge graph with natural language.

        Uses Cognee's hybrid search to find relevant knowledge
        across graph and vector stores.

        Args:
            query: Natural language query
            content_types: Optional filter by content type
            limit: Maximum results

        Returns:
            Relevant knowledge items

        Raises:
            CogneeSearchError: If searching the knowledge graph fails
        """
        all_results = []
        errors = []

        # Search across specified content types or all
        types_to_search = content_types or ["codebase", "tests", "failures", "general"]

        for content_type in types_to_search:
            dataset_name = self._get_dataset_name([content_type])
            try:
                results = await cognee.search(
                    query_text=query,
                    datasets=[dataset_name],
                    top_k=limit // len(types_to_search),
                )
                if results:
                    for result in results:
                        if isinstance(result, str):
                            all_results.append({"content": result, "type": content_type})
                        else:
                            result["type"] = content_type
                            all_results.append(result)
            except Exception as e:
                self._log.error(
                    "Failed to search content type",
                    content_type=content_type,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                errors.append(f"{content_type}: {e}")

        # If ALL searches failed, raise an error
        if errors and not all_results:
            raise CogneeSearchError(
                f"Knowledge graph query failed for all content types. Errors: {'; '.join(errors)}"
            )

        # Log partial failures but return available results
        if errors:
            self._log.warning(
                "Partial knowledge graph query failure",
                query=query[:50] if query else "",
                failed_types=len(errors),
                successful_results=len(all_results),
            )

        return all_results[:limit]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Get entities related to a given entity via graph traversal.

        Uses Cognee's graph capabilities for multi-hop reasoning.

        Args:
            entity_id: Starting entity ID
            relationship_types: Filter by relationship types
            hops: Maximum traversal depth

        Returns:
            Related entities with relationship info
        """
        # Build a query that leverages Cognee's graph search
        query = f"entities related to {entity_id}"
        if relationship_types:
            query += f" via {', '.join(relationship_types)}"

        results = await self.query_knowledge_graph(
            query=query,
            limit=20,
        )

        return results

    # =========================================================================
    # Discovery Patterns Interface (replaces PatternService + CloudflareVectorize)
    # =========================================================================

    async def store_discovery_pattern(
        self,
        pattern_type: str,
        pattern_name: str,
        pattern_signature: str,
        pattern_data: dict[str, Any],
        source_url: str | None = None,
        source_project_id: str | None = None,
    ) -> str:
        """Store a UI discovery pattern for cross-project learning.

        This replaces PatternService._store_pattern() and uses Cognee's ECL pipeline
        for embedding generation and storage instead of Cloudflare Vectorize.

        Args:
            pattern_type: Type of pattern (page_layout, navigation, form, authentication, etc.)
            pattern_name: Human-readable pattern name
            pattern_signature: Hash for deduplication
            pattern_data: Full pattern details including features
            source_url: URL where pattern was discovered
            source_project_id: Project ID where pattern was found

        Returns:
            Pattern ID
        """
        # Create searchable text for embedding
        embed_parts = [
            f"Pattern type: {pattern_type}",
            f"Pattern name: {pattern_name}",
        ]

        features = pattern_data.get("features", {})
        for key, value in features.items():
            if value:
                embed_parts.append(f"{key}: {value}")

        embed_text = " | ".join(embed_parts)

        # Generate pattern ID
        pattern_id = f"discovery_{pattern_signature}"

        # Store via Cognee ECL pipeline
        value = {
            "id": pattern_id,
            "pattern_type": pattern_type,
            "pattern_name": pattern_name,
            "pattern_signature": pattern_signature,
            "pattern_data": pattern_data,
            "source_url": source_url,
            "source_project_id": source_project_id,
            "times_seen": 1,
            "projects_seen": 1,
            "test_success_rate": 0.0,
            "self_heal_success_rate": 0.0,
            "created_at": datetime.now(UTC).isoformat(),
        }

        await self.put(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
            value=value,
            embed_text=embed_text,
        )

        self._log.info(
            "Stored discovery pattern",
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            pattern_name=pattern_name,
        )

        return pattern_id

    async def find_similar_discovery_patterns(
        self,
        query_text: str,
        pattern_type: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Find similar UI patterns for cross-project learning.

        This replaces PatternService.find_similar_patterns() using Cognee's
        semantic search capabilities.

        Args:
            query_text: Text description or pattern to match
            pattern_type: Optional filter by pattern type
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching patterns with similarity scores
        """
        # Build namespace for search
        if pattern_type:
            namespaces = [["discovery_patterns", pattern_type]]
        else:
            # Search across all pattern types
            namespaces = [["discovery_patterns"]]

        results = []
        errors = []
        for ns in namespaces:
            try:
                matches = await self.search(
                    namespace=ns,
                    query=query_text,
                    limit=limit,
                )
                results.extend(matches)
            except CogneeSearchError as e:
                self._log.error(
                    "Failed to search discovery patterns",
                    namespace=ns,
                    error=str(e),
                )
                errors.append(str(e))

        # If all searches failed, raise an error
        if errors and not results:
            raise CogneeSearchError(
                f"Failed to search discovery patterns: {'; '.join(errors)}"
            )

        # Filter by similarity and deduplicate
        filtered = []
        seen_ids = set()
        for match in results:
            if isinstance(match, dict):
                pattern_id = match.get("id", match.get("key", ""))
                similarity = match.get("similarity", match.get("score", 0.8))

                if pattern_id not in seen_ids and similarity >= min_similarity:
                    seen_ids.add(pattern_id)
                    match["similarity"] = similarity
                    filtered.append(match)

        # Sort by similarity descending
        filtered.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return filtered[:limit]

    async def increment_pattern_times_seen(
        self,
        pattern_id: str,
        pattern_type: str,
    ) -> bool:
        """Increment times_seen counter for an existing pattern.

        Args:
            pattern_id: Pattern ID
            pattern_type: Pattern type (for namespace lookup)

        Returns:
            True if updated successfully, False if pattern not found

        Raises:
            CogneeStorageError: If the update operation fails
        """
        # Get current pattern - this may raise CogneeRetrievalError
        current = await self.get(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
        )

        if not current:
            self._log.warning(
                "Pattern not found for increment",
                pattern_id=pattern_id,
                pattern_type=pattern_type,
            )
            return False

        current["times_seen"] = current.get("times_seen", 0) + 1
        current["updated_at"] = datetime.now(UTC).isoformat()

        # This may raise CogneeStorageError
        await self.put(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
            value=current,
        )
        return True

    async def update_discovery_pattern_stats(
        self,
        pattern_id: str,
        pattern_type: str,
        test_passed: bool,
        self_healed: bool = False,
    ) -> bool:
        """Update pattern success rates after test execution.

        This enables learning: patterns with high success rates
        are prioritized in future discoveries.

        Args:
            pattern_id: Pattern ID
            pattern_type: Pattern type (for namespace lookup)
            test_passed: Whether the test passed
            self_healed: Whether self-healing was applied

        Returns:
            True if updated successfully, False if pattern not found

        Raises:
            CogneeStorageError: If the update operation fails
        """
        # Get current pattern - may raise CogneeRetrievalError
        current = await self.get(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
        )

        if not current:
            self._log.warning(
                "Pattern not found for stats update",
                pattern_id=pattern_id,
                pattern_type=pattern_type,
            )
            return False

        times_seen = current.get("times_seen", 1)
        current_test_rate = float(current.get("test_success_rate", 0) or 0)
        current_heal_rate = float(current.get("self_heal_success_rate", 0) or 0)

        # Calculate new rolling average
        new_test_rate = (
            (current_test_rate * (times_seen - 1)) + (100 if test_passed else 0)
        ) / times_seen

        new_heal_rate = current_heal_rate
        if self_healed:
            new_heal_rate = (
                (current_heal_rate * (times_seen - 1)) + 100
            ) / times_seen

        current["test_success_rate"] = round(new_test_rate, 2)
        current["self_heal_success_rate"] = round(new_heal_rate, 2)
        current["updated_at"] = datetime.now(UTC).isoformat()

        # This may raise CogneeStorageError
        await self.put(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
            value=current,
        )

        self._log.info(
            "Updated discovery pattern stats",
            pattern_id=pattern_id,
            test_success_rate=new_test_rate,
            self_heal_success_rate=new_heal_rate,
        )

        return True

    async def get_discovery_pattern_insights(
        self,
        pattern_type: str | None = None,
    ) -> dict[str, Any]:
        """Get insights about stored discovery patterns.

        Returns statistics about pattern types, success rates, etc.

        Args:
            pattern_type: Optional filter by pattern type

        Returns:
            Insights dictionary with statistics
        """
        # Search all patterns
        if pattern_type:
            namespaces = [["discovery_patterns", pattern_type]]
        else:
            namespaces = [["discovery_patterns"]]

        patterns = []
        errors = []
        for ns in namespaces:
            try:
                # Use a generic query to get all patterns
                results = await self.search(
                    namespace=ns,
                    query="pattern",  # Generic query to match all
                    limit=1000,
                )
                patterns.extend(results)
            except CogneeSearchError as e:
                self._log.error(
                    "Failed to get patterns for insights",
                    namespace=ns,
                    error=str(e),
                )
                errors.append(str(e))

        # If all searches failed, raise an error
        if errors and not patterns:
            raise CogneeSearchError(
                f"Failed to retrieve discovery pattern insights: {'; '.join(errors)}"
            )

        # Calculate insights
        by_type = {}
        total_patterns = len(patterns)
        total_seen = 0
        avg_test_success = 0
        avg_heal_success = 0

        for p in patterns:
            if not isinstance(p, dict):
                continue

            ptype = p.get("pattern_type", "unknown")
            if ptype not in by_type:
                by_type[ptype] = {"count": 0, "total_seen": 0}
            by_type[ptype]["count"] += 1
            by_type[ptype]["total_seen"] += p.get("times_seen", 0)
            total_seen += p.get("times_seen", 0)
            avg_test_success += float(p.get("test_success_rate", 0) or 0)
            avg_heal_success += float(p.get("self_heal_success_rate", 0) or 0)

        if total_patterns > 0:
            avg_test_success /= total_patterns
            avg_heal_success /= total_patterns

        return {
            "total_patterns": total_patterns,
            "total_occurrences": total_seen,
            "by_type": by_type,
            "avg_test_success_rate": round(avg_test_success, 2),
            "avg_self_heal_success_rate": round(avg_heal_success, 2),
        }


# =========================================================================
# Global Instance Management
# =========================================================================

_cognee_client: CogneeKnowledgeClient | None = None


def get_cognee_client(
    org_id: str | None = None,
    project_id: str | None = None,
) -> CogneeKnowledgeClient:
    """Get or create the global Cognee client instance.

    Args:
        org_id: Organization ID (defaults to DEFAULT_ORG_ID env var)
        project_id: Project ID (defaults to DEFAULT_PROJECT_ID env var)

    Returns:
        CogneeKnowledgeClient instance
    """
    global _cognee_client

    # Ensure Cognee is configured on first use
    _configure_cognee()

    org = org_id or os.environ.get("DEFAULT_ORG_ID", "default")
    proj = project_id or os.environ.get("DEFAULT_PROJECT_ID", "default")

    if _cognee_client is None or (
        _cognee_client.org_id != org or _cognee_client.project_id != proj
    ):
        _cognee_client = CogneeKnowledgeClient(org_id=org, project_id=proj)

    return _cognee_client


def reset_cognee_client() -> None:
    """Reset the global Cognee client instance."""
    global _cognee_client
    _cognee_client = None


async def init_cognee_client(
    org_id: str | None = None,
    project_id: str | None = None,
) -> CogneeKnowledgeClient:
    """Initialize and return the Cognee client.

    This function validates that Cognee is properly configured and available.
    If Cognee is not installed or cannot be initialized, an error will be raised.

    Args:
        org_id: Organization ID
        project_id: Project ID

    Returns:
        Initialized CogneeKnowledgeClient

    Raises:
        ImportError: If cognee package is not installed
        CogneeError: If Cognee cannot be initialized
    """
    client = get_cognee_client(org_id=org_id, project_id=project_id)
    logger.info(
        "Cognee client initialized",
        org_id=client.org_id,
        project_id=client.project_id,
        cognee_version=getattr(cognee, "__version__", "unknown"),
    )
    return client

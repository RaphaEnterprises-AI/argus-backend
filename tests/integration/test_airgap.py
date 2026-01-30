"""
Air-Gap Integration Tests for Argus Enterprise.

Tests for validating fully offline operation with zero internet connectivity:
- Ollama provider for local LLM inference
- MinIO storage for artifacts
- No external API calls when using air-gap configuration

RAP-302: Test fully offline operation for air-gap deployments

Run with:
    pytest tests/integration/test_airgap.py -v

For full air-gap validation:
    pytest tests/integration/test_airgap.py -v -m airgap --network-isolation

Requirements:
    - Ollama running locally: ollama serve
    - MinIO running locally: docker run -p 9000:9000 minio/minio server /data
    - Models pre-pulled: ollama pull llama3.1:8b
"""

import asyncio
import base64
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import patch, MagicMock

import httpx
import pytest
import structlog

logger = structlog.get_logger()


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class AirGapTestConfig:
    """Configuration for air-gap tests."""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"  # Smaller model for testing
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "argus-test-artifacts"
    minio_secure: bool = False

    @classmethod
    def from_env(cls) -> "AirGapTestConfig":
        """Load configuration from environment."""
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            minio_endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            minio_bucket=os.getenv("MINIO_BUCKET", "argus-test-artifacts"),
            minio_secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )


@dataclass
class AirGapTestResult:
    """Result from an air-gap validation test."""
    test_name: str
    component: str
    passed: bool
    latency_ms: float
    error: str | None = None
    details: dict = field(default_factory=dict)
    external_calls_detected: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "component": self.component,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "details": self.details,
            "external_calls_detected": self.external_calls_detected,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# NETWORK MONITORING
# =============================================================================

class NetworkMonitor:
    """Monitors network calls to detect external API usage.

    Used to verify that air-gap mode doesn't make external calls.
    """

    # Known external API endpoints that should NOT be called in air-gap mode
    EXTERNAL_ENDPOINTS = [
        "api.anthropic.com",
        "api.openai.com",
        "api.openrouter.ai",
        "api.cohere.ai",
        "generativelanguage.googleapis.com",
        "bedrock.us-east-1.amazonaws.com",
        "*.openai.azure.com",
        # Cloud storage
        "*.r2.cloudflarestorage.com",
        "s3.amazonaws.com",
        "*.s3.amazonaws.com",
        "storage.googleapis.com",
    ]

    # Local endpoints that are allowed
    ALLOWED_ENDPOINTS = [
        "localhost",
        "127.0.0.1",
        "::1",
        "host.docker.internal",
        "kubernetes.default.svc",
    ]

    def __init__(self):
        self.calls: list[dict] = []
        self._original_socket_connect = None
        self._original_httpx_send = None
        self._monitoring = False

    def start(self):
        """Start monitoring network calls."""
        if self._monitoring:
            return

        self._monitoring = True
        self._original_socket_connect = socket.socket.connect

        monitor = self

        def patched_connect(self_socket, address):
            host = address[0] if isinstance(address, tuple) else str(address)
            port = address[1] if isinstance(address, tuple) else 0

            monitor.calls.append({
                "host": host,
                "port": port,
                "timestamp": datetime.utcnow().isoformat(),
            })

            return monitor._original_socket_connect(self_socket, address)

        socket.socket.connect = patched_connect

    def stop(self):
        """Stop monitoring and restore original functions."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._original_socket_connect:
            socket.socket.connect = self._original_socket_connect

    def get_external_calls(self) -> list[str]:
        """Get list of detected external API calls."""
        external = []
        for call in self.calls:
            host = call["host"]

            # Check if it's a local endpoint
            is_local = any(
                allowed in host or host.startswith(allowed)
                for allowed in self.ALLOWED_ENDPOINTS
            )

            if not is_local:
                # Check against known external endpoints
                for ext in self.EXTERNAL_ENDPOINTS:
                    if ext.startswith("*"):
                        if host.endswith(ext[1:]):
                            external.append(f"{host}:{call['port']}")
                            break
                    elif ext in host:
                        external.append(f"{host}:{call['port']}")
                        break
                else:
                    # Unknown external host
                    external.append(f"{host}:{call['port']}")

        return list(set(external))

    def reset(self):
        """Clear recorded calls."""
        self.calls = []


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def airgap_config() -> AirGapTestConfig:
    """Provide air-gap test configuration."""
    return AirGapTestConfig.from_env()


@pytest.fixture
def network_monitor():
    """Provide network monitor for detecting external calls."""
    monitor = NetworkMonitor()
    yield monitor
    monitor.stop()


@pytest.fixture
async def ollama_provider(airgap_config):
    """Provide Ollama provider for testing."""
    from src.core.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(base_url=airgap_config.ollama_host)
    yield provider
    await provider.close()


@pytest.fixture
async def minio_provider(airgap_config):
    """Provide MinIO storage provider for testing."""
    from src.services.storage.minio_provider import MinIOStorageProvider, MinIOConfig

    config = MinIOConfig(
        endpoint=airgap_config.minio_endpoint,
        access_key=airgap_config.minio_access_key,
        secret_key=airgap_config.minio_secret_key,
        bucket=airgap_config.minio_bucket,
        secure=airgap_config.minio_secure,
    )
    provider = MinIOStorageProvider(config)

    # Ensure test bucket exists
    try:
        await provider.ensure_bucket()
    except Exception as e:
        pytest.skip(f"MinIO not available: {e}")

    yield provider


# =============================================================================
# OLLAMA PROVIDER TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.airgap
async def test_ollama_connectivity(airgap_config):
    """Test that Ollama server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{airgap_config.ollama_host}/api/tags")
            assert response.status_code == 200, f"Ollama not responding: {response.status_code}"

            data = response.json()
            logger.info(
                "Ollama connectivity verified",
                models=len(data.get("models", [])),
            )
    except httpx.ConnectError:
        pytest.skip("Ollama not running - start with 'ollama serve'")


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_ollama_no_external_calls(airgap_config, network_monitor):
    """Verify Ollama provider doesn't make external API calls."""
    from src.core.providers.ollama_provider import OllamaProvider, ChatMessage

    network_monitor.start()
    network_monitor.reset()

    try:
        provider = OllamaProvider(base_url=airgap_config.ollama_host)

        # Make a chat completion request
        try:
            response = await provider.chat(
                messages=[ChatMessage(role="user", content="Say 'hello' in one word.")],
                model=airgap_config.ollama_model,
                max_tokens=50,
            )

            assert response.content, "Response should have content"
            logger.info(
                "Ollama chat completed",
                model=response.model,
                tokens=response.output_tokens,
            )
        except Exception as e:
            if "not found" in str(e).lower():
                pytest.skip(f"Model {airgap_config.ollama_model} not available - pull with 'ollama pull {airgap_config.ollama_model}'")
            raise

        await provider.close()

    finally:
        network_monitor.stop()

    # Check for external calls
    external_calls = network_monitor.get_external_calls()
    assert len(external_calls) == 0, f"External API calls detected: {external_calls}"


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_ollama_embedding_no_external_calls(airgap_config, network_monitor):
    """Verify Ollama embedding doesn't make external API calls."""
    from src.core.providers.ollama_provider import OllamaProvider

    network_monitor.start()
    network_monitor.reset()

    try:
        provider = OllamaProvider(base_url=airgap_config.ollama_host)

        try:
            # Test embedding (requires nomic-embed-text model)
            embeddings = await provider.embed("Test embedding for air-gap validation")

            assert embeddings, "Should return embeddings"
            assert isinstance(embeddings, list), "Embeddings should be a list"
            assert len(embeddings) > 0, "Embeddings should not be empty"

            logger.info(
                "Ollama embedding completed",
                dimensions=len(embeddings),
            )
        except Exception as e:
            if "not found" in str(e).lower():
                pytest.skip("Embedding model not available - pull with 'ollama pull nomic-embed-text'")
            raise

        await provider.close()

    finally:
        network_monitor.stop()

    external_calls = network_monitor.get_external_calls()
    assert len(external_calls) == 0, f"External API calls detected: {external_calls}"


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_ollama_health_check(ollama_provider):
    """Test Ollama health check functionality."""
    health = await ollama_provider.health_check()

    assert health["status"] in ["healthy", "error"], f"Unexpected status: {health['status']}"

    if health["status"] == "healthy":
        assert health["authenticated"] == True  # Ollama doesn't require auth
        logger.info(
            "Ollama health check passed",
            models=health.get("available_models", 0),
        )
    else:
        pytest.skip(f"Ollama unhealthy: {health.get('error')}")


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_ollama_list_models(ollama_provider):
    """Test listing available Ollama models."""
    models = await ollama_provider.list_models()

    assert isinstance(models, list), "Should return list of models"

    logger.info(
        "Ollama models listed",
        count=len(models),
        models=[m.model_id for m in models[:5]],
    )


# =============================================================================
# MINIO STORAGE TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.airgap
async def test_minio_connectivity(airgap_config):
    """Test that MinIO server is reachable."""
    try:
        # MinIO uses different port for API vs console
        protocol = "https" if airgap_config.minio_secure else "http"
        url = f"{protocol}://{airgap_config.minio_endpoint}/minio/health/live"

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(url)
            # MinIO returns 200 for healthy, but some endpoints return different codes
            assert response.status_code in [200, 403], f"MinIO not responding correctly: {response.status_code}"

            logger.info("MinIO connectivity verified")
    except httpx.ConnectError:
        pytest.skip("MinIO not running - start with 'docker run -p 9000:9000 minio/minio server /data'")


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_minio_local_storage(minio_provider, network_monitor):
    """Verify MinIO works for screenshot/artifact storage without external calls."""
    network_monitor.start()
    network_monitor.reset()

    try:
        # Create a simple test image (1x1 red pixel PNG)
        test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        # Store the screenshot
        ref = await minio_provider.store_screenshot(
            base64_data=test_image_base64,
            metadata={
                "test_id": "airgap-test-001",
                "purpose": "air-gap validation",
            }
        )

        assert ref.artifact_id, "Should have artifact ID"
        assert ref.url, "Should have URL"
        assert ref.storage_backend.value == "minio", "Should use MinIO backend"

        logger.info(
            "Screenshot stored in MinIO",
            artifact_id=ref.artifact_id,
            url=ref.url[:100] + "..." if len(ref.url) > 100 else ref.url,
        )

        # Retrieve the artifact
        data = await minio_provider.get_artifact(ref.artifact_id)
        assert data is not None, "Should retrieve stored artifact"
        assert len(data) > 0, "Retrieved data should not be empty"

        # Verify content matches
        original_data = base64.b64decode(test_image_base64)
        assert data == original_data, "Retrieved data should match original"

        # Clean up
        deleted = await minio_provider.delete_artifact(ref.artifact_id)
        assert deleted, "Should successfully delete artifact"

        logger.info("MinIO storage test passed")

    finally:
        network_monitor.stop()

    # Verify no external calls
    external_calls = network_monitor.get_external_calls()
    # Filter out localhost/MinIO calls
    truly_external = [
        c for c in external_calls
        if "localhost" not in c and "127.0.0.1" not in c
    ]
    assert len(truly_external) == 0, f"External API calls detected: {truly_external}"


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_minio_artifact_lifecycle(minio_provider):
    """Test full artifact lifecycle: store, retrieve, list, delete."""
    from src.services.storage.base import ArtifactType

    # Store a test artifact
    test_data = b"Test artifact content for air-gap validation"
    ref = await minio_provider.store_artifact(
        data=test_data,
        artifact_type=ArtifactType.LOG,
        content_type="text/plain",
        metadata={"test": "lifecycle"},
    )

    assert ref.artifact_id.startswith("log_"), "Artifact ID should have correct prefix"

    # List artifacts
    artifacts = await minio_provider.list_artifacts(artifact_type=ArtifactType.LOG, limit=10)
    artifact_ids = [a.artifact_id for a in artifacts]
    assert ref.artifact_id in artifact_ids, "Stored artifact should appear in list"

    # Get URL
    url = await minio_provider.get_artifact_url(ref.artifact_id)
    assert url, "Should generate URL"

    # Delete
    deleted = await minio_provider.delete_artifact(ref.artifact_id)
    assert deleted, "Should delete artifact"

    logger.info("MinIO artifact lifecycle test passed")


@pytest.mark.asyncio
@pytest.mark.airgap
async def test_minio_health_check(minio_provider):
    """Test MinIO health check functionality."""
    health = await minio_provider.health_check()

    assert health["backend"] == "minio", "Should report MinIO backend"

    if health["healthy"]:
        assert health["bucket_exists"], "Bucket should exist"
        logger.info(
            "MinIO health check passed",
            endpoint=health.get("endpoint"),
            bucket=health.get("bucket"),
        )
    else:
        pytest.skip(f"MinIO unhealthy: {health.get('error')}")


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.airgap
async def test_full_test_workflow_offline(airgap_config, network_monitor):
    """Verify complete test generation workflow works offline.

    This test simulates the core Argus workflow:
    1. Code analysis (via Ollama)
    2. Test generation (via Ollama)
    3. Result storage (via MinIO)

    All operations should work without any external network calls.
    """
    from src.core.providers.ollama_provider import OllamaProvider, ChatMessage
    from src.services.storage.minio_provider import MinIOStorageProvider, MinIOConfig

    network_monitor.start()
    network_monitor.reset()

    ollama = None
    minio = None

    try:
        # Initialize providers
        ollama = OllamaProvider(base_url=airgap_config.ollama_host)

        minio_config = MinIOConfig(
            endpoint=airgap_config.minio_endpoint,
            access_key=airgap_config.minio_access_key,
            secret_key=airgap_config.minio_secret_key,
            bucket=airgap_config.minio_bucket,
            secure=airgap_config.minio_secure,
        )
        minio = MinIOStorageProvider(minio_config)

        # Step 1: Code Analysis (simulated)
        analysis_prompt = """
        Analyze this Python function and identify potential test cases:

        def calculate_discount(price: float, discount_percent: float) -> float:
            if price < 0:
                raise ValueError("Price cannot be negative")
            if discount_percent < 0 or discount_percent > 100:
                raise ValueError("Discount must be between 0 and 100")
            return price * (1 - discount_percent / 100)

        List 3 test cases in a simple format.
        """

        try:
            analysis_response = await ollama.chat(
                messages=[ChatMessage(role="user", content=analysis_prompt)],
                model=airgap_config.ollama_model,
                max_tokens=500,
            )

            assert analysis_response.content, "Should get analysis response"
            logger.info(
                "Code analysis completed",
                response_length=len(analysis_response.content),
            )
        except Exception as e:
            if "not found" in str(e).lower():
                pytest.skip(f"Model not available: {airgap_config.ollama_model}")
            raise

        # Step 2: Generate test suggestion
        test_prompt = """
        Generate a simple pytest test function for this test case:
        "Test that calculate_discount returns correct value for 100 price with 20% discount"

        Return only the Python code, no explanations.
        """

        test_response = await ollama.chat(
            messages=[ChatMessage(role="user", content=test_prompt)],
            model=airgap_config.ollama_model,
            max_tokens=300,
        )

        assert test_response.content, "Should get test code"
        logger.info(
            "Test generation completed",
            response_length=len(test_response.content),
        )

        # Step 3: Store test result
        await minio.ensure_bucket()

        result_data = {
            "test_id": "airgap-workflow-test",
            "analysis": analysis_response.content[:200],
            "generated_test": test_response.content[:200],
            "model": airgap_config.ollama_model,
            "timestamp": datetime.utcnow().isoformat(),
        }

        import json
        result_bytes = json.dumps(result_data).encode("utf-8")

        from src.services.storage.base import ArtifactType
        ref = await minio.store_artifact(
            data=result_bytes,
            artifact_type=ArtifactType.TEST_RESULT,
            content_type="application/json",
            metadata={"workflow": "airgap-validation"},
        )

        assert ref.artifact_id, "Should store test result"
        logger.info(
            "Test result stored",
            artifact_id=ref.artifact_id,
        )

        # Clean up
        await minio.delete_artifact(ref.artifact_id)

    finally:
        if ollama:
            await ollama.close()
        network_monitor.stop()

    # Verify no external calls were made
    external_calls = network_monitor.get_external_calls()
    truly_external = [
        c for c in external_calls
        if not any(local in c for local in ["localhost", "127.0.0.1", airgap_config.minio_endpoint.split(":")[0]])
    ]

    assert len(truly_external) == 0, (
        f"External API calls detected during offline workflow: {truly_external}\n"
        "Air-gap mode should not make any external network requests."
    )

    logger.info("Full offline workflow test passed - no external calls detected")


# =============================================================================
# AIR-GAP VALIDATION REPORT
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.airgap
async def test_airgap_validation_report(airgap_config, network_monitor, capsys):
    """Generate a comprehensive air-gap validation report.

    This test runs all validation checks and produces a report
    suitable for air-gap deployment certification.
    """
    import json

    results: list[AirGapTestResult] = []

    # Test 1: Ollama Connectivity
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{airgap_config.ollama_host}/api/tags")
            latency = (time.perf_counter() - start) * 1000

            results.append(AirGapTestResult(
                test_name="ollama_connectivity",
                component="ollama",
                passed=response.status_code == 200,
                latency_ms=latency,
                details={"models_available": len(response.json().get("models", []))},
            ))
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        results.append(AirGapTestResult(
            test_name="ollama_connectivity",
            component="ollama",
            passed=False,
            latency_ms=latency,
            error=str(e),
        ))

    # Test 2: MinIO Connectivity
    start = time.perf_counter()
    try:
        from src.services.storage.minio_provider import MinIOStorageProvider, MinIOConfig

        config = MinIOConfig(
            endpoint=airgap_config.minio_endpoint,
            access_key=airgap_config.minio_access_key,
            secret_key=airgap_config.minio_secret_key,
            bucket=airgap_config.minio_bucket,
            secure=airgap_config.minio_secure,
        )
        provider = MinIOStorageProvider(config)
        health = await provider.health_check()
        latency = (time.perf_counter() - start) * 1000

        results.append(AirGapTestResult(
            test_name="minio_connectivity",
            component="minio",
            passed=health.get("healthy", False),
            latency_ms=latency,
            details=health,
        ))
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        results.append(AirGapTestResult(
            test_name="minio_connectivity",
            component="minio",
            passed=False,
            latency_ms=latency,
            error=str(e),
        ))

    # Test 3: Ollama Chat (with network monitoring)
    network_monitor.start()
    network_monitor.reset()
    start = time.perf_counter()

    try:
        from src.core.providers.ollama_provider import OllamaProvider, ChatMessage

        provider = OllamaProvider(base_url=airgap_config.ollama_host)
        response = await provider.chat(
            messages=[ChatMessage(role="user", content="Say 'test' in one word.")],
            model=airgap_config.ollama_model,
            max_tokens=10,
        )
        await provider.close()

        latency = (time.perf_counter() - start) * 1000
        external_calls = network_monitor.get_external_calls()

        results.append(AirGapTestResult(
            test_name="ollama_chat_no_external",
            component="ollama",
            passed=len(external_calls) == 0 and bool(response.content),
            latency_ms=latency,
            details={
                "response_received": bool(response.content),
                "tokens": response.output_tokens,
            },
            external_calls_detected=external_calls,
        ))
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        results.append(AirGapTestResult(
            test_name="ollama_chat_no_external",
            component="ollama",
            passed=False,
            latency_ms=latency,
            error=str(e),
        ))
    finally:
        network_monitor.stop()

    # Generate Report
    report = {
        "title": "Air-Gap Validation Report",
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "ollama_host": airgap_config.ollama_host,
            "ollama_model": airgap_config.ollama_model,
            "minio_endpoint": airgap_config.minio_endpoint,
            "minio_bucket": airgap_config.minio_bucket,
        },
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "external_calls_detected": any(r.external_calls_detected for r in results),
        },
        "tests": [r.to_dict() for r in results],
        "certification": {
            "air_gap_ready": all(r.passed for r in results) and not any(r.external_calls_detected for r in results),
            "notes": [],
        },
    }

    # Add certification notes
    if report["summary"]["failed"] > 0:
        report["certification"]["notes"].append(
            f"{report['summary']['failed']} test(s) failed - review errors before deployment"
        )

    if report["summary"]["external_calls_detected"]:
        report["certification"]["notes"].append(
            "External network calls detected - air-gap certification FAILED"
        )

    if report["certification"]["air_gap_ready"]:
        report["certification"]["notes"].append(
            "All tests passed with no external calls - READY for air-gap deployment"
        )

    # Print report
    print("\n" + "=" * 70)
    print("AIR-GAP VALIDATION REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)

    # Assert overall pass
    assert report["certification"]["air_gap_ready"], (
        f"Air-gap validation failed: {report['certification']['notes']}"
    )


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register air-gap marker."""
    config.addinivalue_line(
        "markers", "airgap: mark test as air-gap validation test"
    )

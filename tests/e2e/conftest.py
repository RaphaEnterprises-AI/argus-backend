"""E2E test fixtures — Docker container management.

Provides a session-scoped DVWA (Damn Vulnerable Web Application) container
for real-world penetration testing validation.
"""

from __future__ import annotations

import time

import httpx
import pytest


def _wait_for_health(url: str, timeout: int = 60, interval: float = 2.0) -> None:
    """Wait for a URL to respond with 200 (or any non-error)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=5, follow_redirects=True)
            if resp.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Service at {url} did not become healthy within {timeout}s")


@pytest.fixture(scope="session")
def dvwa_container():
    """Start a DVWA Docker container for the test session.

    Requires Docker to be running. Skips if docker SDK is unavailable.

    Yields the base URL (http://localhost:4280).
    """
    docker = pytest.importorskip("docker", reason="docker SDK not installed")

    client = docker.from_env()
    container = None

    try:
        container = client.containers.run(
            "vulnerables/web-dvwa",
            detach=True,
            ports={"80/tcp": 4280},
            auto_remove=True,
            name="argus-dvwa-test",
        )

        base_url = "http://localhost:4280"
        _wait_for_health(f"{base_url}/login.php", timeout=90)

        # DVWA needs initial database setup — POST to setup.php
        try:
            httpx.post(
                f"{base_url}/setup.php",
                data={"create_db": "Create / Reset Database"},
                follow_redirects=True,
                timeout=30,
            )
        except httpx.HTTPError:
            pass  # May redirect, that's fine

        yield base_url

    finally:
        if container:
            try:
                container.stop(timeout=5)
            except Exception:
                pass

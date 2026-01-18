"""MCP Authentication - OAuth2 Device Flow for CLI/MCP tools.

This module provides authentication for MCP servers and CLI tools
using the OAuth2 Device Authorization Grant (RFC 8628).

Usage:
    from src.mcp.auth import MCPAuthenticator

    auth = MCPAuthenticator()
    token = await auth.get_token()

    # Use token for API calls
    headers = {"Authorization": f"Bearer {token}"}

The authentication flow:
1. If cached token exists and is valid, return it
2. Otherwise, initiate device flow:
   - Display URL and code to user
   - User opens URL in browser, signs in
   - Module polls for completion
3. Cache tokens for future use
"""

import asyncio
import json
import os
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================

# Default API base URL
DEFAULT_API_URL = os.getenv("ARGUS_API_URL", "https://api.heyargus.ai")

# Token cache location
TOKEN_CACHE_DIR = Path.home() / ".argus"
TOKEN_CACHE_FILE = TOKEN_CACHE_DIR / "tokens.json"

# Device flow polling
DEFAULT_POLL_INTERVAL = 5  # seconds
MAX_POLL_DURATION = 600  # 10 minutes


# =============================================================================
# Token Storage
# =============================================================================

@dataclass
class TokenData:
    """Stored token data."""
    access_token: str
    refresh_token: str | None
    expires_at: datetime
    scope: str
    user_id: str | None = None
    organization_id: str | None = None


def load_cached_tokens() -> TokenData | None:
    """Load tokens from disk cache."""
    if not TOKEN_CACHE_FILE.exists():
        return None

    try:
        data = json.loads(TOKEN_CACHE_FILE.read_text())
        expires_at = datetime.fromisoformat(data["expires_at"])

        # Check if expired (with 5 minute buffer)
        if expires_at < datetime.now(UTC) + timedelta(minutes=5):
            return None

        return TokenData(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=expires_at,
            scope=data.get("scope", "read write"),
            user_id=data.get("user_id"),
            organization_id=data.get("organization_id"),
        )
    except Exception as e:
        logger.debug("Failed to load cached tokens", error=str(e))
        return None


def save_tokens(token_data: TokenData) -> None:
    """Save tokens to disk cache."""
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "access_token": token_data.access_token,
        "refresh_token": token_data.refresh_token,
        "expires_at": token_data.expires_at.isoformat(),
        "scope": token_data.scope,
        "user_id": token_data.user_id,
        "organization_id": token_data.organization_id,
    }

    TOKEN_CACHE_FILE.write_text(json.dumps(data, indent=2))
    TOKEN_CACHE_FILE.chmod(0o600)  # Secure file permissions


def clear_tokens() -> None:
    """Clear cached tokens."""
    if TOKEN_CACHE_FILE.exists():
        TOKEN_CACHE_FILE.unlink()


# =============================================================================
# Device Flow Client
# =============================================================================

class MCPAuthenticator:
    """OAuth2 Device Flow authenticator for MCP servers."""

    def __init__(
        self,
        api_url: str = None,
        client_id: str = "argus-mcp",
        scope: str = "read write",
        auto_open_browser: bool = True,
    ):
        """Initialize the authenticator.

        Args:
            api_url: Argus API base URL (default: from env or https://api.heyargus.ai)
            client_id: OAuth2 client ID for this MCP server
            scope: Space-separated scopes to request
            auto_open_browser: Automatically open browser for verification
        """
        self.api_url = api_url or os.getenv("ARGUS_API_URL", DEFAULT_API_URL)
        self.client_id = client_id
        self.scope = scope
        self.auto_open_browser = auto_open_browser
        self._token_data: TokenData | None = None
        self.log = logger.bind(component="mcp_auth")

    async def get_token(self, force_refresh: bool = False) -> str:
        """Get a valid access token.

        This method handles the full authentication flow:
        1. Check for cached token
        2. Refresh if expired
        3. Start device flow if needed

        Args:
            force_refresh: Force a new authentication even if cached token exists

        Returns:
            Valid access token string
        """
        # Check cached token
        if not force_refresh:
            cached = load_cached_tokens()
            if cached:
                self.log.debug("Using cached token")
                self._token_data = cached
                return cached.access_token

        # Try to refresh if we have a refresh token
        if self._token_data and self._token_data.refresh_token:
            try:
                new_token = await self._refresh_token(self._token_data.refresh_token)
                if new_token:
                    self._token_data = new_token
                    save_tokens(new_token)
                    return new_token.access_token
            except Exception as e:
                self.log.warning("Token refresh failed", error=str(e))

        # Start device flow
        token_data = await self._device_flow()
        self._token_data = token_data
        save_tokens(token_data)
        return token_data.access_token

    async def _device_flow(self) -> TokenData:
        """Execute OAuth2 device authorization flow."""
        async with httpx.AsyncClient() as client:
            # Step 1: Request device code
            self.log.info("Starting device authorization flow")

            response = await client.post(
                f"{self.api_url}/api/v1/auth/device/authorize",
                data={
                    "client_id": self.client_id,
                    "scope": self.scope,
                },
            )

            if response.status_code != 200:
                raise AuthenticationError(f"Failed to start device flow: {response.text}")

            data = response.json()
            device_code = data["device_code"]
            user_code = data["user_code"]
            verification_uri = data["verification_uri"]
            verification_uri_complete = data["verification_uri_complete"]
            expires_in = data["expires_in"]
            interval = data.get("interval", DEFAULT_POLL_INTERVAL)

            # Step 2: Display instructions to user
            self._display_auth_prompt(user_code, verification_uri, verification_uri_complete)

            # Step 3: Open browser if enabled
            if self.auto_open_browser:
                try:
                    webbrowser.open(verification_uri_complete)
                except Exception:
                    pass  # Browser opening is optional

            # Step 4: Poll for completion
            token_data = await self._poll_for_token(
                client, device_code, expires_in, interval
            )

            self.log.info("Device authorization completed successfully")
            return token_data

    def _display_auth_prompt(
        self,
        user_code: str,
        verification_uri: str,
        verification_uri_complete: str,
    ) -> None:
        """Display authentication instructions to the user."""
        # Check if we're in a terminal that supports formatting
        is_tty = sys.stdout.isatty()

        if is_tty:
            # Rich terminal output
            print("\n" + "=" * 60)
            print("  🔐 ARGUS AUTHENTICATION REQUIRED")
            print("=" * 60)
            print()
            print("  Please visit the following URL to sign in:")
            print()
            print(f"  \033[1;36m{verification_uri}\033[0m")
            print()
            print("  And enter this code:")
            print()
            print(f"  \033[1;33m{user_code}\033[0m")
            print()
            print("  Or visit directly:")
            print(f"  {verification_uri_complete}")
            print()
            print("  Waiting for authorization...")
            print("=" * 60 + "\n")
        else:
            # Plain text output for non-interactive environments
            print("\nARGUS AUTHENTICATION REQUIRED")
            print(f"Visit: {verification_uri}")
            print(f"Enter code: {user_code}")
            print(f"Or visit: {verification_uri_complete}")
            print("Waiting for authorization...\n")

    async def _poll_for_token(
        self,
        client: httpx.AsyncClient,
        device_code: str,
        expires_in: int,
        interval: int,
    ) -> TokenData:
        """Poll token endpoint until user authorizes or timeout."""
        start_time = time.time()
        max_time = min(expires_in, MAX_POLL_DURATION)

        while (time.time() - start_time) < max_time:
            await asyncio.sleep(interval)

            try:
                response = await client.post(
                    f"{self.api_url}/api/v1/auth/device/token",
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code,
                        "client_id": self.client_id,
                    },
                )

                if response.status_code == 200:
                    # Success!
                    data = response.json()
                    return TokenData(
                        access_token=data["access_token"],
                        refresh_token=data.get("refresh_token"),
                        expires_at=datetime.now(UTC) + timedelta(seconds=data["expires_in"]),
                        scope=data.get("scope", self.scope),
                    )

                # Check for expected errors
                error_data = response.json()
                error = error_data.get("detail", {}).get("error", error_data.get("error", ""))

                if error == "authorization_pending":
                    # User hasn't completed auth yet, keep polling
                    continue
                elif error == "slow_down":
                    # Increase polling interval
                    interval += 5
                    continue
                elif error == "expired_token":
                    raise AuthenticationError("Device code expired. Please try again.")
                elif error == "access_denied":
                    raise AuthenticationError("Authorization denied by user.")
                else:
                    raise AuthenticationError(f"Token exchange failed: {error}")

            except httpx.RequestError as e:
                self.log.warning("Network error during polling", error=str(e))
                # Continue polling on network errors
                continue

        raise AuthenticationError("Authorization timed out. Please try again.")

    async def _refresh_token(self, refresh_token: str) -> TokenData | None:
        """Refresh an access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/api/v1/auth/device/refresh",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return TokenData(
                access_token=data["access_token"],
                refresh_token=refresh_token,  # Keep existing refresh token
                expires_at=datetime.now(UTC) + timedelta(seconds=data["expires_in"]),
                scope=data.get("scope", self.scope),
            )

    def logout(self) -> None:
        """Clear authentication and cached tokens."""
        clear_tokens()
        self._token_data = None
        self.log.info("Logged out successfully")


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


# =============================================================================
# Authenticated HTTP Client
# =============================================================================

class AuthenticatedClient:
    """HTTP client with automatic authentication."""

    def __init__(
        self,
        api_url: str = None,
        authenticator: MCPAuthenticator = None,
    ):
        """Initialize authenticated client.

        Args:
            api_url: Argus API base URL
            authenticator: MCPAuthenticator instance (creates one if not provided)
        """
        self.api_url = api_url or os.getenv("ARGUS_API_URL", DEFAULT_API_URL)
        self.auth = authenticator or MCPAuthenticator(api_url=self.api_url)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(base_url=self.api_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        token = await self.auth.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Make authenticated GET request."""
        headers = await self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        return await self._client.get(path, headers=headers, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        """Make authenticated POST request."""
        headers = await self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        return await self._client.post(path, headers=headers, **kwargs)

    async def put(self, path: str, **kwargs) -> httpx.Response:
        """Make authenticated PUT request."""
        headers = await self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        return await self._client.put(path, headers=headers, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        """Make authenticated DELETE request."""
        headers = await self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        return await self._client.delete(path, headers=headers, **kwargs)


# =============================================================================
# CLI Commands
# =============================================================================

async def login_command():
    """CLI command to authenticate with Argus."""
    auth = MCPAuthenticator()
    try:
        await auth.get_token(force_refresh=True)
        print("\n✅ Successfully authenticated with Argus!")
        print(f"Token cached at: {TOKEN_CACHE_FILE}")
    except AuthenticationError as e:
        print(f"\n❌ Authentication failed: {e}")
        sys.exit(1)


async def logout_command():
    """CLI command to clear authentication."""
    auth = MCPAuthenticator()
    auth.logout()
    print("✅ Logged out successfully")


async def status_command():
    """CLI command to check authentication status."""
    cached = load_cached_tokens()
    if cached:
        print("✅ Authenticated")
        print(f"  User ID: {cached.user_id or 'N/A'}")
        print(f"  Organization: {cached.organization_id or 'N/A'}")
        print(f"  Scope: {cached.scope}")
        print(f"  Expires: {cached.expires_at.isoformat()}")
    else:
        print("❌ Not authenticated")
        print("  Run: python -m src.mcp.auth login")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Argus MCP Authentication")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("login", help="Authenticate with Argus")
    subparsers.add_parser("logout", help="Clear authentication")
    subparsers.add_parser("status", help="Check authentication status")

    args = parser.parse_args()

    if args.command == "login":
        asyncio.run(login_command())
    elif args.command == "logout":
        asyncio.run(logout_command())
    elif args.command == "status":
        asyncio.run(status_command())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

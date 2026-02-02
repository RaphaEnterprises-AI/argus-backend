"""Security headers middleware for OWASP compliance.

Implements security headers recommended by:
- OWASP Secure Headers Project
- Mozilla Observatory
- Security Headers (securityheaders.com)
"""

from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Headers implemented:
    - Content-Security-Policy (CSP)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Strict-Transport-Security (HSTS)
    - Referrer-Policy
    - Permissions-Policy
    - Cache-Control
    - X-Permitted-Cross-Domain-Policies

    CSP Behavior by Environment:
    - production: Strict CSP without unsafe-inline/unsafe-eval
    - staging: Same as production (test CSP before prod)
    - development: Permissive CSP with unsafe-inline/unsafe-eval for dev tools
    """

    def __init__(
        self,
        app: ASGIApp,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,  # 1 year
        enable_csp: bool = True,
        csp_report_uri: str | None = None,
        allowed_hosts: list[str] | None = None,
        frame_options: str = "DENY",
        # CSP customization (RAP-335)
        environment: str = "development",
        csp_extra_script_sources: list[str] | None = None,
        csp_extra_style_sources: list[str] | None = None,
        csp_extra_connect_sources: list[str] | None = None,
    ):
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.enable_csp = enable_csp
        self.csp_report_uri = csp_report_uri
        self.allowed_hosts = allowed_hosts or []
        self.frame_options = frame_options
        # CSP customization
        self.environment = environment.lower()
        self.csp_extra_script_sources = csp_extra_script_sources or []
        self.csp_extra_style_sources = csp_extra_style_sources or []
        self.csp_extra_connect_sources = csp_extra_connect_sources or []

    def _is_production_csp(self) -> bool:
        """Check if strict (production) CSP should be used."""
        return self.environment in ("production", "staging")

    def _build_csp(self) -> str:
        """Build Content-Security-Policy header.

        In production/staging: Strict CSP without unsafe directives
        In development: Permissive CSP for dev tools compatibility
        """
        # Base script-src depends on environment
        if self._is_production_csp():
            # Production: No unsafe-inline or unsafe-eval
            # Applications should use nonces or hashes for inline scripts
            script_src_parts = ["'self'"]
        else:
            # Development: Allow inline scripts for dev tools, hot reload, etc.
            script_src_parts = ["'self'", "'unsafe-inline'", "'unsafe-eval'"]

        # Add any extra script sources
        script_src_parts.extend(self.csp_extra_script_sources)
        script_src = f"script-src {' '.join(script_src_parts)}"

        # Base style-src depends on environment
        if self._is_production_csp():
            # Production: No unsafe-inline for styles
            # Use external stylesheets or style hashes
            style_src_parts = ["'self'"]
        else:
            # Development: Allow inline styles for dev tools
            style_src_parts = ["'self'", "'unsafe-inline'"]

        # Add any extra style sources
        style_src_parts.extend(self.csp_extra_style_sources)
        style_src = f"style-src {' '.join(style_src_parts)}"

        # Base connect-src
        connect_src_parts = ["'self'", "https:"]
        connect_src_parts.extend(self.csp_extra_connect_sources)
        connect_src = f"connect-src {' '.join(connect_src_parts)}"

        directives = [
            "default-src 'self'",
            script_src,
            style_src,
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            connect_src,
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests",
        ]

        if self.csp_report_uri:
            directives.append(f"report-uri {self.csp_report_uri}")

        return "; ".join(directives)

    def _build_permissions_policy(self) -> str:
        """Build Permissions-Policy header."""
        policies = [
            "accelerometer=()",
            "ambient-light-sensor=()",
            "autoplay=()",
            "battery=()",
            "camera=()",
            "cross-origin-isolated=()",
            "display-capture=()",
            "document-domain=()",
            "encrypted-media=()",
            "execution-while-not-rendered=()",
            "execution-while-out-of-viewport=()",
            "fullscreen=(self)",
            "geolocation=()",
            "gyroscope=()",
            "keyboard-map=()",
            "magnetometer=()",
            "microphone=()",
            "midi=()",
            "navigation-override=()",
            "payment=()",
            "picture-in-picture=()",
            "publickey-credentials-get=()",
            "screen-wake-lock=()",
            "sync-xhr=(self)",
            "usb=()",
            "web-share=()",
            "xr-spatial-tracking=()",
        ]
        return ", ".join(policies)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Always set these headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = self.frame_options
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-DNS-Prefetch-Control"] = "off"

        # Permissions Policy
        response.headers["Permissions-Policy"] = self._build_permissions_policy()

        # HSTS (only for HTTPS)
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )

        # Content-Security-Policy
        if self.enable_csp:
            response.headers["Content-Security-Policy"] = self._build_csp()

        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security controls.

    More restrictive than default CORS middleware for production use.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list[str] = None,
        allowed_methods: list[str] = None,
        allowed_headers: list[str] = None,
        allow_credentials: bool = True,
        max_age: int = 600,
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or []
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allowed_headers = allowed_headers or [
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "Accept",
            "Accept-Language",
            "Origin",
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not self.allowed_origins:
            return False
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get("origin")

        # Handle preflight requests
        if request.method == "OPTIONS":
            if origin and self._is_origin_allowed(origin):
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
                        "Access-Control-Allow-Headers": ", ".join(self.allowed_headers),
                        "Access-Control-Allow-Credentials": str(self.allow_credentials).lower(),
                        "Access-Control-Max-Age": str(self.max_age),
                    },
                )
            return Response(status_code=403)

        response = await call_next(request)

        # Add CORS headers to response
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()
            response.headers["Access-Control-Expose-Headers"] = (
                "X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset"
            )

        return response

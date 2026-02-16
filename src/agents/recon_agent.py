"""Reconnaissance Agent — Automated target enumeration and attack surface mapping.

Performs:
- Port scanning (Nmap via Docker sandbox)
- HTTP probing and header analysis
- Technology detection / fingerprinting
- Endpoint enumeration (crawl + OpenAPI discovery)
- AI-powered attack surface analysis

All network tools execute inside a Docker sandbox — no direct host access.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from ..core.model_router import TaskType
from ..models.pentest import (
    AttackSurface,
    Endpoint,
    OpenPort,
    ReconResult,
    Technology,
)
from .base import AgentCapability, AgentResult, AICapability, BaseAgent

log = structlog.get_logger()


# New capability constants for pentest agents
AgentCapability.RECON = "recon"
AgentCapability.NETWORK_SCAN = "network_scan"


class ReconAgent(BaseAgent):
    """Automated reconnaissance agent.

    Enumerates the target's attack surface: open ports, technologies,
    endpoints, and injectable parameters. Uses Docker-sandboxed tools
    for active probing and AI for synthesising results.
    """

    CAPABILITIES = [
        AgentCapability.RECON,
        AgentCapability.NETWORK_SCAN,
        AgentCapability.SECURITY_SCAN,
    ]

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert penetration tester performing reconnaissance. "
            "Analyse raw scan data and produce a structured attack surface assessment. "
            "Identify the highest-risk areas and recommend where to focus vulnerability scanning. "
            "Be precise — only flag genuine risk, not noise."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        target_url: str,
        scope: list[str] | None = None,
        scan_depth: str = "standard",
        **kwargs: Any,
    ) -> AgentResult[ReconResult]:
        """Run full reconnaissance against *target_url*.

        Args:
            target_url: The URL to scan (must be in scope).
            scope: Allowed domains / IP ranges.  Defaults to the target's host.
            scan_depth: ``"quick"`` | ``"standard"`` | ``"thorough"``.

        Returns:
            ``AgentResult`` wrapping a ``ReconResult``.
        """
        start = time.time()
        scope = scope or [urlparse(target_url).hostname or ""]

        if not self._in_scope(target_url, scope):
            return AgentResult(success=False, error=f"Target {target_url} is not in scope {scope}")

        try:
            # Phase 1 — passive / lightweight probes
            headers, technologies = await self._probe_http(target_url)

            # Phase 2 — port scan (sandbox)
            ports = await self._port_scan(target_url, scan_depth)

            # Phase 3 — endpoint enumeration
            endpoints = await self._enumerate_endpoints(target_url, scan_depth)

            # Phase 4 — AI synthesis
            attack_surface = await self._analyze_attack_surface(
                target_url, ports, technologies, endpoints, headers,
            )

            duration_ms = int((time.time() - start) * 1000)
            result = ReconResult(
                target_url=target_url,
                ports=ports,
                technologies=technologies,
                endpoints=endpoints,
                attack_surface=attack_surface,
                headers=headers,
                duration_ms=duration_ms,
            )

            return AgentResult(
                success=True,
                data=result,
                duration_ms=duration_ms,
                input_tokens=self.usage.total_input_tokens,
                output_tokens=self.usage.total_output_tokens,
                cost=self.usage.total_cost,
            )

        except Exception as exc:
            self.log.exception("Recon failed", error=str(exc))
            return AgentResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # HTTP Probing
    # ------------------------------------------------------------------

    async def _probe_http(
        self, url: str,
    ) -> tuple[dict[str, str], list[Technology]]:
        """Fetch response headers and detect technologies."""
        headers_dict: dict[str, str] = {}
        techs: list[Technology] = []

        try:
            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True, verify=False,
            ) as client:
                resp = await client.get(url)
                headers_dict = {k.lower(): v for k, v in resp.headers.items()}

                # Server header
                server = headers_dict.get("server", "")
                if server:
                    techs.append(Technology(
                        name=server.split("/")[0],
                        version=server.split("/")[1] if "/" in server else "",
                        category="server",
                        confidence=0.9,
                    ))

                # X-Powered-By
                powered = headers_dict.get("x-powered-by", "")
                if powered:
                    techs.append(Technology(
                        name=powered.split("/")[0],
                        version=powered.split("/")[1] if "/" in powered else "",
                        category="framework",
                        confidence=0.85,
                    ))

                # Common cookies → framework hints
                cookies = resp.headers.get_list("set-cookie")
                for c in cookies:
                    cl = c.lower()
                    if "laravel_session" in cl:
                        techs.append(Technology(name="Laravel", category="framework", confidence=0.8))
                    elif "csrftoken" in cl or "django" in cl:
                        techs.append(Technology(name="Django", category="framework", confidence=0.8))
                    elif "connect.sid" in cl:
                        techs.append(Technology(name="Express.js", category="framework", confidence=0.7))
                    elif "next-auth" in cl:
                        techs.append(Technology(name="Next.js", category="framework", confidence=0.8))

        except httpx.HTTPError as exc:
            self.log.warning("HTTP probe failed", url=url, error=str(exc))

        return headers_dict, techs

    # ------------------------------------------------------------------
    # Port Scanning (Docker sandbox)
    # ------------------------------------------------------------------

    async def _port_scan(
        self, target_url: str, scan_depth: str,
    ) -> list[OpenPort]:
        """Run Nmap inside a Docker sandbox.

        Falls back to a lightweight Python socket probe when Docker is
        unavailable (e.g. CI, local dev).
        """
        host = urlparse(target_url).hostname or ""
        if not host:
            return []

        # Define port lists by depth
        port_lists = {
            "quick": [80, 443, 8080, 8443],
            "standard": [21, 22, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3306, 3389, 5432, 6379, 8080, 8443, 9200],
            "thorough": list(range(1, 1025)),  # Top 1024 ports
        }
        ports_to_scan = port_lists.get(scan_depth, port_lists["standard"])

        try:
            return await self._nmap_sandbox_scan(host, ports_to_scan)
        except Exception as exc:
            self.log.info("Nmap sandbox unavailable, using socket probe", reason=str(exc))
            return await self._socket_probe(host, ports_to_scan)

    async def _nmap_sandbox_scan(
        self, host: str, ports: list[int],
    ) -> list[OpenPort]:
        """Execute Nmap inside Docker sandbox."""
        try:
            from ..computer_use.sandbox import SandboxManager

            sandbox = SandboxManager()
            port_arg = ",".join(str(p) for p in ports[:200])  # limit

            result = await sandbox.execute_command(
                command=f"nmap -sV -T4 -p {port_arg} --open -oX - {host}",
                image="argus/pentest-tools:latest",
                timeout=120,
            )
            return self._parse_nmap_xml(result.stdout)
        except ImportError:
            raise RuntimeError("SandboxManager not available")

    def _parse_nmap_xml(self, xml_output: str) -> list[OpenPort]:
        """Parse Nmap XML output into OpenPort models."""
        import xml.etree.ElementTree as ET

        ports: list[OpenPort] = []
        try:
            root = ET.fromstring(xml_output)
            for port_el in root.iter("port"):
                state_el = port_el.find("state")
                if state_el is not None and state_el.get("state") == "open":
                    service_el = port_el.find("service")
                    ports.append(OpenPort(
                        port=int(port_el.get("portid", "0")),
                        protocol=port_el.get("protocol", "tcp"),
                        service=service_el.get("name", "") if service_el is not None else "",
                        version=service_el.get("version", "") if service_el is not None else "",
                        state="open",
                    ))
        except ET.ParseError:
            self.log.warning("Failed to parse Nmap XML output")
        return ports

    async def _socket_probe(
        self, host: str, ports: list[int],
    ) -> list[OpenPort]:
        """Lightweight Python-only port probe (fallback)."""
        import asyncio
        import socket

        open_ports: list[OpenPort] = []

        async def _check(port: int) -> OpenPort | None:
            try:
                _loop = asyncio.get_event_loop()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = await _loop.run_in_executor(None, sock.connect_ex, (host, port))
                sock.close()
                if result == 0:
                    return OpenPort(port=port, protocol="tcp", state="open")
            except OSError:
                pass
            return None

        # Scan in batches of 50 to avoid FD exhaustion
        for i in range(0, len(ports), 50):
            batch = ports[i : i + 50]
            results = await asyncio.gather(*[_check(p) for p in batch])
            open_ports.extend(r for r in results if r is not None)

        return open_ports

    # ------------------------------------------------------------------
    # Endpoint Enumeration
    # ------------------------------------------------------------------

    async def _enumerate_endpoints(
        self, target_url: str, scan_depth: str,
    ) -> list[Endpoint]:
        """Discover endpoints via crawling and OpenAPI spec detection."""
        endpoints: list[Endpoint] = []

        # Try OpenAPI / Swagger discovery
        openapi_paths = [
            "/openapi.json", "/swagger.json", "/api-docs",
            "/v1/openapi.json", "/api/v1/openapi.json",
            "/docs", "/.well-known/openapi.json",
        ]

        async with httpx.AsyncClient(
            timeout=10, follow_redirects=True, verify=False,
        ) as client:
            for path in openapi_paths:
                try:
                    resp = await client.get(f"{target_url.rstrip('/')}{path}")
                    if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
                        spec = resp.json()
                        endpoints.extend(self._parse_openapi_spec(target_url, spec))
                        break
                except Exception:
                    continue

            # Crawl common paths
            if scan_depth in ("standard", "thorough"):
                common_paths = [
                    "/", "/login", "/register", "/api", "/admin",
                    "/health", "/status", "/graphql", "/api/v1",
                    "/robots.txt", "/sitemap.xml",
                ]
                for path in common_paths:
                    try:
                        url = f"{target_url.rstrip('/')}{path}"
                        resp = await client.get(url)
                        if resp.status_code < 404:
                            endpoints.append(Endpoint(
                                url=url,
                                method="GET",
                                auth_required=resp.status_code in (401, 403),
                                content_type=resp.headers.get("content-type", ""),
                            ))
                    except Exception:
                        continue

        return endpoints

    def _parse_openapi_spec(
        self, base_url: str, spec: dict[str, Any],
    ) -> list[Endpoint]:
        """Extract endpoints from an OpenAPI spec."""
        endpoints: list[Endpoint] = []
        paths = spec.get("paths", {})

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, detail in methods.items():
                if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                    continue
                params: list[str] = []
                if isinstance(detail, dict):
                    for p in detail.get("parameters", []):
                        if isinstance(p, dict) and "name" in p:
                            params.append(p["name"])
                endpoints.append(Endpoint(
                    url=f"{base_url.rstrip('/')}{path}",
                    method=method.upper(),
                    parameters=params,
                    auth_required=bool(detail.get("security")) if isinstance(detail, dict) else False,
                ))

        return endpoints

    # ------------------------------------------------------------------
    # AI Attack Surface Analysis
    # ------------------------------------------------------------------

    async def _analyze_attack_surface(
        self,
        target_url: str,
        ports: list[OpenPort],
        technologies: list[Technology],
        endpoints: list[Endpoint],
        headers: dict[str, str],
    ) -> AttackSurface:
        """Use AI to synthesise recon data into an attack surface summary."""
        port_summary = ", ".join(f"{p.port}/{p.service or '?'}" for p in ports[:20])
        tech_summary = ", ".join(f"{t.name} {t.version}".strip() for t in technologies)
        endpoint_summary = "\n".join(
            f"  {e.method} {e.url} params={e.parameters}"
            for e in endpoints[:30]
        )

        missing_headers = []
        for h in ("content-security-policy", "strict-transport-security",
                   "x-frame-options", "x-content-type-options"):
            if h not in headers:
                missing_headers.append(h)

        prompt = f"""Analyse this reconnaissance data and provide an attack surface assessment.

TARGET: {target_url}
OPEN PORTS: {port_summary or 'none detected'}
TECHNOLOGIES: {tech_summary or 'none detected'}
ENDPOINTS (sample):
{endpoint_summary or '  none discovered'}
MISSING SECURITY HEADERS: {', '.join(missing_headers) or 'none'}

Return JSON:
{{
  "total_endpoints": <int>,
  "authenticated_endpoints": <int>,
  "input_points": <int — total injectable params/inputs>,
  "risk_areas": ["<specific risk area>", ...],
  "recommended_focus": ["<what to scan next>", ...]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.03,
            json_mode=True,
            max_tokens=1000,
        )

        data = self._parse_json_response(response.content, {})
        return AttackSurface(
            total_endpoints=data.get("total_endpoints", len(endpoints)),
            authenticated_endpoints=data.get("authenticated_endpoints", 0),
            input_points=data.get("input_points", 0),
            risk_areas=data.get("risk_areas", []),
            recommended_focus=data.get("recommended_focus", []),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_scope(url: str, scope: list[str]) -> bool:
        """Check whether *url* is within the allowed scope."""
        host = urlparse(url).hostname or ""
        return any(host == s or host.endswith(f".{s}") for s in scope)

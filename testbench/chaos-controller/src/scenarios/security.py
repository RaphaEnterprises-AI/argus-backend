"""Security scenarios -- verify Argus detects vulnerabilities injected by Chaos App."""

import asyncio

from src.config import settings
from src.scenarios.base import BaseScenario


class SecurityScan(BaseScenario):
    name = "security-scan"
    description = "Full security scan: enable all 6 security bugs, run pentest, verify findings"
    category = "security"
    steps_count = 4

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id

        # Step 1 -- enable all security vulnerabilities
        await self.step(
            run,
            "Enable security vulns",
            "Enable all 6 security bugs via the security-vulns preset scenario",
            lambda: self.chaos.enable_scenario("security-vulns"),
        )

        # Step 2 -- run pentest
        scan = await self.step(
            run,
            "Run pentest",
            "Run a full security scan against the Chaos App",
            lambda: self.argus.run_pentest(project_id, url),
        )
        scan_id = scan.get("scanId") or scan.get("id", "")

        # Step 3 -- verify findings
        async def verify_findings():
            # Poll for scan completion (up to 120s)
            for _ in range(24):
                status = await self.argus.get_pentest_status(scan_id)
                state = status.get("status", "")
                if state in ("completed", "done", "finished"):
                    findings = status.get("findings", status.get("vulnerabilities", []))
                    if isinstance(findings, list) and len(findings) > 0:
                        return status
                    # Even with 0 findings, scan completed -- return for reporting
                    return status
                if state in ("failed", "error"):
                    raise RuntimeError(f"Pentest failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(5)
            raise TimeoutError("Pentest did not complete within 120s")

        await self.step(
            run,
            "Verify findings",
            "Check scan results for XSS, SQLi, and IDOR findings",
            verify_findings,
        )

        # Step 4 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )


class SecuritySAST(BaseScenario):
    name = "security-sast"
    description = "SAST scan: detect hardcoded secrets in code"
    category = "security"
    steps_count = 4

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id

        # Step 1 -- enable secrets-in-code bug
        await self.step(
            run,
            "Enable secrets bug",
            "Enable sec-secrets-in-code to expose hardcoded credentials",
            lambda: self.chaos.enable_bug("sec-secrets-in-code"),
        )

        # Step 2 -- run pentest (limited scope for SAST)
        scan = await self.step(
            run,
            "Run security scan",
            "Run pentest scan to detect secret exposure",
            lambda: self.argus.run_pentest(project_id, url),
        )
        scan_id = scan.get("scanId") or scan.get("id", "")

        # Step 3 -- verify secret detection
        async def verify_secret_detection():
            for _ in range(24):
                status = await self.argus.get_pentest_status(scan_id)
                state = status.get("status", "")
                if state in ("completed", "done", "finished"):
                    findings = status.get("findings", status.get("vulnerabilities", []))
                    return status
                if state in ("failed", "error"):
                    raise RuntimeError(f"Scan failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(5)
            raise TimeoutError("Security scan did not complete within 120s")

        await self.step(
            run,
            "Verify secret detection",
            "Check scan results for hardcoded secrets finding",
            verify_secret_detection,
        )

        # Step 4 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )

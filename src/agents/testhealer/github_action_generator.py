"""GitHub Action YAML generator for healing CI integration.

Generates GitHub Action workflow YAML that:
1. Runs tests on PR events
2. On failure, calls TestHealerCICDAgent CI callback
3. Posts fix suggestions as PR comments
"""

import structlog
from typing import Any

logger = structlog.get_logger()


class GitHubActionGenerator:
    """Generates GitHub Action YAML for automated test healing."""

    def __init__(self, api_base_url: str = "https://argus-brain-production.up.railway.app"):
        self.api_base_url = api_base_url

    def generate_healing_workflow(
        self,
        project_id: str,
        test_command: str = "npm test",
        test_framework: str = "playwright",
        node_version: str = "20",
        python_version: str = "3.12",
        branches: list[str] | None = None,
    ) -> str:
        """Generate a complete GitHub Action workflow for test healing.

        Args:
            project_id: Argus project ID
            test_command: Command to run tests
            test_framework: Test framework (playwright, cypress, pytest, etc.)
            node_version: Node.js version for JS frameworks
            python_version: Python version for Python frameworks
            branches: Branches to trigger on (default: main, develop)

        Returns:
            Complete GitHub Action YAML as string
        """
        branches = branches or ["main", "develop"]
        branches_yaml = "\n".join(f"        - {b}" for b in branches)

        is_python = test_framework in ("pytest", "selenium", "robot")

        setup_steps = self._get_setup_steps(test_framework, node_version, python_version, is_python)
        test_step = self._get_test_step(test_command, test_framework)

        return f"""name: Argus Test Healing

on:
  pull_request:
    branches:
{branches_yaml}
    types: [opened, synchronize, reopened]
  push:
    branches:
{branches_yaml}

permissions:
  contents: read
  pull-requests: write
  checks: write

jobs:
  test-and-heal:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout
        uses: actions/checkout@v4

{setup_steps}

{test_step}

      - name: Report failures to Argus
        if: failure()
        env:
          ARGUS_API_KEY: ${{{{ secrets.ARGUS_API_KEY }}}}
          ARGUS_ORG_ID: ${{{{ secrets.ARGUS_ORG_ID }}}}
        run: |
          curl -X POST "{self.api_base_url}/api/v1/healing-cicd/${{ARGUS_ORG_ID}}/{project_id}/ci-callback" \\
            -H "X-API-Key: ${{ARGUS_API_KEY}}" \\
            -H "X-Organization-Id: ${{ARGUS_ORG_ID}}" \\
            -H "Content-Type: application/json" \\
            -d '{{
              "run_url": "${{{{ github.server_url }}}}/${{{{ github.repository }}}}/actions/runs/${{{{ github.run_id }}}}",
              "commit_sha": "${{{{ github.sha }}}}",
              "branch": "${{{{ github.head_ref || github.ref_name }}}}",
              "pr_number": "${{{{ github.event.pull_request.number || 0 }}}}",
              "test_framework": "{test_framework}",
              "test_results_path": "test-results/"
            }}'
"""

    def _get_setup_steps(self, framework: str, node_version: str, python_version: str, is_python: bool) -> str:
        if is_python:
            return f"""      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '{python_version}'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt"""
        else:
            extra = ""
            if framework == "playwright":
                extra = """
      - name: Install Playwright browsers
        run: npx playwright install --with-deps"""
            elif framework == "cypress":
                extra = """
      - name: Install Cypress
        run: npx cypress install"""

            return f"""      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '{node_version}'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci{extra}"""

    def _get_test_step(self, test_command: str, framework: str) -> str:
        artifacts = ""
        if framework == "playwright":
            artifacts = """
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results/
          retention-days: 7"""
        elif framework == "cypress":
            artifacts = """
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: |
            cypress/screenshots/
            cypress/videos/
          retention-days: 7"""

        return f"""      - name: Run tests
        id: tests
        continue-on-error: true
        run: {test_command}{artifacts}"""

    def generate_scheduled_scan_workflow(
        self,
        project_id: str,
        cron: str = "0 2 * * *",  # Daily at 2 AM
    ) -> str:
        """Generate a scheduled proactive scan workflow."""
        return f"""name: Argus Proactive Healing Scan

on:
  schedule:
    - cron: '{cron}'
  workflow_dispatch:

jobs:
  proactive-scan:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Trigger Proactive Scan
        env:
          ARGUS_API_KEY: ${{{{ secrets.ARGUS_API_KEY }}}}
          ARGUS_ORG_ID: ${{{{ secrets.ARGUS_ORG_ID }}}}
        run: |
          curl -X POST "{self.api_base_url}/api/v1/healing-cicd/${{ARGUS_ORG_ID}}/{project_id}/scan" \\
            -H "X-API-Key: ${{ARGUS_API_KEY}}" \\
            -H "X-Organization-Id: ${{ARGUS_ORG_ID}}" \\
            -H "Content-Type: application/json" \\
            -d '{{
              "scan_type": "proactive",
              "trigger_context": {{
                "source": "github_action_schedule",
                "cron": "{cron}"
              }}
            }}'
"""

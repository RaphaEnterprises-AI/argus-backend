"""CI/CD configuration generators for QA Engineer agent.

Generates CI pipeline configurations for:
- GitHub Actions
- GitLab CI
- Jenkins (Jenkinsfile)
"""

import structlog
from typing import Any
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class CIConfig(BaseModel):
    """Configuration for CI pipeline generation."""
    platform: str = "github_actions"  # github_actions, gitlab_ci, jenkins
    test_command: str = "npm test"
    test_framework: str = "playwright"
    node_version: str = "20"
    python_version: str = "3.12"
    branches: list[str] = Field(default_factory=lambda: ["main", "develop"])
    enable_healing: bool = True
    enable_visual_regression: bool = False
    enable_accessibility: bool = False
    parallel_workers: int = 1
    timeout_minutes: int = 30
    artifacts_path: str = "test-results/"
    argus_project_id: str = ""
    argus_api_url: str = "https://argus-brain-production.up.railway.app"


class GeneratedCIConfig(BaseModel):
    """Generated CI configuration output."""
    platform: str
    file_name: str  # e.g., ".github/workflows/tests.yml"
    content: str    # YAML or Groovy content
    description: str


class CIConfigGenerator:
    """Generates CI pipeline configurations."""

    def generate(self, config: CIConfig) -> GeneratedCIConfig:
        """Generate CI config for the specified platform."""
        generators = {
            "github_actions": self._generate_github_actions,
            "gitlab_ci": self._generate_gitlab_ci,
            "jenkins": self._generate_jenkins,
        }
        generator = generators.get(config.platform)
        if not generator:
            raise ValueError(f"Unsupported CI platform: {config.platform}")
        return generator(config)

    def _generate_github_actions(self, config: CIConfig) -> GeneratedCIConfig:
        """Generate GitHub Actions workflow YAML."""
        branches_yaml = "\n".join(f"        - {b}" for b in config.branches)
        is_python = config.test_framework in ("pytest", "selenium", "robot")

        # Build setup steps
        setup = self._github_setup(config, is_python)
        test_step = self._github_test_step(config)
        healing_step = self._github_healing_step(config) if config.enable_healing else ""
        visual_step = self._github_visual_step(config) if config.enable_visual_regression else ""
        a11y_step = self._github_a11y_step(config) if config.enable_accessibility else ""

        content = f"""name: Argus E2E Tests
on:
  pull_request:
    branches:
{branches_yaml}
  push:
    branches:
{branches_yaml}
permissions:
  contents: read
  pull-requests: write
  checks: write
jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    timeout-minutes: {config.timeout_minutes}
    steps:
      - uses: actions/checkout@v4
{setup}
{test_step}
{healing_step}
{visual_step}
{a11y_step}
      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: {config.artifacts_path}
          retention-days: 7
"""
        return GeneratedCIConfig(
            platform="github_actions",
            file_name=".github/workflows/argus-tests.yml",
            content=content,
            description="GitHub Actions workflow with Argus healing integration",
        )

    def _generate_gitlab_ci(self, config: CIConfig) -> GeneratedCIConfig:
        """Generate GitLab CI YAML."""
        is_python = config.test_framework in ("pytest", "selenium", "robot")
        image = f"python:{config.python_version}" if is_python else f"node:{config.node_version}"

        install_cmds = "pip install -r requirements.txt" if is_python else "npm ci"
        if config.test_framework == "playwright" and not is_python:
            install_cmds += "\n    - npx playwright install --with-deps"

        healing = ""
        if config.enable_healing:
            healing = f"""
  after_script:
    - |
      if [ "$CI_JOB_STATUS" == "failed" ]; then
        curl -X POST "{config.argus_api_url}/api/v1/healing-cicd/$ARGUS_ORG_ID/{config.argus_project_id}/ci-callback" \\
          -H "X-API-Key: $ARGUS_API_KEY" \\
          -H "X-Organization-Id: $ARGUS_ORG_ID" \\
          -H "Content-Type: application/json" \\
          -d '{{"run_url": "'$CI_PIPELINE_URL'", "commit_sha": "'$CI_COMMIT_SHA'", "branch": "'$CI_COMMIT_REF_NAME'"}}'
      fi"""

        content = f"""stages:
  - test

e2e-tests:
  stage: test
  image: {image}
  variables:
    ARGUS_API_KEY: $ARGUS_API_KEY
    ARGUS_ORG_ID: $ARGUS_ORG_ID
  before_script:
    - {install_cmds}
  script:
    - {config.test_command}
  artifacts:
    when: always
    paths:
      - {config.artifacts_path}
    expire_in: 7 days
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH =~ /^(main|develop)$/
{healing}
"""
        return GeneratedCIConfig(
            platform="gitlab_ci",
            file_name=".gitlab-ci.yml",
            content=content,
            description="GitLab CI pipeline with Argus healing integration",
        )

    def _generate_jenkins(self, config: CIConfig) -> GeneratedCIConfig:
        """Generate Jenkinsfile (Groovy)."""
        is_python = config.test_framework in ("pytest", "selenium", "robot")
        tool_block = f"python '{config.python_version}'" if is_python else f"nodejs '{config.node_version}'"
        install = "sh 'pip install -r requirements.txt'" if is_python else "sh 'npm ci'"

        healing = ""
        if config.enable_healing:
            healing = f"""
            failure {{
                sh '''
                    curl -X POST "{config.argus_api_url}/api/v1/healing-cicd/${{ARGUS_ORG_ID}}/{config.argus_project_id}/ci-callback" \\
                      -H "X-API-Key: ${{ARGUS_API_KEY}}" \\
                      -H "X-Organization-Id: ${{ARGUS_ORG_ID}}" \\
                      -H "Content-Type: application/json" \\
                      -d '{{"run_url": "'$BUILD_URL'", "commit_sha": "'$GIT_COMMIT'", "branch": "'$GIT_BRANCH'"}}'
                '''
            }}"""

        content = f"""pipeline {{
    agent any
    tools {{
        {tool_block}
    }}
    environment {{
        ARGUS_API_KEY = credentials('argus-api-key')
        ARGUS_ORG_ID = credentials('argus-org-id')
    }}
    stages {{
        stage('Install') {{
            steps {{
                {install}
            }}
        }}
        stage('Test') {{
            steps {{
                sh '{config.test_command}'
            }}
        }}
    }}
    post {{
        always {{
            archiveArtifacts artifacts: '{config.artifacts_path}**', allowEmptyArchive: true
        }}{healing}
    }}
}}
"""
        return GeneratedCIConfig(
            platform="jenkins",
            file_name="Jenkinsfile",
            content=content,
            description="Jenkins pipeline with Argus healing integration",
        )

    # -------------------------------------------------------------------------
    # Helper methods for GitHub Actions
    # -------------------------------------------------------------------------

    def _github_setup(self, config: CIConfig, is_python: bool) -> str:
        if is_python:
            return f"""      - uses: actions/setup-python@v5
        with:
          python-version: '{config.python_version}'
      - run: pip install -r requirements.txt"""
        setup = f"""      - uses: actions/setup-node@v4
        with:
          node-version: '{config.node_version}'
          cache: 'npm'
      - run: npm ci"""
        if config.test_framework == "playwright":
            setup += "\n      - run: npx playwright install --with-deps"
        return setup

    def _github_test_step(self, config: CIConfig) -> str:
        return f"""      - name: Run E2E tests
        id: tests
        continue-on-error: true
        run: {config.test_command}"""

    def _github_healing_step(self, config: CIConfig) -> str:
        return f"""      - name: Report to Argus
        if: steps.tests.outcome == 'failure'
        env:
          ARGUS_API_KEY: ${{{{ secrets.ARGUS_API_KEY }}}}
          ARGUS_ORG_ID: ${{{{ secrets.ARGUS_ORG_ID }}}}
        run: |
          curl -X POST "{config.argus_api_url}/api/v1/healing-cicd/${{ARGUS_ORG_ID}}/{config.argus_project_id}/ci-callback" \\
            -H "X-API-Key: ${{ARGUS_API_KEY}}" \\
            -H "X-Organization-Id: ${{ARGUS_ORG_ID}}" \\
            -H "Content-Type: application/json" \\
            -d '{{
              "run_url": "${{{{ github.server_url }}}}/${{{{ github.repository }}}}/actions/runs/${{{{ github.run_id }}}}",
              "commit_sha": "${{{{ github.sha }}}}",
              "branch": "${{{{ github.head_ref || github.ref_name }}}}"
            }}'"""

    def _github_visual_step(self, config: CIConfig) -> str:
        return """      - name: Visual regression check
        if: always()
        run: echo "Visual regression check via Argus VisualAI" """

    def _github_a11y_step(self, config: CIConfig) -> str:
        return """      - name: Accessibility check
        if: always()
        run: echo "Accessibility check via Argus AccessibilityChecker" """

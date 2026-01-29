"""CI/CD integration plugins.

Plugins for continuous integration and deployment platforms:
- GitHub Actions: GitHub's built-in CI/CD
- GitLab CI: GitLab's integrated CI/CD (including self-hosted)
- Jenkins: Open-source automation server
- CircleCI: Cloud-native CI/CD platform
- Azure Pipelines: Azure DevOps CI/CD
- Bitbucket Pipelines: Bitbucket Cloud CI/CD
- TeamCity: JetBrains TeamCity

Each plugin implements the IntegrationPlugin interface for:
- Connection testing
- Data synchronization (pipelines, builds, etc.)
- Webhook support
"""

from src.integrations.plugins.cicd.azure_pipelines_plugin import AzurePipelinesPlugin
from src.integrations.plugins.cicd.bitbucket_pipelines_plugin import BitbucketPipelinesPlugin
from src.integrations.plugins.cicd.circleci_plugin import CircleCIPlugin
from src.integrations.plugins.cicd.github_actions_plugin import GitHubActionsPlugin
from src.integrations.plugins.cicd.gitlab_ci_plugin import GitLabCIPlugin
from src.integrations.plugins.cicd.jenkins_plugin import JenkinsPlugin
from src.integrations.plugins.cicd.teamcity_plugin import TeamCityPlugin

__all__ = [
    "AzurePipelinesPlugin",
    "BitbucketPipelinesPlugin",
    "CircleCIPlugin",
    "GitHubActionsPlugin",
    "GitLabCIPlugin",
    "JenkinsPlugin",
    "TeamCityPlugin",
]

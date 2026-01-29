"""
Testing platform integration plugins.

Provides integrations with cross-browser testing and QA platforms:
- BrowserStack: Cloud-based cross-browser testing
- Sauce Labs: Comprehensive testing platform with tunnels
- LambdaTest: Cross-browser testing with 3000+ browsers
- Percy: Visual regression testing by BrowserStack
- Chromatic: UI visual testing for Storybook
- Playwright: E2E test results import
- Cypress Cloud: Test recording and analytics
"""

from src.integrations.plugins.testing.browserstack_plugin import (
    BrowserStackPlugin,
    BrowserStackBuild,
    BrowserStackSession,
    BrowserInfo,
    BuildStatus as BrowserStackBuildStatus,
    SessionStatus as BrowserStackSessionStatus,
    create_browserstack_plugin,
)
from src.integrations.plugins.testing.saucelabs_plugin import (
    SauceLabsPlugin,
    SauceLabsBuild,
    SauceLabsJob,
    SauceLabsTunnel,
    DataCenter,
    JobStatus as SauceLabsJobStatus,
    TunnelStatus as SauceLabsTunnelStatus,
    create_saucelabs_plugin,
)
from src.integrations.plugins.testing.lambdatest_plugin import (
    LambdaTestPlugin,
    LambdaTestBuild,
    LambdaTestSession,
    LambdaTestScreenshot,
    LambdaTestUser,
    BuildStatus as LambdaTestBuildStatus,
    SessionStatus as LambdaTestSessionStatus,
    create_lambdatest_plugin,
)
from src.integrations.plugins.testing.percy_plugin import PercyPlugin
from src.integrations.plugins.testing.chromatic_plugin import ChromaticPlugin
from src.integrations.plugins.testing.playwright_plugin import PlaywrightPlugin
from src.integrations.plugins.testing.cypress_plugin import CypressPlugin

__all__ = [
    # BrowserStack
    "BrowserStackPlugin",
    "BrowserStackBuild",
    "BrowserStackSession",
    "BrowserInfo",
    "BrowserStackBuildStatus",
    "BrowserStackSessionStatus",
    "create_browserstack_plugin",
    # Sauce Labs
    "SauceLabsPlugin",
    "SauceLabsBuild",
    "SauceLabsJob",
    "SauceLabsTunnel",
    "DataCenter",
    "SauceLabsJobStatus",
    "SauceLabsTunnelStatus",
    "create_saucelabs_plugin",
    # LambdaTest
    "LambdaTestPlugin",
    "LambdaTestBuild",
    "LambdaTestSession",
    "LambdaTestScreenshot",
    "LambdaTestUser",
    "LambdaTestBuildStatus",
    "LambdaTestSessionStatus",
    "create_lambdatest_plugin",
    # Percy (Visual Testing)
    "PercyPlugin",
    # Chromatic (Storybook Visual Testing)
    "ChromaticPlugin",
    # Playwright (Test Results Import)
    "PlaywrightPlugin",
    # Cypress Cloud (Test Recording)
    "CypressPlugin",
]

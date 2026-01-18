"""Multi-Language Test Export Module.

This module exports Argus test specifications to multiple programming
languages and test frameworks.

Supported combinations:
- Python: Playwright, Selenium
- TypeScript: Playwright, Puppeteer, Cypress
- Java: Selenium
- C#: Selenium, Playwright
- Ruby: Capybara, Selenium
- Go: Rod

Example:
    from src.export import ExportEngine, ExportConfig

    # Export a test to Python Playwright
    engine = ExportEngine()
    result = engine.export(
        test_spec=test_spec,
        config=ExportConfig(
            language="python",
            framework="playwright",
        )
    )
    print(result.code)
"""

from .engine import ExportEngine
from .models import ExportConfig, ExportResult, SupportedFramework, SupportedLanguage

__all__ = [
    "ExportConfig",
    "ExportResult",
    "SupportedLanguage",
    "SupportedFramework",
    "ExportEngine",
]

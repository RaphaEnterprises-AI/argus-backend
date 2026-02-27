"""Scenario registry -- maps scenario slug to its class."""

from src.scenarios.flaky import FlakyDetect, FlakyVsReal
from src.scenarios.golden_path import GoldenPath
from src.scenarios.healing import HealCascade, HealProactive, HealSelectorDrift
from src.scenarios.onboarding import OnboardAPI, OnboardConduit, OnboardPlane
from src.scenarios.regression import RegressAPI, RegressPerf, RegressVisual
from src.scenarios.security import SecuritySAST, SecurityScan

SCENARIOS: dict = {
    "onboard-conduit": OnboardConduit,
    "onboard-plane": OnboardPlane,
    "onboard-api": OnboardAPI,
    "heal-selector-drift": HealSelectorDrift,
    "heal-cascade": HealCascade,
    "heal-proactive": HealProactive,
    "regress-visual": RegressVisual,
    "regress-perf": RegressPerf,
    "regress-api": RegressAPI,
    "security-scan": SecurityScan,
    "security-sast": SecuritySAST,
    "flaky-detect": FlakyDetect,
    "flaky-vs-real": FlakyVsReal,
    "golden-path": GoldenPath,
}

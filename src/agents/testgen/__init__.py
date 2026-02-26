"""TestGen sub-package - Test generation orchestration.

Contains the TestGenAgent orchestrator and supporting modules for
requirements parsing and framework-specific code emission.
"""

from .code_emitters import (
    BDDScenario,
    CodeEmitter,
    EmitterConfig,
    GeneratedTestCase,
    TestFramework,
)
from .requirements_parser import (
    AmbiguityDetail,
    AmbiguityType,
    ParsedRequirement,
    RequirementsParser,
    RequirementSource,
)

__all__ = [
    "AmbiguityDetail",
    "AmbiguityType",
    "BDDScenario",
    "CodeEmitter",
    "EmitterConfig",
    "GeneratedTestCase",
    "ParsedRequirement",
    "RequirementsParser",
    "RequirementSource",
    "TestFramework",
]

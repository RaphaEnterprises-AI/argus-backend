"""Specialized Code Analyzers for different repository types.

Each analyzer understands the conventions and patterns of specific
frameworks and domains:

- Frontend: React, Vue, Angular, Svelte components and hooks
- Backend: FastAPI, Express, Django routes and endpoints
- Database: SQL queries, migrations, schema definitions
- Microservices: Kubernetes, Docker, service mesh configs
- Tests: pytest, jest, playwright test patterns

All analysis happens locally - code never leaves.
"""

from .base import BaseAnalyzer, AnalysisResult, ComponentInfo, RouteInfo, QueryInfo
from .frontend import FrontendAnalyzer
from .backend import BackendAnalyzer
from .database import DatabaseAnalyzer
from .microservices import MicroservicesAnalyzer
from .tests import TestAnalyzer

__all__ = [
    "BaseAnalyzer",
    "AnalysisResult",
    "ComponentInfo",
    "RouteInfo",
    "QueryInfo",
    "FrontendAnalyzer",
    "BackendAnalyzer",
    "DatabaseAnalyzer",
    "MicroservicesAnalyzer",
    "TestAnalyzer",
]

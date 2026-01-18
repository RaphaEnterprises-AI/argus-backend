"""Backend Analyzer - FastAPI, Express, Django, Flask route analysis.

Understands:
- API routes and endpoints
- Request/response schemas
- Authentication requirements
- Middleware chains
- Database interactions
"""

import logging
import re

from src.indexer import ParsedFile

from .base import (
    BaseAnalyzer,
    ComponentInfo,
    ComponentType,
    Issue,
    RouteInfo,
    Severity,
)

logger = logging.getLogger(__name__)


class BackendAnalyzer(BaseAnalyzer):
    """Analyzer for backend APIs (FastAPI, Express, Django, Flask)."""

    @property
    def analyzer_type(self) -> str:
        return "backend"

    def get_file_patterns(self) -> list[str]:
        return [
            # Python backends
            "**/routes/**/*.py",
            "**/routers/**/*.py",
            "**/api/**/*.py",
            "**/views/**/*.py",
            "**/endpoints/**/*.py",
            # Node.js backends
            "**/routes/**/*.ts",
            "**/routes/**/*.js",
            "**/controllers/**/*.ts",
            "**/controllers/**/*.js",
            # General patterns
            "**/*_routes.py",
            "**/*_router.py",
            "**/*_api.py",
            "**/*Controller.ts",
            "**/*Router.ts",
        ]

    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a backend file for routes and endpoints."""
        components: list[ComponentInfo] = []

        framework = self._detect_framework(parsed)

        if framework == "fastapi":
            components.extend(self._analyze_fastapi(parsed))
        elif framework == "flask":
            components.extend(self._analyze_flask(parsed))
        elif framework == "django":
            components.extend(self._analyze_django(parsed))
        elif framework == "express":
            components.extend(self._analyze_express(parsed))
        elif framework == "nestjs":
            components.extend(self._analyze_nestjs(parsed))

        return components

    def _detect_framework(self, parsed: ParsedFile) -> str:
        """Detect the backend framework."""
        content = parsed.content

        # Python frameworks
        if "from fastapi" in content or "FastAPI" in content:
            return "fastapi"
        if "from flask" in content or "Flask" in content:
            return "flask"
        if "from django" in content or "from rest_framework" in content:
            return "django"

        # Node.js frameworks
        if "express" in content and ("Router" in content or "app." in content):
            return "express"
        if "@Controller" in content or "@nestjs" in content:
            return "nestjs"

        return "unknown"

    def _analyze_fastapi(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze FastAPI routes."""
        components: list[ComponentInfo] = []
        content = parsed.content
        content.split("\n")

        # Find router instance
        router_name = "router"
        router_match = re.search(r'(\w+)\s*=\s*APIRouter\(', content)
        if router_match:
            router_name = router_match.group(1)

        # Find route decorators
        route_pattern = rf'@(?:{router_name}|app)\.(get|post|put|patch|delete|options|head)\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(route_pattern, content, re.IGNORECASE):
            method = match.group(1).upper()
            path = match.group(2)

            # Find the function definition after the decorator
            decorator_pos = match.end()
            func_match = re.search(
                r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)',
                content[decorator_pos:decorator_pos + 500]
            )

            if not func_match:
                continue

            handler_name = func_match.group(1)
            params_str = func_match.group(2)

            # Calculate line number
            line_number = content[:match.start()].count("\n") + 1

            # Extract path parameters
            path_params = re.findall(r'\{(\w+)\}', path)

            # Extract query parameters from function signature
            query_params = self._extract_query_params(params_str)

            # Check for request body
            body_schema = None
            if "Body" in params_str or method in ["POST", "PUT", "PATCH"]:
                body_match = re.search(r'(\w+)\s*:\s*(\w+)', params_str)
                if body_match:
                    body_schema = body_match.group(2)

            # Check for authentication
            requires_auth = self._check_auth_requirement(content, match.start())

            # Create route info
            route = RouteInfo(
                path=path,
                method=method,
                handler_name=handler_name,
                file_path=parsed.file_path,
                line_number=line_number,
                path_params=path_params,
                query_params=query_params,
                body_schema=body_schema,
                requires_auth=requires_auth,
            )

            # Check for issues
            route.issues.extend(self._check_fastapi_issues(route, content))

            # Convert to ComponentInfo
            component = ComponentInfo(
                name=f"{method} {path}",
                component_type=ComponentType.ROUTE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,  # Approximate
                methods=[handler_name],
                issues=route.issues,
            )
            components.append(component)

        return components

    def _extract_query_params(self, params_str: str) -> list[str]:
        """Extract query parameters from function signature."""
        query_params = []

        # Pattern for FastAPI query params
        for match in re.finditer(r'(\w+)\s*:\s*(?:str|int|float|bool|Query)', params_str):
            param = match.group(1)
            # Skip common non-query params
            if param not in ["self", "request", "db", "session", "current_user"]:
                query_params.append(param)

        return query_params

    def _check_auth_requirement(self, content: str, position: int) -> bool:
        """Check if a route requires authentication."""
        # Look backwards for Depends with auth
        context = content[max(0, position - 200):position]

        auth_patterns = [
            "Depends(get_current_user)",
            "Depends(auth)",
            "Depends(verify_token)",
            "@require_auth",
            "@login_required",
            "Security(",
        ]

        return any(pattern in context for pattern in auth_patterns)

    def _check_fastapi_issues(self, route: RouteInfo, content: str) -> list[Issue]:
        """Check for FastAPI route issues."""
        issues = []

        # No response model
        handler_context = content[content.find(route.handler_name):content.find(route.handler_name) + 500]
        if "response_model" not in handler_context and route.method == "GET":
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Route {route.method} {route.path} lacks response_model",
                file_path=route.file_path,
                line_number=route.line_number,
                suggestion="Add response_model for better API documentation",
                code="FASTAPI_NO_RESPONSE_MODEL",
            ))

        # No error handling
        if "HTTPException" not in handler_context and "raise" not in handler_context:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Route {route.method} {route.path} may lack error handling",
                file_path=route.file_path,
                line_number=route.line_number,
                suggestion="Add proper error handling with HTTPException",
                code="FASTAPI_NO_ERROR_HANDLING",
            ))

        # Sensitive data without auth
        sensitive_paths = ["/admin", "/user", "/account", "/payment", "/settings"]
        if any(s in route.path for s in sensitive_paths) and not route.requires_auth:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Route {route.path} appears sensitive but may lack authentication",
                file_path=route.file_path,
                line_number=route.line_number,
                suggestion="Add authentication dependency",
                code="BACKEND_UNPROTECTED_SENSITIVE_ROUTE",
            ))

        return issues

    def _analyze_flask(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Flask routes."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find route decorators
        route_pattern = r'@(?:app|bp|\w+)\.route\s*\(\s*["\']([^"\']+)["\'](?:[^)]*methods\s*=\s*\[([^\]]+)\])?'

        for match in re.finditer(route_pattern, content):
            path = match.group(1)
            methods_str = match.group(2)

            methods = ["GET"]
            if methods_str:
                methods = [m.strip().strip("'\"").upper() for m in methods_str.split(",")]

            # Find the function
            decorator_pos = match.end()
            func_match = re.search(
                r'def\s+(\w+)\s*\(',
                content[decorator_pos:decorator_pos + 200]
            )

            if not func_match:
                continue

            handler_name = func_match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            for method in methods:
                component = ComponentInfo(
                    name=f"{method} {path}",
                    component_type=ComponentType.ROUTE,
                    file_path=parsed.file_path,
                    start_line=line_number,
                    end_line=line_number + 10,
                    methods=[handler_name],
                )
                components.append(component)

        return components

    def _analyze_django(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Django views and URL patterns."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Check if this is a urls.py file
        if "urlpatterns" in content:
            components.extend(self._analyze_django_urls(parsed))
        else:
            components.extend(self._analyze_django_views(parsed))

        return components

    def _analyze_django_urls(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Django URL patterns."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find path() calls
        path_pattern = r'path\s*\(\s*["\']([^"\']*)["\'],\s*(\w+)'

        for match in re.finditer(path_pattern, content):
            path = match.group(1)
            view_name = match.group(2)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=f"URL /{path}",
                component_type=ComponentType.ROUTE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number,
                methods=[view_name],
            )
            components.append(component)

        return components

    def _analyze_django_views(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Django view functions/classes."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find class-based views
        for match in re.finditer(r'class\s+(\w+View)\s*\(', content):
            name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=name,
                component_type=ComponentType.CONTROLLER,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 20,
            )
            components.append(component)

        # Find function-based views
        for func in parsed.get_functions():
            if "request" in func.text:  # Likely a view
                component = ComponentInfo(
                    name=func.name or "anonymous",
                    component_type=ComponentType.CONTROLLER,
                    file_path=parsed.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                )
                components.append(component)

        return components

    def _analyze_express(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Express.js routes."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find route patterns
        route_pattern = r'(?:router|app)\.(get|post|put|patch|delete|all)\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(route_pattern, content, re.IGNORECASE):
            method = match.group(1).upper()
            path = match.group(2)
            line_number = content[:match.start()].count("\n") + 1

            # Check for auth middleware
            context = content[match.start():match.start() + 300]
            requires_auth = any(auth in context for auth in [
                "authenticate", "isAuth", "requireAuth", "verifyToken", "passport"
            ])

            component = ComponentInfo(
                name=f"{method} {path}",
                component_type=ComponentType.ROUTE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,
            )

            # Check for issues
            if not requires_auth and any(s in path for s in ["/admin", "/user", "/api/private"]):
                component.issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Route {path} may need authentication",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="EXPRESS_UNPROTECTED_ROUTE",
                ))

            components.append(component)

        return components

    def _analyze_nestjs(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze NestJS controllers."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find controller class
        controller_match = re.search(
            r"@Controller\s*\(\s*['\"]?([^'\")\s]*)['\"]?\s*\)",
            content
        )

        if not controller_match:
            return components

        base_path = controller_match.group(1)

        # Find class definition
        class_match = re.search(r'class\s+(\w+)', content)
        if not class_match:
            return components

        class_match.group(1)

        # Find route methods
        method_pattern = r'@(Get|Post|Put|Patch|Delete)\s*\(\s*["\']?([^"\')\s]*)["\']?\s*\)'

        for match in re.finditer(method_pattern, content):
            method = match.group(1).upper()
            path = match.group(2)
            full_path = f"/{base_path}/{path}".replace("//", "/")

            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=f"{method} {full_path}",
                component_type=ComponentType.ROUTE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,
            )
            components.append(component)

        return components

    def analyze(self):
        """Override to also extract routes."""
        result = super().analyze()

        # Extract RouteInfo from components
        routes = []
        for component in result.components:
            if component.component_type == ComponentType.ROUTE:
                # Parse route info from component name
                parts = component.name.split(" ", 1)
                if len(parts) == 2:
                    method, path = parts
                    routes.append(RouteInfo(
                        path=path,
                        method=method,
                        handler_name=component.methods[0] if component.methods else "",
                        file_path=component.file_path,
                        line_number=component.start_line,
                        issues=component.issues,
                    ))

        result.routes = routes
        return result

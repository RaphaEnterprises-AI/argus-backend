"""Microservices Analyzer - Kubernetes, Docker, service mesh analysis.

Understands:
- Kubernetes manifests (Deployments, Services, ConfigMaps)
- Docker/Dockerfile patterns
- Docker Compose configurations
- Helm charts
- Service mesh configs (Istio, Linkerd)
"""

import re
import logging
from pathlib import Path
from typing import Optional
import yaml

from .base import (
    BaseAnalyzer,
    ComponentInfo,
    ComponentType,
    Issue,
    Severity,
)
from src.indexer import ParsedFile

logger = logging.getLogger(__name__)


class MicroservicesAnalyzer(BaseAnalyzer):
    """Analyzer for microservices infrastructure configs."""

    @property
    def analyzer_type(self) -> str:
        return "microservices"

    def get_file_patterns(self) -> list[str]:
        return [
            # Kubernetes
            "**/k8s/**/*.yaml",
            "**/k8s/**/*.yml",
            "**/kubernetes/**/*.yaml",
            "**/manifests/**/*.yaml",
            "**/deploy/**/*.yaml",
            # Docker
            "**/Dockerfile*",
            "**/docker-compose*.yaml",
            "**/docker-compose*.yml",
            # Helm
            "**/helm/**/*.yaml",
            "**/charts/**/*.yaml",
            "**/values*.yaml",
            # Terraform
            "**/*.tf",
            # Serverless
            "**/serverless.yaml",
            "**/serverless.yml",
            # CloudFlare
            "**/wrangler.toml",
        ]

    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze infrastructure configuration file."""
        components: list[ComponentInfo] = []
        file_path = parsed.file_path.lower()
        file_name = Path(parsed.file_path).name.lower()

        if file_name.startswith("dockerfile"):
            components.extend(self._analyze_dockerfile(parsed))
        elif "docker-compose" in file_name:
            components.extend(self._analyze_docker_compose(parsed))
        elif file_path.endswith((".yaml", ".yml")):
            components.extend(self._analyze_yaml_config(parsed))
        elif file_path.endswith(".tf"):
            components.extend(self._analyze_terraform(parsed))
        elif file_name == "wrangler.toml":
            components.extend(self._analyze_wrangler(parsed))

        return components

    def _analyze_dockerfile(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Dockerfile for best practices."""
        components: list[ComponentInfo] = []
        content = parsed.content
        lines = content.split("\n")

        # Extract base image
        from_match = re.search(r'^FROM\s+(\S+)', content, re.MULTILINE)
        base_image = from_match.group(1) if from_match else "unknown"

        component = ComponentInfo(
            name=f"Dockerfile ({base_image})",
            component_type=ComponentType.DEPLOYMENT,
            file_path=parsed.file_path,
            start_line=1,
            end_line=len(lines),
        )

        # Check for issues

        # 1. Using latest tag
        if ":latest" in base_image or ":" not in base_image:
            component.issues.append(Issue(
                severity=Severity.WARNING,
                message="Using 'latest' or untagged base image - pin a specific version",
                file_path=parsed.file_path,
                line_number=1,
                code="DOCKER_LATEST_TAG",
            ))

        # 2. Running as root
        if "USER" not in content:
            component.issues.append(Issue(
                severity=Severity.WARNING,
                message="No USER instruction - container runs as root",
                file_path=parsed.file_path,
                suggestion="Add 'USER nonroot' or similar",
                code="DOCKER_ROOT_USER",
            ))

        # 3. Secrets in build args
        for line_num, line in enumerate(lines, 1):
            if re.search(r'ARG\s+(password|secret|token|key|api_key)', line, re.IGNORECASE):
                component.issues.append(Issue(
                    severity=Severity.CRITICAL,
                    message="Potential secret in build argument",
                    file_path=parsed.file_path,
                    line_number=line_num,
                    code="DOCKER_SECRET_IN_ARG",
                ))

        # 4. apt-get without cleanup
        if "apt-get install" in content and "rm -rf /var/lib/apt/lists" not in content:
            component.issues.append(Issue(
                severity=Severity.INFO,
                message="apt-get install without cache cleanup increases image size",
                file_path=parsed.file_path,
                suggestion="Add 'rm -rf /var/lib/apt/lists/*' after apt-get",
                code="DOCKER_APT_CACHE",
            ))

        # 5. COPY before RUN (layer caching issue)
        copy_positions = [i for i, line in enumerate(lines) if line.strip().startswith("COPY")]
        run_positions = [i for i, line in enumerate(lines) if line.strip().startswith("RUN")]

        # Check if package install happens after source copy
        if copy_positions and run_positions:
            source_copy = next((i for i, line in enumerate(lines)
                               if line.strip().startswith("COPY") and ("." in line or "src" in line.lower())), None)
            package_install = next((i for i, line in enumerate(lines)
                                   if "npm install" in line or "pip install" in line or "yarn" in line), None)

            if source_copy is not None and package_install is not None:
                if source_copy < package_install:
                    component.issues.append(Issue(
                        severity=Severity.INFO,
                        message="Source files copied before package install - reduces layer cache effectiveness",
                        file_path=parsed.file_path,
                        suggestion="Copy package files first, install, then copy source",
                        code="DOCKER_LAYER_ORDER",
                    ))

        components.append(component)
        return components

    def _analyze_docker_compose(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze docker-compose.yaml file."""
        components: list[ComponentInfo] = []
        content = parsed.content

        try:
            compose = yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.debug(f"Failed to parse YAML: {e}")
            return components

        if not compose or "services" not in compose:
            return components

        for service_name, service_config in compose.get("services", {}).items():
            if not isinstance(service_config, dict):
                continue

            # Calculate line number (approximate)
            line_number = content.find(f"{service_name}:")
            line_number = content[:line_number].count("\n") + 1 if line_number != -1 else 1

            component = ComponentInfo(
                name=service_name,
                component_type=ComponentType.SERVICE_MESH,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,
            )

            # Check for issues

            # 1. Privileged mode
            if service_config.get("privileged", False):
                component.issues.append(Issue(
                    severity=Severity.CRITICAL,
                    message=f"Service {service_name} runs in privileged mode",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="COMPOSE_PRIVILEGED",
                ))

            # 2. Host network mode
            if service_config.get("network_mode") == "host":
                component.issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Service {service_name} uses host network - security risk",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="COMPOSE_HOST_NETWORK",
                ))

            # 3. Environment secrets
            env_vars = service_config.get("environment", {})
            if isinstance(env_vars, list):
                env_vars = {e.split("=")[0]: e.split("=")[1] if "=" in e else "" for e in env_vars}

            for key, value in env_vars.items():
                if any(s in key.lower() for s in ["password", "secret", "key", "token"]):
                    if isinstance(value, str) and not value.startswith("${"):
                        component.issues.append(Issue(
                            severity=Severity.CRITICAL,
                            message=f"Hardcoded secret in {key}",
                            file_path=parsed.file_path,
                            suggestion="Use environment variable substitution: ${SECRET}",
                            code="COMPOSE_HARDCODED_SECRET",
                        ))

            # 4. No healthcheck
            if "healthcheck" not in service_config:
                component.issues.append(Issue(
                    severity=Severity.INFO,
                    message=f"Service {service_name} lacks healthcheck",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="COMPOSE_NO_HEALTHCHECK",
                ))

            # 5. No restart policy
            if "restart" not in service_config:
                component.issues.append(Issue(
                    severity=Severity.INFO,
                    message=f"Service {service_name} lacks restart policy",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    suggestion="Add 'restart: unless-stopped' for production",
                    code="COMPOSE_NO_RESTART",
                ))

            components.append(component)

        return components

    def _analyze_yaml_config(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Kubernetes YAML manifests."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Split multi-document YAML
        documents = content.split("---")

        for doc_content in documents:
            doc_content = doc_content.strip()
            if not doc_content:
                continue

            try:
                doc = yaml.safe_load(doc_content)
            except yaml.YAMLError:
                continue

            if not isinstance(doc, dict) or "kind" not in doc:
                continue

            kind = doc.get("kind", "Unknown")
            metadata = doc.get("metadata", {})
            name = metadata.get("name", "unnamed")

            # Determine component type
            component_type = self._get_k8s_component_type(kind)

            # Calculate line number
            line_number = content.find(f"kind: {kind}")
            line_number = content[:line_number].count("\n") + 1 if line_number != -1 else 1

            component = ComponentInfo(
                name=f"{kind}/{name}",
                component_type=component_type,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + doc_content.count("\n"),
            )

            # Check for Kubernetes-specific issues
            component.issues.extend(self._check_k8s_issues(doc, parsed.file_path, line_number))

            components.append(component)

        return components

    def _get_k8s_component_type(self, kind: str) -> ComponentType:
        """Map Kubernetes kind to component type."""
        mapping = {
            "Deployment": ComponentType.DEPLOYMENT,
            "StatefulSet": ComponentType.DEPLOYMENT,
            "DaemonSet": ComponentType.DEPLOYMENT,
            "Service": ComponentType.SERVICE_MESH,
            "ConfigMap": ComponentType.CONFIG_MAP,
            "Secret": ComponentType.SECRET,
            "Ingress": ComponentType.INGRESS,
        }
        return mapping.get(kind, ComponentType.DEPLOYMENT)

    def _check_k8s_issues(self, doc: dict, file_path: str, line_number: int) -> list[Issue]:
        """Check Kubernetes manifest for issues."""
        issues = []
        kind = doc.get("kind", "")
        name = doc.get("metadata", {}).get("name", "unnamed")

        spec = doc.get("spec", {})

        if kind in ["Deployment", "StatefulSet", "DaemonSet"]:
            pod_spec = spec.get("template", {}).get("spec", {})
            containers = pod_spec.get("containers", [])

            for container in containers:
                container_name = container.get("name", "unnamed")

                # 1. No resource limits
                resources = container.get("resources", {})
                if not resources.get("limits"):
                    issues.append(Issue(
                        severity=Severity.WARNING,
                        message=f"Container {container_name} lacks resource limits",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion="Add resources.limits.cpu and resources.limits.memory",
                        code="K8S_NO_RESOURCE_LIMITS",
                    ))

                # 2. Running as root
                security_context = container.get("securityContext", {})
                if security_context.get("runAsRoot", True) and not security_context.get("runAsNonRoot", False):
                    issues.append(Issue(
                        severity=Severity.WARNING,
                        message=f"Container {container_name} may run as root",
                        file_path=file_path,
                        line_number=line_number,
                        suggestion="Add 'securityContext.runAsNonRoot: true'",
                        code="K8S_ROOT_CONTAINER",
                    ))

                # 3. No readiness probe
                if not container.get("readinessProbe"):
                    issues.append(Issue(
                        severity=Severity.INFO,
                        message=f"Container {container_name} lacks readiness probe",
                        file_path=file_path,
                        line_number=line_number,
                        code="K8S_NO_READINESS",
                    ))

                # 4. No liveness probe
                if not container.get("livenessProbe"):
                    issues.append(Issue(
                        severity=Severity.INFO,
                        message=f"Container {container_name} lacks liveness probe",
                        file_path=file_path,
                        line_number=line_number,
                        code="K8S_NO_LIVENESS",
                    ))

                # 5. Latest tag
                image = container.get("image", "")
                if ":latest" in image or ":" not in image:
                    issues.append(Issue(
                        severity=Severity.WARNING,
                        message=f"Container {container_name} uses 'latest' tag",
                        file_path=file_path,
                        line_number=line_number,
                        code="K8S_LATEST_TAG",
                    ))

            # 6. Single replica
            replicas = spec.get("replicas", 1)
            if replicas == 1:
                issues.append(Issue(
                    severity=Severity.INFO,
                    message=f"{kind} {name} has only 1 replica - no high availability",
                    file_path=file_path,
                    line_number=line_number,
                    code="K8S_SINGLE_REPLICA",
                ))

        elif kind == "Secret":
            # Check for base64-encoded secrets in plain text
            data = doc.get("data", {})
            string_data = doc.get("stringData", {})

            if string_data:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Secret {name} uses stringData - consider external secrets manager",
                    file_path=file_path,
                    line_number=line_number,
                    code="K8S_PLAIN_SECRET",
                ))

        elif kind == "Ingress":
            # Check for TLS
            tls = spec.get("tls", [])
            if not tls:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Ingress {name} lacks TLS configuration",
                    file_path=file_path,
                    line_number=line_number,
                    code="K8S_NO_TLS",
                ))

        return issues

    def _analyze_terraform(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Terraform files."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find resource blocks
        resource_pattern = r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{'

        for match in re.finditer(resource_pattern, content):
            resource_type = match.group(1)
            resource_name = match.group(2)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=f"{resource_type}.{resource_name}",
                component_type=ComponentType.DEPLOYMENT,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,
            )

            # Check for security issues
            block_start = match.end()
            brace_count = 1
            block_end = block_start

            for i, char in enumerate(content[block_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = block_start + i
                        break

            block_content = content[block_start:block_end]

            # Check for hardcoded secrets
            secret_pattern = r'(password|secret|api_key|token)\s*=\s*"[^"$]+'
            if re.search(secret_pattern, block_content, re.IGNORECASE):
                component.issues.append(Issue(
                    severity=Severity.CRITICAL,
                    message="Potential hardcoded secret in Terraform resource",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    suggestion="Use variables or secret management",
                    code="TF_HARDCODED_SECRET",
                ))

            # Check for public access
            if "publicly_accessible" in block_content and "true" in block_content:
                component.issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Resource {resource_name} is publicly accessible",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="TF_PUBLIC_ACCESS",
                ))

            components.append(component)

        return components

    def _analyze_wrangler(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Cloudflare wrangler.toml configuration."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Extract worker name
        name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
        worker_name = name_match.group(1) if name_match else "unnamed-worker"

        component = ComponentInfo(
            name=f"CF Worker: {worker_name}",
            component_type=ComponentType.DEPLOYMENT,
            file_path=parsed.file_path,
            start_line=1,
            end_line=content.count("\n") + 1,
        )

        # Check for bindings
        bindings = []
        if "kv_namespaces" in content:
            bindings.append("KV")
        if "r2_buckets" in content:
            bindings.append("R2")
        if "d1_databases" in content:
            bindings.append("D1")
        if "queues" in content:
            bindings.append("Queues")
        if "durable_objects" in content:
            bindings.append("Durable Objects")
        if "vectorize" in content:
            bindings.append("Vectorize")

        component.methods = bindings

        # Check for issues

        # 1. No compatibility date
        if "compatibility_date" not in content:
            component.issues.append(Issue(
                severity=Severity.INFO,
                message="No compatibility_date specified",
                file_path=parsed.file_path,
                suggestion="Add compatibility_date for predictable behavior",
                code="CF_NO_COMPAT_DATE",
            ))

        # 2. Hardcoded secrets in vars
        if re.search(r'\[vars\].*?(api_key|secret|token|password)\s*=\s*"[^"$]+', content, re.IGNORECASE | re.DOTALL):
            component.issues.append(Issue(
                severity=Severity.CRITICAL,
                message="Potential hardcoded secret in [vars]",
                file_path=parsed.file_path,
                suggestion="Use secrets or environment variables",
                code="CF_HARDCODED_SECRET",
            ))

        components.append(component)
        return components

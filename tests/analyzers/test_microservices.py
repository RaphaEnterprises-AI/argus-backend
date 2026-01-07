"""Tests for the microservices analyzer module."""

import pytest
from pathlib import Path

from src.analyzers.microservices import MicroservicesAnalyzer
from src.analyzers.base import ComponentType, Severity


class TestMicroservicesAnalyzer:
    """Test MicroservicesAnalyzer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with infrastructure files."""
        # Create Dockerfile
        (tmp_path / "Dockerfile").write_text('''
FROM node:18

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000
CMD ["npm", "start"]
''')

        # Create docker-compose
        (tmp_path / "docker-compose.yml").write_text('''
version: "3.8"
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://localhost:5432/app
      - SECRET_KEY=hardcoded_secret
    depends_on:
      - db

  db:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: password123
    volumes:
      - db_data:/var/lib/postgresql/data

  worker:
    build: .
    command: npm run worker
    privileged: true

volumes:
  db_data:
''')

        # Create Kubernetes manifests
        k8s_dir = tmp_path / "k8s"
        k8s_dir.mkdir()

        (k8s_dir / "deployment.yaml").write_text('''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: myapp:latest
        ports:
        - containerPort: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: 3000
''')

        (k8s_dir / "ingress.yaml").write_text('''
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
''')

        # Create wrangler.toml
        (tmp_path / "wrangler.toml").write_text('''
name = "argus-worker"
main = "src/index.ts"
compatibility_date = "2024-01-01"

[vars]
API_KEY = "secret_value"

[[kv_namespaces]]
binding = "CACHE"
id = "abc123"

[[r2_buckets]]
binding = "STORAGE"
bucket_name = "argus-storage"
''')

        return tmp_path

    @pytest.fixture
    def analyzer(self, temp_repo):
        """Create a MicroservicesAnalyzer for the temp repo."""
        return MicroservicesAnalyzer(str(temp_repo))

    def test_analyzer_type(self, analyzer):
        """Test analyzer type property."""
        assert analyzer.analyzer_type == "microservices"

    def test_file_patterns(self, analyzer):
        """Test that file patterns include infrastructure files."""
        patterns = analyzer.get_file_patterns()

        assert any("Dockerfile" in p for p in patterns)
        assert any("docker-compose" in p for p in patterns)
        assert any("k8s" in p or "kubernetes" in p for p in patterns)
        assert any(".yaml" in p or ".yml" in p for p in patterns)

    def test_analyze_finds_dockerfile(self, analyzer):
        """Test that analysis finds Dockerfile."""
        result = analyzer.analyze()

        docker_components = [
            c for c in result.components
            if "Dockerfile" in c.name or "FROM" in (c.name or "")
        ]
        assert len(docker_components) >= 1

    def test_analyze_finds_docker_compose_services(self, analyzer):
        """Test that analysis finds docker-compose services."""
        result = analyzer.analyze()

        # Should find api, db, worker services
        service_names = [c.name for c in result.components]
        assert any("api" in n.lower() for n in service_names if n)

    def test_analyze_finds_k8s_resources(self, analyzer):
        """Test that analysis finds Kubernetes resources."""
        result = analyzer.analyze()

        k8s_components = [
            c for c in result.components
            if "Deployment" in (c.name or "") or "Service" in (c.name or "")
        ]
        assert len(k8s_components) >= 1

    def test_analyze_detects_hardcoded_secrets(self, analyzer):
        """Test that analysis detects hardcoded secrets."""
        result = analyzer.analyze()

        secret_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "secret" in issue.message.lower() or "hardcoded" in issue.message.lower()
        ]
        # Should have detected hardcoded secrets in docker-compose

    def test_analyze_detects_privileged_container(self, analyzer):
        """Test that analysis detects privileged containers."""
        result = analyzer.analyze()

        privileged_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "privileged" in issue.message.lower()
        ]
        # Should have detected the privileged: true in docker-compose

    def test_analyze_detects_latest_tag(self, analyzer):
        """Test that analysis detects 'latest' image tag."""
        result = analyzer.analyze()

        latest_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "latest" in issue.message.lower()
        ]
        # Should have detected myapp:latest in k8s deployment

    def test_analyze_detects_single_replica(self, analyzer):
        """Test that analysis detects single replica deployment."""
        result = analyzer.analyze()

        replica_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "replica" in issue.message.lower()
        ]
        # Should have detected replicas: 1

    def test_analyze_detects_missing_tls(self, analyzer):
        """Test that analysis detects missing TLS in ingress."""
        result = analyzer.analyze()

        tls_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "tls" in issue.message.lower()
        ]
        # Should have detected missing TLS

    def test_analyze_finds_wrangler_config(self, analyzer):
        """Test that analysis finds Cloudflare wrangler config."""
        result = analyzer.analyze()

        cf_components = [
            c for c in result.components
            if "CF Worker" in (c.name or "") or "argus-worker" in (c.name or "")
        ]
        assert len(cf_components) >= 1


class TestMicroservicesAnalyzerDockerfile:
    """Test Dockerfile analysis in MicroservicesAnalyzer."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create repo with Dockerfile issues."""
        (tmp_path / "Dockerfile").write_text('''
FROM ubuntu:latest

ARG DATABASE_PASSWORD=secret123

RUN apt-get update && apt-get install -y curl

COPY . .
RUN npm install

CMD ["npm", "start"]
''')
        return tmp_path

    def test_detects_latest_tag_in_from(self, temp_repo):
        """Test detection of :latest tag in FROM."""
        analyzer = MicroservicesAnalyzer(str(temp_repo))
        result = analyzer.analyze()

        latest_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "latest" in issue.message.lower()
        ]
        assert len(latest_issues) >= 1

    def test_detects_secret_in_arg(self, temp_repo):
        """Test detection of secrets in ARG."""
        analyzer = MicroservicesAnalyzer(str(temp_repo))
        result = analyzer.analyze()

        secret_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "secret" in issue.message.lower() or "password" in issue.message.lower()
        ]
        # Should detect DATABASE_PASSWORD in ARG

    def test_detects_no_user(self, temp_repo):
        """Test detection of missing USER instruction."""
        analyzer = MicroservicesAnalyzer(str(temp_repo))
        result = analyzer.analyze()

        user_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "root" in issue.message.lower() or "user" in issue.message.lower()
        ]
        # Should detect running as root

    def test_detects_apt_cache(self, temp_repo):
        """Test detection of apt cache not cleaned."""
        analyzer = MicroservicesAnalyzer(str(temp_repo))
        result = analyzer.analyze()

        apt_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "apt" in issue.message.lower() and "cache" in issue.message.lower()
        ]
        # Should detect apt-get without cache cleanup

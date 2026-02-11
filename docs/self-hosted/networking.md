# Networking Guide

Configure networking for Skopaq Enterprise deployments.

## Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              External Traffic                                │
│                                                                              │
│  Users ────────────────► Ingress Controller ─────────────► Skopaq Services   │
│  AI Assistants ─────────►  (nginx/traefik)                                  │
│  CI/CD Systems ─────────►                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Skopaq Namespace                                 │
│                                                                              │
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│  │    Brain API      │    │    MCP Server     │    │   Selenium Hub    │   │
│  │    :8000          │    │    :3000          │    │    :4444          │   │
│  └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘   │
│            │                        │                        │              │
│  ┌─────────┴────────────────────────┴────────────────────────┴───────────┐  │
│  │                         Internal Network                               │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐  │  │
│  │  │PostgreSQL │  │   Redis   │  │   MinIO   │  │  Chrome Nodes     │  │  │
│  │  │  :5432    │  │  :6379    │  │  :9000    │  │  (internal only)  │  │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Port Reference

### External Ports (Exposed via Ingress)

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Brain API | 8000 | HTTPS | REST API endpoints |
| MCP Server | 3000 | HTTPS/SSE | AI assistant integration |

### Internal Ports (Cluster-only)

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| PostgreSQL | 5432 | TCP | Database connections |
| Redis | 6379 | TCP | Cache and sessions |
| MinIO API | 9000 | HTTP | Object storage |
| MinIO Console | 9001 | HTTP | Admin interface |
| Selenium Hub | 4444 | HTTP | Test distribution |
| Chrome Nodes | 5555 | HTTP | Browser automation |
| Ollama | 11434 | HTTP | Local LLM inference |

## Ingress Configuration

### NGINX Ingress

```yaml
# values.yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    brain: "argus-api.company.com"
    mcp: "argus-mcp.company.com"
  tls:
    enabled: true
    secretName: "argus-tls"
```

### Traefik Ingress

```yaml
ingress:
  enabled: true
  className: "traefik"
  annotations:
    traefik.ingress.kubernetes.io/router.middlewares: "default-compress@kubernetescrd"
    traefik.ingress.kubernetes.io/router.tls: "true"
```

### AWS ALB Ingress

```yaml
ingress:
  enabled: true
  className: "alb"
  annotations:
    alb.ingress.kubernetes.io/scheme: "internet-facing"
    alb.ingress.kubernetes.io/target-type: "ip"
    alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:..."
    alb.ingress.kubernetes.io/ssl-policy: "ELBSecurityPolicy-TLS13-1-2-2021-06"
```

## Network Policies

### Default Deny Policy

The Helm chart includes a default-deny policy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: argus
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
```

### Allow Ingress Traffic

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress
  namespace: argus
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/component: brain
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
```

### Allow Inter-Service Communication

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-internal
  namespace: argus
spec:
  podSelector: {}
  ingress:
    - from:
        - podSelector: {}
  egress:
    - to:
        - podSelector: {}
```

### Enable Network Policies

```yaml
# values.yaml
networkPolicies:
  enabled: true
  allowedNamespaces:
    - ingress-nginx
    - monitoring
    - kube-system
```

## TLS Configuration

### Using cert-manager

```yaml
# Create ClusterIssuer
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@company.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

```yaml
# values.yaml
ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  tls:
    enabled: true
    secretName: "argus-tls"  # cert-manager will create this
```

### Using Pre-existing Certificate

```bash
# Create TLS secret
kubectl create secret tls argus-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n argus
```

```yaml
# values.yaml
ingress:
  tls:
    enabled: true
    secretName: "argus-tls"
```

## DNS Configuration

### External DNS (Kubernetes)

```yaml
# Annotations for external-dns
ingress:
  annotations:
    external-dns.alpha.kubernetes.io/hostname: "argus-api.company.com,argus-mcp.company.com"
    external-dns.alpha.kubernetes.io/ttl: "300"
```

### Manual DNS Records

Create these DNS records:

| Type | Name | Value |
|------|------|-------|
| A/CNAME | argus-api.company.com | Ingress IP/hostname |
| A/CNAME | argus-mcp.company.com | Ingress IP/hostname |

## Load Balancing

### Session Affinity for SSE

MCP Server uses Server-Sent Events (SSE), which requires sticky sessions:

```yaml
# For NGINX Ingress
ingress:
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "argus-sticky"
    nginx.ingress.kubernetes.io/session-cookie-expires: "172800"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "172800"
```

### Health Check Configuration

```yaml
# For AWS ALB
ingress:
  annotations:
    alb.ingress.kubernetes.io/healthcheck-path: "/health"
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: "5"
    alb.ingress.kubernetes.io/healthy-threshold-count: "2"
    alb.ingress.kubernetes.io/unhealthy-threshold-count: "3"
```

## Firewall Rules

### Required Outbound (Cloud LLM)

| Destination | Port | Purpose |
|-------------|------|---------|
| api.anthropic.com | 443 | Claude API |
| api.openai.com | 443 | OpenAI API (if used) |
| openrouter.ai | 443 | OpenRouter (if used) |

### Required Outbound (Package Managers)

| Destination | Port | Purpose |
|-------------|------|---------|
| pypi.org | 443 | Python packages |
| registry.npmjs.org | 443 | Node packages |
| ghcr.io | 443 | Container images |

### Air-Gap Mode

No outbound connectivity required when using:
- Local container registry
- Ollama for LLM inference
- Pre-loaded models

## Service Mesh Integration

### Istio

```yaml
# PeerAuthentication for mTLS
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: argus-mtls
  namespace: argus
spec:
  mtls:
    mode: STRICT
```

### Linkerd

```yaml
# Annotations for automatic injection
brain:
  podAnnotations:
    linkerd.io/inject: enabled
```

## Troubleshooting

### Test Connectivity

```bash
# From inside cluster
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
nslookup argus-brain.argus.svc.cluster.local
wget -qO- http://argus-brain.argus.svc.cluster.local:8000/health
```

### Check Network Policies

```bash
# List policies
kubectl get networkpolicies -n argus

# Describe policy
kubectl describe networkpolicy allow-ingress -n argus
```

### Debug Ingress

```bash
# Check ingress status
kubectl get ingress -n argus

# Check ingress controller logs
kubectl logs -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx
```

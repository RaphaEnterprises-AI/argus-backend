#!/bin/bash
# K3s Server (Control Plane) Initialization Script

set -e

# Wait for cloud-init to complete
cloud-init status --wait

# Update system
apt-get update
apt-get upgrade -y

# Install required packages
apt-get install -y \
    curl \
    wget \
    jq \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Disable swap (required for Kubernetes)
swapoff -a
sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Load required kernel modules
cat <<EOF | tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

# Set sysctl params
cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system

# Install K3s
%{ if is_first_server }
# First server - initialize cluster
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server" sh -s - \
    --token=${k3s_token} \
    --cluster-init \
    --disable=traefik \
    --disable=servicelb \
    --write-kubeconfig-mode=644 \
    --node-label="node.kubernetes.io/role=control-plane" \
    --tls-san=$(curl -s http://169.254.169.254/hetzner/v1/metadata/public-ipv4)
%{ else }
# Additional server - join cluster
sleep 30  # Wait for first server to be ready
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server" sh -s - \
    --token=${k3s_token} \
    --server=https://${server_ip}:6443 \
    --disable=traefik \
    --disable=servicelb \
    --write-kubeconfig-mode=644 \
    --node-label="node.kubernetes.io/role=control-plane"
%{ endif }

# Wait for K3s to be ready
sleep 10
until kubectl get nodes &>/dev/null; do
    echo "Waiting for K3s to be ready..."
    sleep 5
done

%{ if is_first_server }
# Install Traefik (latest version with better ingress support)
kubectl apply -f https://raw.githubusercontent.com/traefik/traefik/v3.0/docs/content/reference/dynamic-configuration/kubernetes-crd-definition-v1.yml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: traefik
---
apiVersion: helm.cattle.io/v1
kind: HelmChart
metadata:
  name: traefik
  namespace: kube-system
spec:
  repo: https://traefik.github.io/charts
  chart: traefik
  targetNamespace: traefik
  valuesContent: |-
    deployment:
      replicas: 2
    service:
      type: NodePort
    ports:
      web:
        nodePort: 80
      websecure:
        nodePort: 443
    ingressRoute:
      dashboard:
        enabled: false
    providers:
      kubernetesCRD:
        enabled: true
      kubernetesIngress:
        enabled: true
EOF

# Install metrics-server for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Patch metrics-server for self-signed certs (K3s default)
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}
]'
%{ endif }

echo "K3s server initialization complete!"

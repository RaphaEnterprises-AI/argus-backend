#!/bin/bash
# K3s Agent (Worker Node) Initialization Script
# Optimized for browser automation workloads

set -e

# Wait for cloud-init to complete
cloud-init status --wait

# Update system
apt-get update
apt-get upgrade -y

# Install required packages for browser automation
apt-get install -y \
    curl \
    wget \
    jq \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    xvfb \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2

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

# Set sysctl params (optimized for browser containers)
cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1

# Increase file descriptors (browsers need many)
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 8192

# Network optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# Memory optimizations for browsers
vm.max_map_count = 262144
vm.swappiness = 10
EOF

sysctl --system

# Increase file descriptor limits
cat <<EOF | tee /etc/security/limits.d/browser.conf
* soft nofile 1048576
* hard nofile 1048576
* soft nproc 65535
* hard nproc 65535
EOF

# Wait for control plane to be ready
echo "Waiting for control plane at ${server_ip}..."
until curl -sf -o /dev/null https://${server_ip}:6443 --insecure 2>/dev/null; do
    echo "Control plane not ready, waiting..."
    sleep 10
done

# Install K3s Agent
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent" sh -s - \
    --token=${k3s_token} \
    --server=https://${server_ip}:6443 \
    --node-label="node.kubernetes.io/role=worker" \
    --node-label="workload-type=browser" \
    --kubelet-arg="max-pods=50" \
    --kubelet-arg="system-reserved=cpu=200m,memory=500Mi" \
    --kubelet-arg="kube-reserved=cpu=200m,memory=500Mi"

echo "K3s agent initialization complete!"
echo "Node is ready for browser workloads."

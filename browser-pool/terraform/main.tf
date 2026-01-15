# Argus Browser Pool - Hetzner Kubernetes Cluster
# Using K3s for lightweight, fast Kubernetes

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = ">= 1.45.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.4.0"
    }
  }
}

# Configure Hetzner Cloud Provider
provider "hcloud" {
  token = var.hcloud_token
}

# SSH Key for cluster access
resource "tls_private_key" "cluster_ssh" {
  algorithm = "ED25519"
}

resource "hcloud_ssh_key" "cluster" {
  name       = "${var.cluster_name}-ssh"
  public_key = tls_private_key.cluster_ssh.public_key_openssh
}

resource "local_file" "ssh_private_key" {
  content         = tls_private_key.cluster_ssh.private_key_openssh
  filename        = "${path.module}/keys/${var.cluster_name}-ssh.pem"
  file_permission = "0600"
}

# Private Network for cluster
resource "hcloud_network" "cluster" {
  name     = "${var.cluster_name}-network"
  ip_range = "10.0.0.0/16"
}

resource "hcloud_network_subnet" "cluster" {
  network_id   = hcloud_network.cluster.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.1.0/24"
}

# Firewall for cluster nodes
resource "hcloud_firewall" "cluster" {
  name = "${var.cluster_name}-firewall"

  # SSH
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # Kubernetes API
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "6443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # HTTP/HTTPS (Traefik)
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # NodePort range
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "30000-32767"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # Allow all internal traffic
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "any"
    source_ips = ["10.0.0.0/16"]
  }

  rule {
    direction  = "in"
    protocol   = "udp"
    port       = "any"
    source_ips = ["10.0.0.0/16"]
  }

  # Allow all outbound
  rule {
    direction       = "out"
    protocol        = "tcp"
    port            = "any"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "udp"
    port            = "any"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction       = "out"
    protocol        = "icmp"
    destination_ips = ["0.0.0.0/0", "::/0"]
  }
}

# Control Plane Node (K3s Server)
resource "hcloud_server" "control_plane" {
  count       = var.control_plane_count
  name        = "${var.cluster_name}-control-${count.index + 1}"
  image       = "ubuntu-22.04"
  server_type = var.control_plane_type
  location    = var.location
  ssh_keys    = [hcloud_ssh_key.cluster.id]
  firewall_ids = [hcloud_firewall.cluster.id]

  labels = {
    cluster = var.cluster_name
    role    = "control-plane"
  }

  network {
    network_id = hcloud_network.cluster.id
    ip         = "10.0.1.${10 + count.index}"
  }

  user_data = count.index == 0 ? templatefile("${path.module}/templates/k3s-server-init.sh", {
    k3s_token       = random_password.k3s_token.result
    cluster_name    = var.cluster_name
    is_first_server = true
    server_ip       = ""
  }) : templatefile("${path.module}/templates/k3s-server-init.sh", {
    k3s_token       = random_password.k3s_token.result
    cluster_name    = var.cluster_name
    is_first_server = false
    server_ip       = hcloud_server.control_plane[0].ipv4_address
  })

  depends_on = [hcloud_network_subnet.cluster]
}

# Worker Nodes (K3s Agents) - For Browser Pods
resource "hcloud_server" "worker" {
  count       = var.worker_count
  name        = "${var.cluster_name}-worker-${count.index + 1}"
  image       = "ubuntu-22.04"
  server_type = var.worker_type
  location    = var.location
  ssh_keys    = [hcloud_ssh_key.cluster.id]
  firewall_ids = [hcloud_firewall.cluster.id]

  labels = {
    cluster = var.cluster_name
    role    = "worker"
  }

  network {
    network_id = hcloud_network.cluster.id
    ip         = "10.0.1.${100 + count.index}"
  }

  user_data = templatefile("${path.module}/templates/k3s-agent-init.sh", {
    k3s_token  = random_password.k3s_token.result
    server_ip  = hcloud_server.control_plane[0].ipv4_address
  })

  depends_on = [hcloud_server.control_plane[0]]
}

# K3s Token
resource "random_password" "k3s_token" {
  length  = 32
  special = false
}

# Load Balancer for Ingress
resource "hcloud_load_balancer" "ingress" {
  name               = "${var.cluster_name}-lb"
  load_balancer_type = var.load_balancer_type
  location           = var.location

  labels = {
    cluster = var.cluster_name
  }
}

resource "hcloud_load_balancer_network" "ingress" {
  load_balancer_id = hcloud_load_balancer.ingress.id
  network_id       = hcloud_network.cluster.id
  ip               = "10.0.1.250"
}

resource "hcloud_load_balancer_target" "ingress" {
  count            = var.worker_count
  type             = "server"
  load_balancer_id = hcloud_load_balancer.ingress.id
  server_id        = hcloud_server.worker[count.index].id
  use_private_ip   = true

  depends_on = [hcloud_load_balancer_network.ingress]
}

# HTTP Service
resource "hcloud_load_balancer_service" "http" {
  load_balancer_id = hcloud_load_balancer.ingress.id
  protocol         = "tcp"
  listen_port      = 80
  destination_port = 80

  health_check {
    protocol = "http"
    port     = 80
    interval = 10
    timeout  = 5
    retries  = 3
    http {
      path         = "/health"
      status_codes = ["2??", "3??"]
    }
  }
}

# HTTPS Service
resource "hcloud_load_balancer_service" "https" {
  load_balancer_id = hcloud_load_balancer.ingress.id
  protocol         = "tcp"
  listen_port      = 443
  destination_port = 443

  health_check {
    protocol = "tcp"
    port     = 443
    interval = 10
    timeout  = 5
    retries  = 3
  }
}

# Generate kubeconfig
resource "null_resource" "kubeconfig" {
  depends_on = [hcloud_server.control_plane[0]]

  provisioner "local-exec" {
    command = <<-EOT
      sleep 60  # Wait for K3s to initialize
      mkdir -p ${path.module}/keys
      ssh -o StrictHostKeyChecking=no -i ${local_file.ssh_private_key.filename} \
        root@${hcloud_server.control_plane[0].ipv4_address} \
        'cat /etc/rancher/k3s/k3s.yaml' | \
        sed "s/127.0.0.1/${hcloud_server.control_plane[0].ipv4_address}/g" \
        > ${path.module}/keys/${var.cluster_name}-kubeconfig.yaml
    EOT
  }
}

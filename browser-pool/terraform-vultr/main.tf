# Vultr Browser Pool - Terraform Configuration
# Deploys a managed Kubernetes cluster with browser workers in Mumbai

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    vultr = {
      source  = "vultr/vultr"
      version = "~> 2.19"
    }
  }
}

# Configure Vultr Provider
provider "vultr" {
  api_key     = var.vultr_api_key
  rate_limit  = 100
  retry_limit = 3
}

# Variables
variable "vultr_api_key" {
  description = "Vultr API key"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "argus-browser-pool"
}

variable "region" {
  description = "Vultr region (bom = Mumbai, del = Delhi)"
  type        = string
  default     = "bom"  # Mumbai - fastest for India
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "v1.32.9+3"  # Stable version (available: v1.34.1+3, v1.33.5+3, v1.32.9+3)
}

variable "node_pool_plan" {
  description = "Plan for worker nodes"
  type        = string
  default     = "vc2-2c-4gb"  # 2 vCPU, 4GB RAM - good for browsers
}

variable "node_pool_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 2
}

variable "node_pool_min" {
  description = "Minimum nodes for autoscaling"
  type        = number
  default     = 1
}

variable "node_pool_max" {
  description = "Maximum nodes for autoscaling"
  type        = number
  default     = 5
}

variable "enable_autoscaling" {
  description = "Enable node autoscaling"
  type        = bool
  default     = true
}

# Create VKE (Vultr Kubernetes Engine) Cluster
resource "vultr_kubernetes" "browser_pool" {
  region  = var.region
  label   = var.cluster_name
  version = var.k8s_version

  # Main node pool for browser workers
  node_pools {
    node_quantity = var.node_pool_count
    plan          = var.node_pool_plan
    label         = "browser-workers"
    auto_scaler   = var.enable_autoscaling
    min_nodes     = var.node_pool_min
    max_nodes     = var.node_pool_max
  }
}

# Firewall for the cluster
resource "vultr_firewall_group" "browser_pool" {
  description = "Firewall for browser pool cluster"
}

# Allow HTTP
resource "vultr_firewall_rule" "http" {
  firewall_group_id = vultr_firewall_group.browser_pool.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "80"
  notes             = "Allow HTTP"
}

# Allow HTTPS
resource "vultr_firewall_rule" "https" {
  firewall_group_id = vultr_firewall_group.browser_pool.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "443"
  notes             = "Allow HTTPS"
}

# Allow Kubernetes API (optional - for kubectl access)
resource "vultr_firewall_rule" "k8s_api" {
  firewall_group_id = vultr_firewall_group.browser_pool.id
  protocol          = "tcp"
  ip_type           = "v4"
  subnet            = "0.0.0.0"
  subnet_size       = 0
  port              = "6443"
  notes             = "Allow Kubernetes API"
}

# Output cluster information
output "cluster_id" {
  description = "VKE Cluster ID"
  value       = vultr_kubernetes.browser_pool.id
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = vultr_kubernetes.browser_pool.endpoint
  sensitive   = true
}

output "cluster_status" {
  description = "Cluster status"
  value       = vultr_kubernetes.browser_pool.status
}

output "kubeconfig" {
  description = "Kubeconfig for kubectl"
  value       = vultr_kubernetes.browser_pool.kube_config
  sensitive   = true
}

output "node_pool_id" {
  description = "Node pool ID"
  value       = vultr_kubernetes.browser_pool.node_pools[0].id
}

output "region" {
  description = "Deployed region"
  value       = var.region
}

# Save kubeconfig to file
resource "local_file" "kubeconfig" {
  content         = vultr_kubernetes.browser_pool.kube_config
  filename        = "${path.module}/kubeconfig.yaml"
  file_permission = "0600"
}

# Argus Browser Pool - Terraform Variables

variable "hcloud_token" {
  description = "Hetzner Cloud API Token"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "argus-browser-pool"
}

variable "location" {
  description = "Hetzner Cloud location (nbg1, fsn1, hel1, ash, hil)"
  type        = string
  default     = "nbg1"  # Nuremberg, Germany - cheapest
}

variable "control_plane_count" {
  description = "Number of control plane nodes (1 for dev, 3 for HA)"
  type        = number
  default     = 1
}

variable "control_plane_type" {
  description = "Server type for control plane nodes"
  type        = string
  default     = "cx21"  # 2 vCPU, 4GB RAM - €4.85/mo
}

variable "worker_count" {
  description = "Number of worker nodes for browser pods"
  type        = number
  default     = 3
}

variable "worker_type" {
  description = "Server type for worker nodes (browser pods)"
  type        = string
  default     = "cx31"  # 2 vCPU, 8GB RAM - €8.98/mo (good for ~4 browsers per node)

  # Options:
  # cx21  - 2 vCPU,  4GB RAM - €4.85/mo  - ~2 browsers
  # cx31  - 2 vCPU,  8GB RAM - €8.98/mo  - ~4 browsers
  # cx41  - 4 vCPU, 16GB RAM - €15.59/mo - ~8 browsers
  # cx51  - 8 vCPU, 32GB RAM - €29.59/mo - ~16 browsers
  # ccx13 - 2 vCPU,  8GB RAM - €12.99/mo - Dedicated, ~4 browsers (production)
  # ccx23 - 4 vCPU, 16GB RAM - €24.99/mo - Dedicated, ~8 browsers (production)
}

variable "load_balancer_type" {
  description = "Load balancer type"
  type        = string
  default     = "lb11"  # 25 targets, €5.39/mo

  # Options:
  # lb11 - 25 targets   - €5.39/mo
  # lb21 - 75 targets   - €14.39/mo
  # lb31 - 150 targets  - €28.79/mo
}

# Scaling presets
variable "scale_preset" {
  description = "Scaling preset (dev, small, medium, large, enterprise)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "small", "medium", "large", "enterprise"], var.scale_preset)
    error_message = "Scale preset must be one of: dev, small, medium, large, enterprise"
  }
}

# Preset configurations
locals {
  scale_presets = {
    dev = {
      control_plane_count = 1
      control_plane_type  = "cx21"
      worker_count        = 3
      worker_type         = "cx21"
      load_balancer_type  = "lb11"
      max_browsers        = 6
      estimated_cost      = "€25/mo"
    }
    small = {
      control_plane_count = 1
      control_plane_type  = "cx21"
      worker_count        = 5
      worker_type         = "cx31"
      load_balancer_type  = "lb11"
      max_browsers        = 20
      estimated_cost      = "€55/mo"
    }
    medium = {
      control_plane_count = 3
      control_plane_type  = "cx31"
      worker_count        = 15
      worker_type         = "cx41"
      load_balancer_type  = "lb21"
      max_browsers        = 120
      estimated_cost      = "€280/mo"
    }
    large = {
      control_plane_count = 3
      control_plane_type  = "cx41"
      worker_count        = 50
      worker_type         = "cx51"
      load_balancer_type  = "lb31"
      max_browsers        = 800
      estimated_cost      = "€1,600/mo"
    }
    enterprise = {
      control_plane_count = 3
      control_plane_type  = "ccx23"
      worker_count        = 100
      worker_type         = "ccx33"
      load_balancer_type  = "lb31"
      max_browsers        = 3200
      estimated_cost      = "€6,000/mo"
    }
  }

  # Use preset values if scale_preset is set, otherwise use individual variables
  effective_control_plane_count = var.scale_preset != "" ? local.scale_presets[var.scale_preset].control_plane_count : var.control_plane_count
  effective_control_plane_type  = var.scale_preset != "" ? local.scale_presets[var.scale_preset].control_plane_type : var.control_plane_type
  effective_worker_count        = var.scale_preset != "" ? local.scale_presets[var.scale_preset].worker_count : var.worker_count
  effective_worker_type         = var.scale_preset != "" ? local.scale_presets[var.scale_preset].worker_type : var.worker_type
  effective_load_balancer_type  = var.scale_preset != "" ? local.scale_presets[var.scale_preset].load_balancer_type : var.load_balancer_type
}

# Argus Browser Pool - Terraform Outputs

output "cluster_name" {
  description = "Name of the Kubernetes cluster"
  value       = var.cluster_name
}

output "control_plane_ips" {
  description = "Public IPs of control plane nodes"
  value       = hcloud_server.control_plane[*].ipv4_address
}

output "worker_ips" {
  description = "Public IPs of worker nodes"
  value       = hcloud_server.worker[*].ipv4_address
}

output "load_balancer_ip" {
  description = "Public IP of the load balancer"
  value       = hcloud_load_balancer.ingress.ipv4
}

output "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  value       = "${path.module}/keys/${var.cluster_name}-kubeconfig.yaml"
}

output "ssh_private_key_path" {
  description = "Path to the SSH private key"
  value       = local_file.ssh_private_key.filename
}

output "browser_pool_url" {
  description = "URL of the browser pool API"
  value       = "http://${hcloud_load_balancer.ingress.ipv4}"
}

output "scale_info" {
  description = "Current scale configuration"
  value = {
    preset              = var.scale_preset
    control_plane_count = var.control_plane_count
    worker_count        = var.worker_count
    worker_type         = var.worker_type
    estimated_browsers  = var.worker_count * 4  # Rough estimate
  }
}

output "connection_commands" {
  description = "Commands to connect to the cluster"
  value       = <<-EOT
    # Set kubeconfig
    export KUBECONFIG=${path.module}/keys/${var.cluster_name}-kubeconfig.yaml

    # Verify cluster
    kubectl get nodes

    # SSH to control plane
    ssh -i ${local_file.ssh_private_key.filename} root@${hcloud_server.control_plane[0].ipv4_address}

    # Deploy browser pool
    kubectl apply -f ../kubernetes/

    # Test API
    curl http://${hcloud_load_balancer.ingress.ipv4}/health
  EOT
}

output "monthly_cost_estimate" {
  description = "Estimated monthly cost"
  value = {
    control_plane = "${var.control_plane_count} x ${var.control_plane_type}"
    workers       = "${var.worker_count} x ${var.worker_type}"
    load_balancer = var.load_balancer_type
    note          = "Check Hetzner pricing for exact costs: https://www.hetzner.com/cloud"
  }
}

# Confluent Cloud Terraform Configuration
# Creates: Environment, Kafka Cluster, Topics, Flink Compute Pool, API Keys
#
# Usage:
#   export CONFLUENT_CLOUD_API_KEY="your-cloud-api-key"
#   export CONFLUENT_CLOUD_API_SECRET="your-cloud-api-secret"
#   terraform init
#   terraform apply

terraform {
  required_version = ">= 1.3.0"

  required_providers {
    confluent = {
      source  = "confluentinc/confluent"
      version = "~> 2.0"
    }
  }
}

# Configure the Confluent Provider
# Credentials from environment variables: CONFLUENT_CLOUD_API_KEY, CONFLUENT_CLOUD_API_SECRET
provider "confluent" {}

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

variable "environment_name" {
  description = "Name of the Confluent Cloud environment"
  type        = string
  default     = "argus-prod"
}

variable "cluster_name" {
  description = "Name of the Kafka cluster"
  type        = string
  default     = "argus-kafka"
}

variable "region" {
  description = "Cloud region for resources"
  type        = string
  default     = "ap-south-2"  # Hyderabad (closest to your current setup)
}

variable "cloud_provider" {
  description = "Cloud provider (AWS, GCP, AZURE)"
  type        = string
  default     = "AWS"
}

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

resource "confluent_environment" "argus" {
  display_name = var.environment_name

  stream_governance {
    package = "ESSENTIALS"  # Free tier for Schema Registry
  }
}

# -----------------------------------------------------------------------------
# Kafka Cluster (Basic - cheapest option)
# -----------------------------------------------------------------------------

resource "confluent_kafka_cluster" "argus" {
  display_name = var.cluster_name
  availability = "SINGLE_ZONE"  # Cost savings (use MULTI_ZONE for prod)
  cloud        = var.cloud_provider
  region       = var.region

  basic {}  # Basic cluster type (cheapest, pay-per-use)

  environment {
    id = confluent_environment.argus.id
  }
}

# -----------------------------------------------------------------------------
# Service Account for Applications
# -----------------------------------------------------------------------------

resource "confluent_service_account" "argus_app" {
  display_name = "argus-application"
  description  = "Service account for Argus application"
}

# -----------------------------------------------------------------------------
# API Key for Application (Kafka operations)
# -----------------------------------------------------------------------------

resource "confluent_api_key" "argus_app_kafka" {
  display_name = "argus-app-kafka-key"
  description  = "Kafka API key for Argus application"

  owner {
    id          = confluent_service_account.argus_app.id
    api_version = confluent_service_account.argus_app.api_version
    kind        = confluent_service_account.argus_app.kind
  }

  managed_resource {
    id          = confluent_kafka_cluster.argus.id
    api_version = confluent_kafka_cluster.argus.api_version
    kind        = confluent_kafka_cluster.argus.kind

    environment {
      id = confluent_environment.argus.id
    }
  }
}

# -----------------------------------------------------------------------------
# Role Binding - Grant CloudClusterAdmin to service account
# -----------------------------------------------------------------------------

resource "confluent_role_binding" "argus_app_cluster_admin" {
  principal   = "User:${confluent_service_account.argus_app.id}"
  role_name   = "CloudClusterAdmin"
  crn_pattern = confluent_kafka_cluster.argus.rbac_crn
}

# -----------------------------------------------------------------------------
# Kafka Topics
# -----------------------------------------------------------------------------

locals {
  topics = {
    "argus.test.executed" = {
      partitions = 6
      config = {
        "retention.ms" = "604800000"  # 7 days
      }
    }
    "argus.test.created" = {
      partitions = 6
      config = {
        "retention.ms" = "604800000"
      }
    }
    "argus.test.failed" = {
      partitions = 6
      config = {
        "retention.ms" = "2592000000"  # 30 days (keep failures longer)
      }
    }
    "argus.codebase.ingested" = {
      partitions = 6
      config = {
        "retention.ms" = "604800000"
      }
    }
    "argus.codebase.analyzed" = {
      partitions = 6
      config = {
        "retention.ms" = "604800000"
      }
    }
    "argus.error.reported" = {
      partitions = 6
      config = {
        "retention.ms" = "2592000000"  # 30 days
      }
    }
    "argus.healing.requested" = {
      partitions = 3
      config = {
        "retention.ms" = "604800000"
      }
    }
    "argus.healing.completed" = {
      partitions = 3
      config = {
        "retention.ms" = "604800000"
      }
    }
    "argus.dlq" = {
      partitions = 3
      config = {
        "retention.ms" = "2592000000"  # 30 days (keep DLQ longer)
      }
    }
  }
}

resource "confluent_kafka_topic" "topics" {
  for_each = local.topics

  kafka_cluster {
    id = confluent_kafka_cluster.argus.id
  }

  topic_name       = each.key
  partitions_count = each.value.partitions
  rest_endpoint    = confluent_kafka_cluster.argus.rest_endpoint

  config = each.value.config

  credentials {
    key    = confluent_api_key.argus_app_kafka.id
    secret = confluent_api_key.argus_app_kafka.secret
  }

  depends_on = [confluent_role_binding.argus_app_cluster_admin]
}

# -----------------------------------------------------------------------------
# Flink Compute Pool
# -----------------------------------------------------------------------------

resource "confluent_flink_compute_pool" "argus" {
  display_name = "argus-flink-pool"
  cloud        = var.cloud_provider
  region       = var.region
  max_cfu      = 10  # Max Confluent Flink Units (start small)

  environment {
    id = confluent_environment.argus.id
  }
}

# -----------------------------------------------------------------------------
# Flink API Key
# -----------------------------------------------------------------------------

resource "confluent_api_key" "flink" {
  display_name = "argus-flink-key"
  description  = "API key for Flink compute pool"

  owner {
    id          = confluent_service_account.argus_app.id
    api_version = confluent_service_account.argus_app.api_version
    kind        = confluent_service_account.argus_app.kind
  }

  managed_resource {
    id          = confluent_flink_compute_pool.argus.id
    api_version = confluent_flink_compute_pool.argus.api_version
    kind        = confluent_flink_compute_pool.argus.kind

    environment {
      id = confluent_environment.argus.id
    }
  }
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "environment_id" {
  value       = confluent_environment.argus.id
  description = "Confluent Cloud Environment ID"
}

output "kafka_cluster_id" {
  value       = confluent_kafka_cluster.argus.id
  description = "Kafka Cluster ID"
}

output "kafka_bootstrap_servers" {
  value       = confluent_kafka_cluster.argus.bootstrap_endpoint
  description = "Kafka Bootstrap Servers"
}

output "kafka_rest_endpoint" {
  value       = confluent_kafka_cluster.argus.rest_endpoint
  description = "Kafka REST Endpoint"
}

output "kafka_api_key" {
  value       = confluent_api_key.argus_app_kafka.id
  description = "Kafka API Key ID"
  sensitive   = false
}

output "kafka_api_secret" {
  value       = confluent_api_key.argus_app_kafka.secret
  description = "Kafka API Secret"
  sensitive   = true
}

output "flink_compute_pool_id" {
  value       = confluent_flink_compute_pool.argus.id
  description = "Flink Compute Pool ID"
}

output "flink_api_key" {
  value       = confluent_api_key.flink.id
  description = "Flink API Key ID"
  sensitive   = false
}

output "flink_api_secret" {
  value       = confluent_api_key.flink.secret
  description = "Flink API Secret"
  sensitive   = true
}

output "topics_created" {
  value       = [for t in confluent_kafka_topic.topics : t.topic_name]
  description = "List of created Kafka topics"
}

# Output for .env file
output "env_file_content" {
  value = <<-EOT
    # Confluent Cloud Configuration (Generated by Terraform)
    CONFLUENT_BOOTSTRAP_SERVERS=${confluent_kafka_cluster.argus.bootstrap_endpoint}
    CONFLUENT_KAFKA_API_KEY=${confluent_api_key.argus_app_kafka.id}
    CONFLUENT_KAFKA_API_SECRET=${confluent_api_key.argus_app_kafka.secret}
    CONFLUENT_FLINK_API_KEY=${confluent_api_key.flink.id}
    CONFLUENT_FLINK_API_SECRET=${confluent_api_key.flink.secret}
    CONFLUENT_ENVIRONMENT_ID=${confluent_environment.argus.id}
    CONFLUENT_CLUSTER_ID=${confluent_kafka_cluster.argus.id}
    CONFLUENT_FLINK_POOL_ID=${confluent_flink_compute_pool.argus.id}
  EOT
  description = "Content for .env file"
  sensitive   = true
}

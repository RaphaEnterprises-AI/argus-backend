#!/usr/bin/env python3
"""
Flink Job Submission Script

Submits Flink SQL jobs to the Flink cluster via the REST API.
Supports both session mode (SQL Gateway) and per-job deployment.

Usage:
    # List available jobs
    python scripts/submit_flink_jobs.py --list

    # Submit a specific job
    python scripts/submit_flink_jobs.py --job test_metrics_aggregation

    # Submit all jobs
    python scripts/submit_flink_jobs.py --all

    # Check job status
    python scripts/submit_flink_jobs.py --status

    # Cancel a running job
    python scripts/submit_flink_jobs.py --cancel <job-id>

    # Port-forward to access Flink UI (requires kubectl)
    python scripts/submit_flink_jobs.py --port-forward

Environment Variables:
    FLINK_REST_URL: Flink JobManager REST API URL (default: http://localhost:8081)
    KAFKA_BOOTSTRAP_SERVERS: Kafka/Redpanda bootstrap servers
    KAFKA_SASL_USERNAME: SASL username
    KAFKA_SASL_PASSWORD: SASL password
    KAFKA_SECURITY_PROTOCOL: Security protocol (default: SASL_PLAINTEXT)
    KAFKA_SASL_MECHANISM: SASL mechanism (default: SCRAM-SHA-512)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if available
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    print(f"Loading environment from {ENV_FILE}")
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key not in os.environ:
                    os.environ[key] = value


# Configuration
FLINK_REST_URL = os.getenv("FLINK_REST_URL", "http://localhost:8081")

# Kafka/Redpanda configuration
KAFKA_CONFIG = {
    "KAFKA_BOOTSTRAP_SERVERS": os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        os.getenv("REDPANDA_BROKERS", "redpanda.argus-data.svc.cluster.local:9092")
    ),
    "KAFKA_SECURITY_PROTOCOL": os.getenv(
        "KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT"
    ),
    "KAFKA_SASL_MECHANISM": os.getenv(
        "KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"
    ),
    "KAFKA_SASL_USERNAME": os.getenv(
        "KAFKA_SASL_USERNAME",
        os.getenv("REDPANDA_SASL_USERNAME", "argus-service")
    ),
    "KAFKA_SASL_PASSWORD": os.getenv(
        "KAFKA_SASL_PASSWORD",
        os.getenv("REDPANDA_SASL_PASSWORD", "")
    ),
}


class FlinkJobManager:
    """Manages Flink job submission and monitoring via REST API."""

    def __init__(self, rest_url: str = FLINK_REST_URL):
        self.rest_url = rest_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self.jobs_dir = PROJECT_ROOT / "src" / "streaming" / "jobs"

    def _api_call(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API call to Flink REST API."""
        url = urljoin(self.rest_url + "/", endpoint.lstrip("/"))
        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.ConnectError as e:
            print(f"Connection Error: Could not connect to {url}")
            print("Hint: Run with --port-forward to set up kubectl port forwarding")
            raise

    def check_cluster_health(self) -> bool:
        """Check if Flink cluster is healthy."""
        try:
            overview = self._api_call("GET", "/overview")
            print(f"Flink Cluster Status:")
            print(f"  Version: {overview.get('flink-version', 'unknown')}")
            print(f"  Task Managers: {overview.get('taskmanagers', 0)}")
            print(f"  Task Slots Total: {overview.get('slots-total', 0)}")
            print(f"  Task Slots Available: {overview.get('slots-available', 0)}")
            print(f"  Jobs Running: {overview.get('jobs-running', 0)}")
            print(f"  Jobs Finished: {overview.get('jobs-finished', 0)}")
            print(f"  Jobs Cancelled: {overview.get('jobs-cancelled', 0)}")
            print(f"  Jobs Failed: {overview.get('jobs-failed', 0)}")
            return overview.get("taskmanagers", 0) > 0
        except Exception as e:
            print(f"Failed to check cluster health: {e}")
            return False

    def list_jobs(self) -> list[dict]:
        """List all jobs (running and completed)."""
        try:
            result = self._api_call("GET", "/jobs/overview")
            return result.get("jobs", [])
        except Exception as e:
            print(f"Failed to list jobs: {e}")
            return []

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get status of a specific job."""
        try:
            return self._api_call("GET", f"/jobs/{job_id}")
        except Exception:
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            self._api_call("PATCH", f"/jobs/{job_id}?mode=cancel")
            print(f"Job {job_id} cancellation requested")
            return True
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")
            return False

    def load_sql_job(self, job_name: str) -> str:
        """Load SQL job file and substitute environment variables."""
        sql_file = self.jobs_dir / f"{job_name}.sql"
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL job file not found: {sql_file}")

        sql_content = sql_file.read_text()

        # Substitute environment variables
        # Format: ${VAR_NAME} or ${VAR_NAME:default_value}
        def replace_var(match):
            var_expr = match.group(1)
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
            else:
                var_name = var_expr
                default_value = ""

            value = KAFKA_CONFIG.get(var_name, os.getenv(var_name, default_value))
            if not value and var_name in ["KAFKA_SASL_PASSWORD"]:
                print(f"Warning: {var_name} is not set")
            return value

        sql_content = re.sub(r"\$\{([^}]+)\}", replace_var, sql_content)
        return sql_content

    def list_available_jobs(self) -> list[str]:
        """List available SQL job files."""
        jobs = []
        if self.jobs_dir.exists():
            for sql_file in self.jobs_dir.glob("*.sql"):
                jobs.append(sql_file.stem)
        return jobs

    def submit_sql_job(self, job_name: str, dry_run: bool = False) -> Optional[str]:
        """
        Submit a SQL job to Flink.

        Note: Direct SQL submission requires Flink SQL Gateway.
        This method generates the SQL and provides instructions for manual submission.
        """
        print(f"\nPreparing job: {job_name}")
        print("-" * 50)

        try:
            sql_content = self.load_sql_job(job_name)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

        if dry_run:
            print("\n[DRY RUN] Generated SQL:")
            print("=" * 60)
            print(sql_content)
            print("=" * 60)
            return "dry-run"

        # Save to temp file for manual submission
        output_file = PROJECT_ROOT / "tmp" / f"{job_name}_generated.sql"
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(sql_content)
        print(f"Generated SQL saved to: {output_file}")

        # Provide submission instructions
        print("\nTo submit this job to Flink:")
        print("=" * 60)
        print("\nOption 1: Using Flink SQL Client (recommended)")
        print("  1. Port-forward to Flink cluster:")
        print("     kubectl port-forward svc/flink-rest -n argus-data 8081:8081")
        print("  2. Start SQL client:")
        print(f"     docker run -it --rm --network host flink:1.20-java17 sql-client.sh")
        print("  3. Execute the SQL file or paste statements")
        print("\nOption 2: Using kubectl exec")
        print("  kubectl exec -it -n argus-data deploy/argus-flink-jobmanager -- \\")
        print(f"    /opt/flink/bin/sql-client.sh -f /tmp/{job_name}.sql")
        print("\nOption 3: Apply as FlinkSessionJob (Kubernetes native)")
        print("  Update data-layer/kubernetes/flink-jobs/test-analytics.yaml")
        print("  kubectl apply -f data-layer/kubernetes/flink-jobs/test-analytics.yaml")
        print("=" * 60)

        return output_file.as_posix()

    def submit_all_jobs(self, dry_run: bool = False) -> dict[str, Optional[str]]:
        """Submit all available SQL jobs."""
        results = {}
        for job_name in self.list_available_jobs():
            results[job_name] = self.submit_sql_job(job_name, dry_run=dry_run)
        return results


def setup_port_forward():
    """Set up kubectl port forwarding to Flink REST API."""
    print("Setting up kubectl port-forward to Flink cluster...")
    print("Press Ctrl+C to stop port forwarding\n")

    try:
        # Check if kubectl is available
        subprocess.run(
            ["kubectl", "version", "--client"],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: kubectl is not installed or not in PATH")
        return False

    try:
        # Start port forwarding
        proc = subprocess.Popen(
            [
                "kubectl", "port-forward",
                "-n", "argus-data",
                "svc/flink-rest",
                "8081:8081"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a moment and check if process is still running
        time.sleep(2)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            print(f"Port forwarding failed: {stderr.decode()}")
            return False

        print("Port forwarding established: http://localhost:8081")
        print("Flink Web UI: http://localhost:8081")
        print("\nPress Ctrl+C to stop...")

        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nPort forwarding stopped")

        return True

    except Exception as e:
        print(f"Error setting up port forwarding: {e}")
        return False


def print_job_status(jobs: list[dict]):
    """Print formatted job status table."""
    if not jobs:
        print("No jobs found")
        return

    print("\nFlink Jobs:")
    print("-" * 100)
    print(f"{'Job ID':<40} {'Name':<30} {'Status':<12} {'Start Time':<20}")
    print("-" * 100)

    for job in jobs:
        job_id = job.get("jid", "unknown")[:38]
        name = job.get("name", "unknown")[:28]
        status = job.get("state", "unknown")
        start_time = job.get("start-time", 0)
        if start_time:
            from datetime import datetime
            start_time = datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "-"
        print(f"{job_id:<40} {name:<30} {status:<12} {start_time:<20}")
    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Flink SQL jobs to the cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--list", action="store_true", help="List available SQL jobs")
    parser.add_argument("--job", type=str, help="Submit a specific job by name")
    parser.add_argument("--all", action="store_true", help="Submit all available jobs")
    parser.add_argument("--status", action="store_true", help="Show running jobs status")
    parser.add_argument("--cancel", type=str, metavar="JOB_ID", help="Cancel a running job")
    parser.add_argument("--port-forward", action="store_true", help="Set up kubectl port forwarding")
    parser.add_argument("--dry-run", action="store_true", help="Generate SQL without submitting")
    parser.add_argument("--url", type=str, default=FLINK_REST_URL, help="Flink REST API URL")

    args = parser.parse_args()

    # Handle port forwarding first (blocking)
    if args.port_forward:
        setup_port_forward()
        return

    manager = FlinkJobManager(rest_url=args.url)

    # List available jobs
    if args.list:
        print("\nAvailable SQL Jobs:")
        print("-" * 40)
        for job_name in manager.list_available_jobs():
            sql_file = manager.jobs_dir / f"{job_name}.sql"
            size = sql_file.stat().st_size if sql_file.exists() else 0
            print(f"  - {job_name} ({size:,} bytes)")
        print("-" * 40)
        return

    # Check cluster health first for most operations
    if args.status or args.job or args.all or args.cancel:
        print(f"\nConnecting to Flink cluster at {args.url}...")
        if not manager.check_cluster_health():
            print("\nWarning: Cluster may not be healthy")
            if not args.dry_run:
                print("Use --port-forward to set up access to the cluster")

    # Show status
    if args.status:
        jobs = manager.list_jobs()
        print_job_status(jobs)
        return

    # Cancel job
    if args.cancel:
        manager.cancel_job(args.cancel)
        return

    # Submit specific job
    if args.job:
        result = manager.submit_sql_job(args.job, dry_run=args.dry_run)
        if result:
            print(f"\nJob preparation complete: {args.job}")
        return

    # Submit all jobs
    if args.all:
        results = manager.submit_all_jobs(dry_run=args.dry_run)
        print("\n" + "=" * 60)
        print("Job Preparation Summary:")
        print("=" * 60)
        for job_name, result in results.items():
            status = "OK" if result else "FAILED"
            print(f"  {job_name}: {status}")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()

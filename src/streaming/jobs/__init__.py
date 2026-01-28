# Flink SQL Job Definitions
# These SQL files define streaming analytics jobs for the Argus platform
#
# Jobs are submitted to the Flink cluster via the REST API or SQL Gateway.
# See scripts/submit_flink_jobs.py for the submission mechanism.

from pathlib import Path

# Directory containing SQL job files
JOBS_DIR = Path(__file__).parent

# Available job definitions
AVAILABLE_JOBS = {
    "test_metrics_aggregation": JOBS_DIR / "test_metrics_aggregation.sql",
    "anomaly_detection": JOBS_DIR / "anomaly_detection.sql",
}


def get_job_sql(job_name: str) -> str:
    """Load SQL content for a job by name."""
    if job_name not in AVAILABLE_JOBS:
        raise ValueError(f"Unknown job: {job_name}. Available: {list(AVAILABLE_JOBS.keys())}")

    sql_file = AVAILABLE_JOBS[job_name]
    if not sql_file.exists():
        raise FileNotFoundError(f"Job SQL file not found: {sql_file}")

    return sql_file.read_text()


def list_jobs() -> list[str]:
    """List all available job names."""
    return list(AVAILABLE_JOBS.keys())

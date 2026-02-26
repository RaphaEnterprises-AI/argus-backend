"""
Argus Stream Processor — replaces 4 Flink SQL jobs with a single Python process.

Jobs reimplemented:
  1. test-metrics-aggregation: 5-min tumbling window, COUNT/AVG/P95 by org+project
  2. anomaly-latency: 5-min window, alert if AVG > 5s or P95 > 15s per test_id
  3. anomaly-failures: 15-min window, trigger healing if 3+ failures per test_id
  4. failure-cluster-update: 15-min window, cluster failures by fingerprint → Supabase + Kafka

Design:
  - Single aiokafka consumer subscribed to argus.test.executed + argus.test.failed
  - In-memory TumblingWindowManager per job
  - Flush expired windows every second → produce to output Kafka topics + upsert Supabase
  - Idempotency keys prevent duplicate writes on restart
  - Health check on port 8080
"""

import asyncio
import json
import logging
import os
import signal
import ssl
import time
from datetime import datetime, timezone

from aiohttp import web

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stream-processor")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
KAFKA_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")
KAFKA_SECURITY = os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
KAFKA_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
KAFKA_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "argus-stream-processor")
KAFKA_AUTO_OFFSET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
PROCESSING_REGION = os.getenv("PROCESSING_REGION", "iad")
HEALTH_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))

# Input topics
TOPIC_EXECUTED = "argus.test.executed"
TOPIC_FAILED = "argus.test.failed"

# Output topics
TOPIC_METRICS_5M = "argus.metrics.test-summary-5m"
TOPIC_LATENCY_ALERT = "argus.alerts.latency-spike"
TOPIC_HEALING_REQ = "argus.healing.requested"
TOPIC_FAILURE_CLUSTER = "argus.patterns.failure-cluster"

# Window durations
WINDOW_5M = 300  # seconds
WINDOW_15M = 900

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_running = True
_healthy = False
_connecting = True  # True while attempting Kafka connection
_stats = {
    "events_consumed": 0,
    "windows_flushed": 0,
    "errors": 0,
    "last_event_time": None,
    "start_time": time.time(),
}


def _shutdown(sig, _frame):
    global _running
    log.info("Received %s, shutting down...", signal.Signals(sig).name)
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------
def _kafka_ssl_context() -> ssl.SSLContext | None:
    if "SSL" in KAFKA_SECURITY:
        ctx = ssl.create_default_context()
        return ctx
    return None


def _kafka_sasl_kwargs() -> dict:
    kw: dict = {}
    if "SASL" in KAFKA_SECURITY:
        kw["sasl_mechanism"] = KAFKA_MECHANISM
        kw["sasl_plain_username"] = KAFKA_USERNAME
        kw["sasl_plain_password"] = KAFKA_PASSWORD
        kw["security_protocol"] = KAFKA_SECURITY
    return kw


# ---------------------------------------------------------------------------
# Supabase client (lazy init)
# ---------------------------------------------------------------------------
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client

        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


# ---------------------------------------------------------------------------
# Health check server
# ---------------------------------------------------------------------------
async def _health_handler(_request):
    if _healthy:
        return web.json_response(
            {
                "status": "healthy",
                "uptime_seconds": int(time.time() - _stats["start_time"]),
                "events_consumed": _stats["events_consumed"],
                "windows_flushed": _stats["windows_flushed"],
                "errors": _stats["errors"],
                "last_event_time": _stats["last_event_time"],
            }
        )
    # Return 200 while connecting so Fly doesn't kill us during retries
    if _connecting:
        return web.json_response({"status": "connecting_to_kafka"})
    return web.json_response({"status": "starting"}, status=503)


async def _start_health_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HEALTH_PORT)
    await site.start()
    log.info("Health server listening on :%d", HEALTH_PORT)
    return runner


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
async def run():
    global _healthy, _connecting

    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    from windows import (
        FailureClusterState,
        FailureWindowState,
        LatencyWindowState,
        MetricsWindowState,
        TumblingWindowManager,
    )

    # Start health server first so Fly sees us alive during Kafka connect retries
    health_runner = await _start_health_server()

    ssl_ctx = _kafka_ssl_context()
    sasl_kw = _kafka_sasl_kwargs()

    def _make_consumer():
        return AIOKafkaConsumer(
            TOPIC_EXECUTED,
            TOPIC_FAILED,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=KAFKA_GROUP,
            auto_offset_reset=KAFKA_AUTO_OFFSET,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda v: json.loads(v) if v else {},
            ssl_context=ssl_ctx,
            **sasl_kw,
        )

    def _make_producer():
        return AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
            ssl_context=ssl_ctx,
            **sasl_kw,
        )

    # Retry Kafka connection with exponential backoff
    MAX_CONNECT_RETRIES = 10
    consumer = None
    producer = None
    for attempt in range(1, MAX_CONNECT_RETRIES + 1):
        if not _running:
            return
        try:
            log.info("Connecting to Kafka (attempt %d/%d) bootstrap=%s",
                     attempt, MAX_CONNECT_RETRIES, KAFKA_BOOTSTRAP)
            consumer = _make_consumer()
            producer = _make_producer()
            await consumer.start()
            await producer.start()
            log.info("Kafka connected successfully")
            break
        except Exception as e:
            log.warning("Kafka connect attempt %d failed: %s", attempt, e)
            # Clean up failed connections
            if consumer:
                try:
                    await consumer.stop()
                except Exception:
                    pass
                consumer = None
            if producer:
                try:
                    await producer.stop()
                except Exception:
                    pass
                producer = None
            if attempt == MAX_CONNECT_RETRIES:
                log.error("Failed to connect to Kafka after %d attempts", MAX_CONNECT_RETRIES)
                await health_runner.cleanup()
                raise
            backoff = min(2 ** attempt, 30)
            log.info("Retrying in %ds...", backoff)
            await asyncio.sleep(backoff)

    _connecting = False

    # Window managers for each job
    metrics_wm = TumblingWindowManager(WINDOW_5M)     # Job 1
    latency_wm = TumblingWindowManager(WINDOW_5M)     # Job 2
    failures_wm = TumblingWindowManager(WINDOW_15M)   # Job 3
    clusters_wm = TumblingWindowManager(WINDOW_15M)   # Job 4

    try:
        _healthy = True
        log.info(
            "Stream processor started — consuming [%s, %s], group=%s",
            TOPIC_EXECUTED, TOPIC_FAILED, KAFKA_GROUP,
        )

        while _running:
            # Poll for messages (100ms timeout to allow frequent window flushes)
            batch = await consumer.getmany(timeout_ms=100, max_records=500)

            now = time.time()

            for _tp, messages in batch.items():
                for msg in messages:
                    try:
                        evt = msg.value
                        if not evt or not isinstance(evt, dict):
                            continue

                        org_id = evt.get("org_id")
                        project_id = evt.get("project_id")
                        if not org_id or not project_id:
                            continue

                        _stats["events_consumed"] += 1
                        _stats["last_event_time"] = datetime.now(timezone.utc).isoformat()

                        # Parse event timestamp (fall back to now)
                        evt_time = now
                        ts_str = evt.get("timestamp") or evt.get("event_time")
                        if ts_str:
                            try:
                                evt_time = datetime.fromisoformat(ts_str).timestamp()
                            except (ValueError, TypeError):
                                pass

                        test_id = evt.get("test_id", "")
                        test_name = evt.get("test_name", "")
                        status = evt.get("status", "")
                        duration_ms = int(evt.get("duration_ms", 0) or 0)

                        topic = msg.topic

                        # ----- Job 1: Test metrics aggregation (5 min) -----
                        if topic == TOPIC_EXECUTED:
                            ws, we = metrics_wm.window_bounds(evt_time)
                            key = (int(ws), org_id, project_id)
                            state = metrics_wm.get_or_create(
                                key,
                                lambda _ws=ws, _we=we: MetricsWindowState(
                                    org_id=org_id, project_id=project_id,
                                    window_start=_ws, window_end=_we,
                                ),
                            )
                            state.add(status, duration_ms)

                        # ----- Job 2: Latency anomaly detection (5 min) -----
                        if topic == TOPIC_EXECUTED and status in ("passed", "failed") and duration_ms > 0:
                            ws, we = latency_wm.window_bounds(evt_time)
                            key = (int(ws), org_id, project_id, test_id)
                            state = latency_wm.get_or_create(
                                key,
                                lambda _ws=ws, _we=we: LatencyWindowState(
                                    org_id=org_id, project_id=project_id,
                                    test_id=test_id, test_name=test_name,
                                    window_start=_ws, window_end=_we,
                                ),
                            )
                            state.add(duration_ms, test_name)

                        # ----- Job 3: Failure spike → healing request (15 min) -----
                        if topic == TOPIC_EXECUTED and status == "failed" and test_id:
                            ws, we = failures_wm.window_bounds(evt_time)
                            key = (int(ws), org_id, project_id, test_id)
                            state = failures_wm.get_or_create(
                                key,
                                lambda _ws=ws, _we=we: FailureWindowState(
                                    org_id=org_id, project_id=project_id,
                                    test_id=test_id, test_name=test_name,
                                    window_start=_ws, window_end=_we,
                                ),
                            )
                            state.add(
                                evt.get("error_message", ""),
                                evt.get("error_type", ""),
                                evt.get("selector", ""),
                                test_name,
                            )

                        # ----- Job 4: Failure cluster (15 min, from argus.test.failed) -----
                        if topic == TOPIC_FAILED and test_id:
                            ws, we = clusters_wm.window_bounds(evt_time)
                            key = (int(ws), org_id, project_id, test_id)
                            state = clusters_wm.get_or_create(
                                key,
                                lambda _ws=ws, _we=we: FailureClusterState(
                                    org_id=org_id, project_id=project_id,
                                    test_id=test_id,
                                    window_start=_ws, window_end=_we,
                                ),
                            )
                            state.add(
                                evt.get("error_message", ""),
                                evt.get("selector", ""),
                            )

                    except Exception:
                        _stats["errors"] += 1
                        log.exception("Error processing event")

            # ----- Flush expired windows -----
            await _flush_windows(
                producer,
                metrics_wm, latency_wm, failures_wm, clusters_wm,
                now,
            )

        # Final flush before shutdown
        log.info("Final flush before shutdown...")
        await _flush_windows(
            producer,
            metrics_wm, latency_wm, failures_wm, clusters_wm,
            time.time() + 86400,  # force flush all
        )

    except Exception:
        log.exception("Fatal error in stream processor")
        raise
    finally:
        _healthy = False
        await consumer.stop()
        await producer.stop()
        await health_runner.cleanup()
        log.info(
            "Shutdown complete — consumed=%d flushed=%d errors=%d",
            _stats["events_consumed"], _stats["windows_flushed"], _stats["errors"],
        )


async def _flush_windows(producer, metrics_wm, latency_wm, failures_wm, clusters_wm, now):
    """Flush all expired windows and produce/write output."""

    # Job 1: Metrics → Kafka
    for state in metrics_wm.flush_expired(now):
        try:
            await producer.send_and_wait(TOPIC_METRICS_5M, state.to_kafka_dict())
            _stats["windows_flushed"] += 1
            log.debug("Flushed metrics window org=%s proj=%s total=%d",
                       state.org_id, state.project_id, state.total)
        except Exception:
            _stats["errors"] += 1
            log.exception("Error flushing metrics window")

    # Job 2: Latency alerts → Kafka (only anomalous)
    for state in latency_wm.flush_expired(now):
        if not state.is_anomalous:
            continue
        try:
            await producer.send_and_wait(TOPIC_LATENCY_ALERT, state.to_kafka_dict())
            _stats["windows_flushed"] += 1
            log.info("Latency alert: test=%s avg=%.0fms p95=%dms severity=%s",
                      state.test_id, state.avg_duration_ms,
                      state.p95_duration_ms, state.severity)
        except Exception:
            _stats["errors"] += 1
            log.exception("Error flushing latency alert")

    # Job 3: Failure healing requests → Kafka (only 3+ failures)
    for state in failures_wm.flush_expired(now):
        if not state.should_heal:
            continue
        try:
            await producer.send_and_wait(TOPIC_HEALING_REQ, state.to_kafka_dict())
            _stats["windows_flushed"] += 1
            log.info("Healing request: test=%s failures=%d priority=%s",
                      state.test_id, state.failure_count, state.priority)
        except Exception:
            _stats["errors"] += 1
            log.exception("Error flushing healing request")

    # Job 4: Failure clusters → Supabase + Kafka
    for state in clusters_wm.flush_expired(now):
        try:
            data = state.to_supabase_dict(PROCESSING_REGION)

            # Write to Supabase with idempotency
            sb = _get_supabase()
            sb.table("flink_failure_patterns").upsert(
                data,
                on_conflict="idempotency_key",
                ignore_duplicates=True,
            ).execute()

            # Also produce to Kafka for real-time consumers
            await producer.send_and_wait(
                TOPIC_FAILURE_CLUSTER,
                state.to_kafka_dict(PROCESSING_REGION),
            )
            _stats["windows_flushed"] += 1
            log.debug("Flushed failure cluster: test=%s count=%d",
                       state.test_id, state.failure_count)
        except Exception:
            _stats["errors"] += 1
            log.exception("Error flushing failure cluster")


if __name__ == "__main__":
    asyncio.run(run())

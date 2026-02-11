# Testing the Autonomous Healing Loop

## Quick Start - Manual Testing

### 1. Test Event Emission (Local Development)

First, verify that test failures emit healing events:

```python
# test_healing_trigger.py
import asyncio
from src.orchestrator.nodes import _emit_test_events
from src.orchestrator.state import TestResult, TestStatus

async def test_healing_trigger():
    """Test that HEALING_REQUESTED is emitted on failure."""
    
    state = {
        "org_id": "org-test",
        "project_id": "proj-test", 
        "user_id": "user-test",
        "run_id": "run-123",
        "app_url": "https://example.com",
    }
    
    test = {
        "id": "test-login",
        "name": "Login Test",
        "type": "ui",
        "priority": "critical",
        "steps": [
            {"action": "goto", "target": "/login"},
            {"action": "fill", "target": "#email", "value": "test@example.com"},
            {"action": "click", "target": "#submit-btn"},
        ],
    }
    
    # Simulate a failed test
    test_result = TestResult(
        status=TestStatus.FAILED,
        duration_seconds=5.0,
        error_message="Element not found: #submit-btn",
        actions_taken=[
            {"step": 1, "action": "goto", "result": "success"},
            {"step": 3, "action": "click", "target": "#submit-btn", "result": "failure", "error": "Element not found"},
        ],
        screenshots=[],
        assertions_passed=0,
        assertions_failed=1,
    )
    
    # Mock logger
    class MockLog:
        def debug(self, msg, **kwargs): print(f"DEBUG: {msg}")
        def info(self, msg, **kwargs): print(f"INFO: {msg}")
        def warning(self, msg, **kwargs): print(f"WARN: {msg}")
    
    log = MockLog()
    
    # This should emit events to Kafka
    await _emit_test_events(state, test, test_result, log)
    print("✅ Event emission test passed!")

if __name__ == "__main__":
    asyncio.run(test_healing_trigger())
```

Run it:
```bash
cd /Users/bvk/Downloads/e2e-testing-agent
python test_healing_trigger.py
```

### 2. Test HealingConsumer Directly

```python
# test_healing_consumer.py
import asyncio
from src.workers.healing_consumer import HealingConsumer, HealingEvent, HealingConfig

async def test_healing_consumer():
    """Test HealingConsumer processes events correctly."""
    
    config = HealingConfig(
        bootstrap_servers="localhost:9092",  # Change to your Redpanda
        supabase_url="your-supabase-url",
        supabase_service_key="your-service-key",
    )
    
    consumer = HealingConsumer(config)
    
    # Create a test event
    event = HealingEvent(
        event_id="test-123",
        failure_id="failure-456",
        test_id="your-test-id-here",  # Replace with real test ID
        org_id="your-org-id",
        project_id="your-project-id",
        error_type="selector",
        error_message="Element not found: #submit-btn",
        failed_selector="#submit-btn",
        page_url="https://your-app.com",
        screenshot_url=None,
        strategy="auto",
        priority="high",
    )
    
    # Process the event
    print("Processing healing event...")
    result = await consumer._process_event(event.dict())
    print(f"Processing result: {result}")
    
if __name__ == "__main__":
    asyncio.run(test_healing_consumer())
```

### 3. End-to-End Integration Test

Create a test that actually fails and verify the full loop:

```bash
#!/bin/bash
# test_autonomous_loop.sh

echo "🧪 Testing Autonomous Healing Loop"
echo "=================================="

# Step 1: Create a test with a broken selector
echo "Step 1: Creating test with broken selector..."
curl -X POST https://skopaq-brain-production.up.railway.app/api/v1/tests \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "project_id": "YOUR_PROJECT_ID",
    "name": "Autonomous Healing Test",
    "spec": {
      "steps": [
        {"action": "goto", "target": "https://your-app.com/login"},
        {"action": "click", "target": "#this-selector-does-not-exist"}
      ]
    }
  }'

# Step 2: Run the test (it will fail)
echo -e "\nStep 2: Running test (expecting failure)..."
curl -X POST https://skopaq-brain-production.up.railway.app/api/v1/tests/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "project_id": "YOUR_PROJECT_ID",
    "test_ids": ["TEST_ID_FROM_STEP_1"]
  }'

# Step 3: Check Kafka for healing events
echo -e "\nStep 3: Checking for healing events..."
# You'll need to check your Redpanda/Kafka logs for:
# - "Emitted HEALING_REQUESTED event"
# - "HealingConsumer started"
# - "Processing healing event"

# Step 4: Wait and check if test was healed
echo -e "\nStep 4: Waiting 30 seconds for healing..."
sleep 30

# Step 5: Check if test spec was updated
echo -e "\nStep 5: Checking if test was healed..."
curl https://skopaq-brain-production.up.railway.app/api/v1/tests/TEST_ID_FROM_STEP_1 \
  -H "X-API-Key: YOUR_API_KEY"

echo -e "\n✅ Test complete!"
```

## Testing Checklist

### Pre-Deployment Tests (Local)
- [ ] Run unit tests: `pytest tests/workers/test_healing_consumer.py -v`
- [ ] Test event emission works
- [ ] Verify SelfHealerAgent can heal a test manually
- [ ] Check Cognee integration (store/retrieve patterns)

### Post-Deployment Tests (Railway)
- [ ] Verify HealingConsumer started in logs
- [ ] Create a test with intentional failure
- [ ] Run test and verify HEALING_REQUESTED event emitted
- [ ] Check that HealingConsumer processes the event
- [ ] Verify test spec is updated after healing
- [ ] Check Cognee for stored healing pattern

### Monitoring in Production

Watch these logs in Railway:
```
# Successful flow:
1. "Emitted HEALING_REQUESTED event - autonomous healing triggered"
2. "HealingConsumer started"
3. "Processing healing event"
4. "Found cached healing pattern" OR "Code-aware healing succeeded"
5. "Updated test spec"
6. "Stored healing pattern"
7. "Sent healing completion event"

# Error conditions:
- "Failed to emit healing request" (Kafka not running)
- "Test spec not found" (test deleted during healing)
- "Healing processing failed" (agent couldn't fix)
```

## Debugging Commands

```bash
# Check if Kafka/Redpanda is receiving events
kubectl exec -n argus-data redpanda-0 -- rpk topic consume argus.healing.requested

# Check HealingConsumer logs
railway logs --service argus-backend | grep -i healing

# Check Cognee patterns
# Query Supabase:
SELECT * FROM graph_healing_pattern_vertices ORDER BY created_at DESC LIMIT 10;

# Test SelfHealerAgent manually
python -c "
from src.agents.self_healer import SelfHealerAgent
import asyncio

async def test():
    healer = SelfHealerAgent(org_id='test', project_id='test')
    result = await healer.heal_test(
        test_spec={'name': 'Test', 'steps': []},
        failure_details={},
        error_message='Element not found: #btn',
        error_type='selector'
    )
    print(result)

asyncio.run(test())
"
```

## Expected Behavior

### When a Test Fails:
1. **Immediately**: TEST_FAILED event emitted
2. **Immediately**: HEALING_REQUESTED event emitted  
3. **Within seconds**: HealingConsumer picks up event
4. **Within 10-30s**: SelfHealerAgent analyzes and attempts fix
5. **On success**: Test spec updated, pattern stored
6. **Always**: HEALING_COMPLETED event emitted

### Success Metrics:
- Healing requested → completed latency: < 30s
- Auto-heal success rate: > 70% for selector changes
- Pattern storage: Every successful heal stored in Cognee
- Zero manual intervention for known failure patterns

## Troubleshooting

### Issue: No HEALING_REQUESTED events
**Check:**
- Event gateway is running: Check `emit_healing_requested` in logs
- Kafka/Redpanda connection: `KAFKA_BOOTSTRAP_SERVERS` env var
- Test has `org_id` set in state

### Issue: HealingConsumer not processing
**Check:**
- Consumer started: Look for "HealingConsumer started" in startup logs
- Kafka connection: Consumer can connect to `argus.healing.requested` topic
- No consumer group lag: Check with `rpk group describe argus-healing-workers`

### Issue: Healing fails every time
**Check:**
- SelfHealerAgent can access test spec
- Cognee is configured: `COGNEE_API_URL` env var
- Git repo accessible for code-aware healing
- LLM API keys valid (Claude/OpenRouter)

## Load Testing

Test with multiple concurrent failures:
```python
import asyncio
import aiohttp

async def trigger_multiple_failures():
    """Trigger 10 test failures simultaneously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(10):
            task = session.post(
                "https://skopaq-brain-production.up.railway.app/api/v1/tests/run",
                headers={"X-API-Key": "YOUR_KEY"},
                json={"project_id": "YOUR_PROJECT", "test_ids": [f"broken-test-{i}"]}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Triggered {len(responses)} failures")

asyncio.run(trigger_multiple_failures())
```

All 10 should be processed concurrently by the HealingConsumer!

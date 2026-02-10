# Commit Impact Predictor - Technical Design Document

## Executive Summary

The Commit Impact Predictor is Skopaq's flagship AI capability that analyzes code changes at commit/PR time and predicts:
1. **Which tests will likely fail** (with confidence scores)
2. **What production issues might occur** (based on historical patterns)
3. **Security vulnerabilities introduced** (DevSecOps integration)
4. **Performance/reliability risks** (AIOps integration)
5. **Suggested mitigations** (proactive recommendations)

This transforms Skopaq from a testing tool into a **predictive quality intelligence platform**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMMIT IMPACT PREDICTOR                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   INGEST     │───▶│   ANALYZE    │───▶│   PREDICT    │───▶│  SUGGEST  │ │
│  │  (Webhooks)  │    │ (AI Models)  │    │ (ML Models)  │    │  (AI)     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                    │                    │                  │       │
│        ▼                    ▼                    ▼                  ▼       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SDLC CORRELATION ENGINE                           │   │
│  │  (sdlc_events + event_correlations + correlation_insights)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  TEST IMPACT  │           │   SECURITY    │           │    AIOPS      │
│   PREDICTOR   │           │   ANALYZER    │           │   PREDICTOR   │
├───────────────┤           ├───────────────┤           ├───────────────┤
│ • Test mapping│           │ • SAST/DAST   │           │ • Performance │
│ • Flaky tests │           │ • Dependency  │           │ • Reliability │
│ • Coverage    │           │ • Secrets     │           │ • Capacity    │
│ • Regression  │           │ • OWASP       │           │ • Cost        │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## Data Flow

### 1. Ingest Phase (Webhooks)

```
GitHub/GitLab Push Event
         │
         ▼
┌─────────────────────────────────────┐
│  Webhook Handler                     │
│  • Parse commit diff                 │
│  • Extract changed files             │
│  • Identify affected components      │
│  • Fetch PR context (if applicable)  │
└─────────────────────────────────────┘
         │
         ▼
    sdlc_events (event_type='commit')
```

### 2. Analyze Phase (AI Models)

```
Commit Data
    │
    ├──▶ Code Change Analyzer
    │    • AST parsing for semantic changes
    │    • Dependency graph updates
    │    • API contract changes
    │    • Database schema changes
    │
    ├──▶ Historical Pattern Matcher
    │    • Similar past commits
    │    • Failure patterns
    │    • Author patterns
    │    • Time-based patterns
    │
    └──▶ Security Scanner
         • Static analysis (SAST)
         • Dependency vulnerabilities
         • Secrets detection
         • OWASP checks
```

### 3. Predict Phase (ML Models)

```
Analysis Results
    │
    ├──▶ Test Failure Predictor
    │    • P(test_fail | commit_features)
    │    • Uses: file→test mapping, flaky history, coverage data
    │
    ├──▶ Production Risk Predictor
    │    • P(incident | commit_features)
    │    • Uses: Sentry errors, PagerDuty incidents, deploy history
    │
    └──▶ Performance Impact Predictor
         • P(degradation | commit_features)
         • Uses: APM data, load test history, capacity metrics
```

### 4. Suggest Phase (AI Recommendations)

```
Predictions
    │
    ▼
┌─────────────────────────────────────┐
│  Recommendation Engine (Claude)      │
│  • Prioritized test suggestions      │
│  • Code review focus areas           │
│  • Rollback risk assessment          │
│  • Deployment strategy (canary %)    │
│  • Mitigation actions                │
└─────────────────────────────────────┘
    │
    ▼
PR Comment / Slack Notification / Dashboard
```

---

## Core Components

### 1. Test Impact Graph

Maps code files to tests that exercise them.

```sql
CREATE TABLE test_impact_graph (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),

    -- Source code mapping
    file_path TEXT NOT NULL,
    function_name TEXT,
    class_name TEXT,

    -- Test mapping
    test_id UUID REFERENCES tests(id),
    test_file_path TEXT NOT NULL,
    test_name TEXT NOT NULL,

    -- Relationship strength
    impact_score DECIMAL(3,2) DEFAULT 1.0,  -- 0.0-1.0, higher = stronger connection
    relationship_type TEXT CHECK (relationship_type IN (
        'direct',       -- Test directly imports/uses the file
        'transitive',   -- Test uses something that uses the file
        'coverage',     -- Determined by code coverage data
        'historical',   -- Determined by past co-failures
        'semantic'      -- AI-inferred semantic relationship
    )),

    -- Metadata
    last_verified_at TIMESTAMPTZ,
    confidence DECIMAL(3,2) DEFAULT 1.0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, file_path, test_id)
);
```

### 2. Failure Pattern Store

Stores learned patterns from historical failures.

```sql
CREATE TABLE failure_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),

    -- Pattern identification
    pattern_type TEXT NOT NULL CHECK (pattern_type IN (
        'file_change',      -- Specific file changes cause failures
        'author',           -- Specific authors have higher failure rates (temporary)
        'time_of_day',      -- Deployments at certain times fail more
        'dependency',       -- Dependency updates cause issues
        'size',             -- Large commits have higher failure rates
        'component',        -- Specific components are fragile
        'combination'       -- Specific file combinations cause issues
    )),

    -- Pattern definition (JSONB for flexibility)
    pattern_definition JSONB NOT NULL,
    -- Example: {"files": ["src/auth/*"], "day_of_week": [5, 6]}

    -- Historical evidence
    occurrences INTEGER DEFAULT 1,
    last_occurrence TIMESTAMPTZ,
    related_events UUID[],  -- References to sdlc_events

    -- Prediction accuracy
    predictions_made INTEGER DEFAULT 0,
    predictions_correct INTEGER DEFAULT 0,
    accuracy DECIMAL(3,2) GENERATED ALWAYS AS (
        CASE WHEN predictions_made > 0
        THEN predictions_correct::DECIMAL / predictions_made
        ELSE 0 END
    ) STORED,

    -- Pattern status
    is_active BOOLEAN DEFAULT true,
    confidence DECIMAL(3,2) DEFAULT 0.5,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3. Commit Analysis Result

Stores the analysis and predictions for each commit.

```sql
CREATE TABLE commit_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    user_id TEXT NOT NULL,

    -- Commit identification
    commit_sha TEXT NOT NULL,
    pr_number INTEGER,
    branch_name TEXT,

    -- Analysis metadata
    analyzed_at TIMESTAMPTZ DEFAULT NOW(),
    analysis_duration_ms INTEGER,

    -- Changed files summary
    files_changed INTEGER,
    lines_added INTEGER,
    lines_deleted INTEGER,
    affected_components TEXT[],

    -- TEST IMPACT PREDICTIONS
    predicted_test_failures JSONB DEFAULT '[]',
    -- Format: [{"test_id": "...", "test_name": "...", "failure_probability": 0.85, "reason": "..."}]

    tests_to_run_suggested UUID[],  -- Subset of tests to run
    estimated_test_duration_ms INTEGER,

    -- SECURITY ANALYSIS
    security_vulnerabilities JSONB DEFAULT '[]',
    -- Format: [{"severity": "high", "type": "sql_injection", "file": "...", "line": 123}]

    security_risk_score DECIMAL(3,2),  -- 0.00-1.00

    -- AIOPS PREDICTIONS
    performance_impact_prediction JSONB DEFAULT '{}',
    -- Format: {"risk": "medium", "affected_endpoints": [...], "estimated_latency_increase_ms": 50}

    reliability_risk_score DECIMAL(3,2),  -- 0.00-1.00
    incident_probability DECIMAL(3,2),    -- P(incident in 24h)

    -- RECOMMENDATIONS
    recommendations JSONB DEFAULT '[]',
    -- Format: [{"priority": "high", "action": "...", "reason": "..."}]

    deployment_strategy TEXT,  -- 'safe', 'canary_10', 'canary_50', 'full', 'manual_review'

    -- Outcome tracking (filled after deployment)
    actual_test_failures UUID[],
    actual_incidents UUID[],
    prediction_accuracy_score DECIMAL(3,2),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, commit_sha)
);
```

---

## Prediction Algorithms

### Test Failure Prediction

```python
def predict_test_failures(commit: CommitAnalysis) -> list[TestPrediction]:
    """
    Predict which tests are likely to fail for a given commit.

    Features used:
    1. Direct file→test mapping (test_impact_graph)
    2. Historical co-failure patterns
    3. Flaky test history
    4. Code change complexity
    5. Time-based patterns
    """

    predictions = []

    # 1. Get directly affected tests from impact graph
    affected_tests = get_tests_for_files(commit.changed_files)

    for test in affected_tests:
        # Base probability from impact graph
        base_prob = test.impact_score

        # 2. Adjust for flaky test history
        flaky_adjustment = get_flaky_score(test.id)

        # 3. Adjust for historical patterns
        pattern_adjustment = match_failure_patterns(commit, test)

        # 4. Adjust for code complexity
        complexity_adjustment = analyze_change_complexity(commit, test.file_path)

        # 5. Adjust for time-based patterns
        time_adjustment = get_time_pattern_adjustment(test.id)

        # Combine with learned weights
        final_prob = combine_probabilities(
            base_prob,
            flaky_adjustment,
            pattern_adjustment,
            complexity_adjustment,
            time_adjustment
        )

        if final_prob > PREDICTION_THRESHOLD:
            predictions.append(TestPrediction(
                test_id=test.id,
                test_name=test.name,
                probability=final_prob,
                reasons=explain_prediction(test, commit)
            ))

    return sorted(predictions, key=lambda p: p.probability, reverse=True)
```

### Production Risk Prediction

```python
def predict_production_risk(commit: CommitAnalysis) -> RiskAssessment:
    """
    Predict the probability of production incidents after deployment.

    Features used:
    1. Historical incidents for similar changes
    2. Component risk scores
    3. Dependency vulnerability data
    4. Deploy time patterns
    5. Change magnitude
    """

    # Query similar historical commits
    similar_commits = find_similar_commits(commit)
    incident_rate = calculate_incident_rate(similar_commits)

    # Component risk assessment
    component_risks = []
    for component in commit.affected_components:
        risk = get_component_risk_score(component)
        component_risks.append(risk)

    # Dependency analysis
    new_deps = extract_dependency_changes(commit)
    dep_risk = assess_dependency_risk(new_deps)

    # Time-based risk
    time_risk = get_deploy_time_risk(datetime.now())

    # Change magnitude risk
    magnitude_risk = assess_change_magnitude(
        commit.lines_added,
        commit.lines_deleted,
        commit.files_changed
    )

    # Combine risks
    overall_risk = weighted_average([
        (incident_rate, 0.3),
        (max(component_risks), 0.25),
        (dep_risk, 0.2),
        (time_risk, 0.1),
        (magnitude_risk, 0.15)
    ])

    # Determine deployment strategy
    if overall_risk > 0.7:
        strategy = 'manual_review'
    elif overall_risk > 0.5:
        strategy = 'canary_10'
    elif overall_risk > 0.3:
        strategy = 'canary_50'
    else:
        strategy = 'full'

    return RiskAssessment(
        risk_score=overall_risk,
        deployment_strategy=strategy,
        risk_factors=explain_risk_factors(commit)
    )
```

---

## Integration Points

### 1. GitHub/GitLab Webhook

```python
@router.post("/webhooks/github/push")
async def handle_github_push(payload: GitHubPushPayload):
    """
    Handle push events from GitHub.
    Triggers commit impact analysis.
    """
    # Extract commit info
    commit = extract_commit_info(payload)

    # Store in sdlc_events
    event_id = await store_sdlc_event(
        event_type='commit',
        source_platform='github',
        external_id=commit.sha,
        commit_sha=commit.sha,
        data=payload.dict()
    )

    # Trigger async analysis
    asyncio.create_task(analyze_commit_impact(commit))

    return {"status": "analysis_started", "event_id": event_id}
```

### 2. PR Comment Integration

```python
async def post_analysis_to_pr(analysis: CommitAnalysis):
    """
    Post analysis results as a PR comment on GitHub.
    """
    comment = format_pr_comment(analysis)

    # Example comment format:
    """
    ## 🔮 Skopaq Commit Impact Analysis

    ### 🧪 Test Predictions
    | Test | Failure Risk | Reason |
    |------|-------------|--------|
    | `test_auth_login` | 🔴 85% | File `auth.py` changed, 3 past failures |
    | `test_api_users` | 🟡 45% | Transitive dependency changed |

    ### 🔒 Security
    - ⚠️ **Medium Risk**: Potential SQL injection in `users.py:123`
    - ✅ No new dependency vulnerabilities

    ### 📈 Production Risk
    - **Overall Risk**: 🟡 Medium (0.42)
    - **Recommended Strategy**: Canary 10%
    - **Incident Probability**: 12% in next 24h

    ### 💡 Recommendations
    1. Run `test_auth_*` tests before merge
    2. Review SQL query in `users.py:123`
    3. Consider deploying during low-traffic hours

    ---
    *Powered by Skopaq AI Quality Intelligence*
    """

    await github.post_pr_comment(
        repo=analysis.repo,
        pr_number=analysis.pr_number,
        body=comment
    )
```

### 3. Slack Notification

```python
async def notify_high_risk_commit(analysis: CommitAnalysis):
    """
    Send Slack notification for high-risk commits.
    """
    if analysis.reliability_risk_score > 0.7:
        await slack.send_message(
            channel="#engineering-alerts",
            blocks=[
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "⚠️ High-Risk Commit Detected"}
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Commit:* `{analysis.commit_sha[:8]}`\n"
                               f"*Author:* {analysis.author}\n"
                               f"*Risk Score:* {analysis.reliability_risk_score:.0%}\n"
                               f"*Predicted Failures:* {len(analysis.predicted_test_failures)}"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View Analysis"},
                            "url": f"https://app.skopaq.ai/commits/{analysis.commit_sha}"
                        }
                    ]
                }
            ]
        )
```

---

## Learning & Feedback Loop

The system improves over time by tracking prediction accuracy:

```python
async def record_actual_outcome(commit_sha: str, outcome: CommitOutcome):
    """
    Record the actual outcome after deployment to improve predictions.
    """
    analysis = await get_commit_analysis(commit_sha)

    # Compare predictions to reality
    predicted_failures = set(p['test_id'] for p in analysis.predicted_test_failures)
    actual_failures = set(outcome.failed_test_ids)

    # Calculate accuracy
    true_positives = predicted_failures & actual_failures
    false_positives = predicted_failures - actual_failures
    false_negatives = actual_failures - predicted_failures

    accuracy = len(true_positives) / (len(true_positives) + len(false_positives) + len(false_negatives))

    # Update analysis record
    await update_commit_analysis(commit_sha, {
        'actual_test_failures': list(actual_failures),
        'actual_incidents': outcome.incident_ids,
        'prediction_accuracy_score': accuracy
    })

    # Update failure patterns based on new data
    await update_failure_patterns(analysis, outcome)

    # Update test impact graph if new relationships discovered
    await update_test_impact_graph(analysis, outcome)
```

---

## DevSecOps Integration

### Security Scanners

1. **SAST (Static Analysis)**
   - Semgrep rules for common vulnerabilities
   - Custom rules for project-specific patterns
   - SQL injection, XSS, CSRF detection

2. **Dependency Scanning**
   - Snyk/Dependabot integration
   - CVE database lookup
   - License compliance checking

3. **Secrets Detection**
   - Gitleaks/TruffleHog integration
   - Custom regex patterns
   - Entropy-based detection

### AIOps Integration

1. **Performance Prediction**
   - APM data correlation (Datadog, New Relic)
   - Load test history analysis
   - Endpoint latency prediction

2. **Reliability Prediction**
   - Error rate prediction (Sentry integration)
   - Incident probability (PagerDuty history)
   - Service dependency mapping

3. **Capacity Planning**
   - Resource usage prediction
   - Auto-scaling recommendations
   - Cost impact estimation

---

## API Endpoints

```python
# Commit Analysis
POST   /api/v1/commits/{sha}/analyze          # Trigger analysis
GET    /api/v1/commits/{sha}/analysis         # Get analysis results
GET    /api/v1/commits/{sha}/predictions      # Get test predictions
POST   /api/v1/commits/{sha}/outcomes         # Record actual outcomes

# Impact Graph
GET    /api/v1/projects/{id}/impact-graph     # Get full impact graph
GET    /api/v1/files/{path}/tests             # Get tests for file
PUT    /api/v1/impact-graph/refresh           # Rebuild impact graph

# Patterns
GET    /api/v1/projects/{id}/failure-patterns # List learned patterns
GET    /api/v1/patterns/{id}                  # Get pattern details
DELETE /api/v1/patterns/{id}                  # Disable pattern

# Metrics
GET    /api/v1/projects/{id}/prediction-accuracy  # Overall accuracy
GET    /api/v1/projects/{id}/risk-trends          # Risk over time
```

---

## Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Test Prediction Accuracy | >80% | Predicted failures match actual |
| False Positive Rate | <15% | Tests predicted to fail that passed |
| Mean Time to Predict | <30s | Time from push to analysis complete |
| Incident Prevention Rate | >50% | High-risk deployments caught early |
| Developer Trust Score | >4.0/5 | Survey: "Predictions are useful" |

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Database schema for impact graph and patterns
- [ ] Webhook handlers for GitHub/GitLab
- [ ] Basic file→test mapping from coverage data
- [ ] Simple prediction based on direct file changes

### Phase 2: Intelligence (Week 2-3)
- [ ] Historical pattern learning
- [ ] Flaky test integration
- [ ] PR comment integration
- [ ] Slack notifications

### Phase 3: Security (Week 3-4)
- [ ] SAST integration (Semgrep)
- [ ] Dependency vulnerability scanning
- [ ] Secrets detection
- [ ] Security risk scoring

### Phase 4: AIOps (Week 4-5)
- [ ] APM data integration
- [ ] Incident correlation
- [ ] Performance prediction
- [ ] Deployment strategy recommendations

### Phase 5: Learning (Week 5-6)
- [ ] Outcome tracking
- [ ] Pattern accuracy improvement
- [ ] Impact graph refinement
- [ ] A/B testing for prediction algorithms

---

## Competitive Advantage

This feature positions Skopaq uniquely because:

1. **Shift-Left Intelligence**: Predictions at commit time, not after deployment
2. **Cross-SDLC Correlation**: Uses data from entire pipeline (Jira→Code→Test→Deploy→Monitor)
3. **Continuous Learning**: Gets smarter with every deployment
4. **Unified View**: Single platform for testing, security, and reliability
5. **AI-Native**: Uses LLMs for semantic understanding, not just pattern matching

No competitor currently offers this level of integration across the SDLC with predictive capabilities.

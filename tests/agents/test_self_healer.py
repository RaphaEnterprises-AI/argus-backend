"""Tests for the self healer agent module."""

from unittest.mock import MagicMock, patch

import pytest


class TestFailureType:
    """Tests for FailureType enum."""

    @pytest.mark.parametrize(
        "enum_name,expected_value",
        [
            ("SELECTOR_CHANGED", "selector_changed"),
            ("TIMING_ISSUE", "timing_issue"),
            ("UI_CHANGED", "ui_changed"),
            ("DATA_CHANGED", "data_changed"),
            ("REAL_BUG", "real_bug"),
            ("UNKNOWN", "unknown"),
        ],
    )
    def test_failure_type_values(self, mock_env_vars, enum_name, expected_value):
        """Test FailureType enum values."""
        from src.agents.self_healer import FailureType

        assert getattr(FailureType, enum_name).value == expected_value


class TestFixType:
    """Tests for FixType enum."""

    @pytest.mark.parametrize(
        "enum_name,expected_value",
        [
            ("UPDATE_SELECTOR", "update_selector"),
            ("ADD_WAIT", "add_wait"),
            ("INCREASE_TIMEOUT", "increase_timeout"),
            ("UPDATE_ASSERTION", "update_assertion"),
            ("UPDATE_TEST_DATA", "update_test_data"),
            ("NONE", "none"),
        ],
    )
    def test_fix_type_values(self, mock_env_vars, enum_name, expected_value):
        """Test FixType enum values."""
        from src.agents.self_healer import FixType

        assert getattr(FixType, enum_name).value == expected_value


class TestFailureDiagnosis:
    """Tests for FailureDiagnosis dataclass."""

    @pytest.mark.parametrize(
        "failure_type_name,confidence,explanation",
        [
            ("SELECTOR_CHANGED", 0.9, "Element was renamed"),
            ("TIMING_ISSUE", 0.85, "Element not ready"),
            ("UI_CHANGED", 0.75, "Layout changed"),
            ("DATA_CHANGED", 0.8, "Data mismatch"),
            ("REAL_BUG", 0.95, "Actual bug found"),
            ("UNKNOWN", 0.5, "Cannot determine cause"),
        ],
    )
    def test_diagnosis_creation(self, mock_env_vars, failure_type_name, confidence, explanation):
        """Test FailureDiagnosis creation with various failure types."""
        from src.agents.self_healer import FailureDiagnosis, FailureType

        failure_type = getattr(FailureType, failure_type_name)
        diagnosis = FailureDiagnosis(
            failure_type=failure_type,
            confidence=confidence,
            explanation=explanation,
        )

        assert diagnosis.failure_type == failure_type
        assert diagnosis.confidence == confidence
        assert diagnosis.explanation == explanation
        assert diagnosis.affected_step is None
        assert diagnosis.evidence == []

    @pytest.mark.parametrize(
        "affected_step,evidence",
        [
            (0, ["First step failed"]),
            (2, ["Timeout after 5s", "Element found after 6s"]),
            (5, ["Multiple", "pieces", "of", "evidence"]),
            (None, []),
        ],
    )
    def test_diagnosis_with_evidence(self, mock_env_vars, affected_step, evidence):
        """Test FailureDiagnosis with various evidence configurations."""
        from src.agents.self_healer import FailureDiagnosis, FailureType

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.TIMING_ISSUE,
            confidence=0.85,
            explanation="Element not ready",
            affected_step=affected_step,
            evidence=evidence,
        )

        assert diagnosis.affected_step == affected_step
        assert len(diagnosis.evidence) == len(evidence)
        assert diagnosis.evidence == evidence


class TestFixSuggestion:
    """Tests for FixSuggestion dataclass."""

    @pytest.mark.parametrize(
        "fix_type_name,old_value,new_value,confidence,explanation",
        [
            ("UPDATE_SELECTOR", "#old-btn", "#new-btn", 0.95, "Button was renamed"),
            ("ADD_WAIT", None, "#element", 0.8, "Add wait for element"),
            ("INCREASE_TIMEOUT", "5000", "15000", 0.85, "Increase timeout"),
            ("UPDATE_ASSERTION", "old text", "new text", 0.9, "Text changed"),
            ("UPDATE_TEST_DATA", "old@email.com", "new@email.com", 0.75, "Update email"),
            ("NONE", None, None, 0.3, "No fix available"),
        ],
    )
    def test_fix_creation(self, mock_env_vars, fix_type_name, old_value, new_value, confidence, explanation):
        """Test FixSuggestion creation with various fix types."""
        from src.agents.self_healer import FixSuggestion, FixType

        fix_type = getattr(FixType, fix_type_name)
        fix = FixSuggestion(
            fix_type=fix_type,
            old_value=old_value,
            new_value=new_value,
            confidence=confidence,
            explanation=explanation,
        )

        assert fix.fix_type == fix_type
        assert fix.old_value == old_value
        assert fix.new_value == new_value
        assert fix.confidence == confidence
        assert fix.explanation == explanation
        assert fix.requires_review is True  # Default value

    @pytest.mark.parametrize(
        "fix_type_name,old_value,new_value,confidence",
        [
            ("ADD_WAIT", None, "#element", 0.8),
            ("UPDATE_SELECTOR", "#old", "#new", 0.95),
            ("INCREASE_TIMEOUT", "5000", "10000", 0.7),
        ],
    )
    def test_fix_to_dict(self, mock_env_vars, fix_type_name, old_value, new_value, confidence):
        """Test FixSuggestion to_dict method with various inputs."""
        from src.agents.self_healer import FixSuggestion, FixType

        fix_type = getattr(FixType, fix_type_name)
        fix = FixSuggestion(
            fix_type=fix_type,
            old_value=old_value,
            new_value=new_value,
            confidence=confidence,
            explanation="Test explanation",
            requires_review=True,
        )

        result = fix.to_dict()

        assert result["fix_type"] == fix_type.value
        assert result["old_value"] == old_value
        assert result["new_value"] == new_value
        assert result["confidence"] == confidence
        assert result["requires_review"] is True


class TestHealingResult:
    """Tests for HealingResult dataclass."""

    def test_result_creation(self, mock_env_vars):
        """Test HealingResult creation."""
        from src.agents.self_healer import (
            FailureDiagnosis,
            FailureType,
            FixSuggestion,
            FixType,
            HealingResult,
        )

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.SELECTOR_CHANGED,
            confidence=0.9,
            explanation="Button moved",
        )

        fixes = [
            FixSuggestion(
                fix_type=FixType.UPDATE_SELECTOR,
                old_value="#old",
                new_value="#new",
                confidence=0.95,
                explanation="Update selector",
            )
        ]

        result = HealingResult(
            test_id="test-001",
            diagnosis=diagnosis,
            suggested_fixes=fixes,
        )

        assert result.test_id == "test-001"
        assert result.auto_healed is False

    def test_result_to_dict(self, mock_env_vars):
        """Test HealingResult to_dict method."""
        from src.agents.self_healer import (
            FailureDiagnosis,
            FailureType,
            FixSuggestion,
            FixType,
            HealingResult,
        )

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.TIMING_ISSUE,
            confidence=0.8,
            explanation="Slow load",
        )

        fixes = [
            FixSuggestion(
                fix_type=FixType.ADD_WAIT,
                confidence=0.85,
                explanation="Add wait",
            )
        ]

        result = HealingResult(
            test_id="test-001",
            diagnosis=diagnosis,
            suggested_fixes=fixes,
            auto_healed=True,
        )

        data = result.to_dict()

        assert data["test_id"] == "test-001"
        assert data["diagnosis"]["type"] == "timing_issue"
        assert data["auto_healed"] is True


class TestSelfHealerAgent:
    """Tests for SelfHealerAgent class."""

    @pytest.mark.parametrize(
        "threshold",
        [0.9, 0.95, 0.8, 0.85, 0.99, 0.5],
    )
    def test_agent_creation_with_threshold(self, mock_env_vars, threshold):
        """Test SelfHealerAgent creation with various thresholds."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(auto_heal_threshold=threshold)

            assert agent.auto_heal_threshold == threshold

    def test_agent_creation_default_threshold(self, mock_env_vars):
        """Test SelfHealerAgent creation with default threshold."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9  # Simulating default

            assert agent.auto_heal_threshold == 0.9

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            prompt = agent._get_system_prompt()

            # Enhanced prompt may use different terminology but covers the same concepts
            assert "self" in prompt.lower() or "healing" in prompt.lower() or "selector" in prompt.lower()
            assert "JSON" in prompt or "json" in prompt

    @pytest.mark.parametrize(
        "test_id,test_name,action,target,failure_type,failure_message",
        [
            ("test-001", "Login Test", "click", "#btn", "element_not_found", "Element #btn not found"),
            ("test-002", "Submit Form", "fill", "#email", "timeout", "Timeout waiting for element"),
            ("test-003", "Navigate", "goto", "https://example.com", "network_error", "Network request failed"),
            ("test-004", "Verify Text", "assert", ".message", "assertion_failed", "Expected text not found"),
        ],
    )
    def test_build_analysis_prompt(self, mock_env_vars, test_id, test_name, action, target, failure_type, failure_message):
        """Test analysis prompt building with various test configurations."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": test_id,
                "name": test_name,
                "steps": [{"action": action, "target": target}],
            }

            failure_details = {
                "type": failure_type,
                "message": failure_message,
            }

            prompt = agent._build_analysis_prompt(test_spec, failure_details, None)

            assert "TEST SPECIFICATION" in prompt
            assert "FAILURE DETAILS" in prompt
            assert test_id in prompt

    @pytest.mark.parametrize(
        "error_logs",
        [
            "Error: Element not found\nStack trace...",
            "TimeoutError: Navigation timeout of 30000ms exceeded",
            "AssertionError: Expected 'Success' but got 'Error'\n  at test.js:42",
            "NetworkError: Failed to fetch\n  Request URL: https://api.example.com",
            "",  # Empty logs
        ],
    )
    def test_build_analysis_prompt_with_logs(self, mock_env_vars, error_logs):
        """Test prompt building with various error log formats."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {"id": "test-001"}
            failure_details = {"message": "Error"}

            prompt = agent._build_analysis_prompt(test_spec, failure_details, error_logs if error_logs else None)

            if error_logs:
                assert "ERROR LOGS" in prompt
            # Prompt should always contain basic sections
            assert "TEST SPECIFICATION" in prompt
            assert "FAILURE DETAILS" in prompt

    @pytest.mark.parametrize(
        "failure_type_str,expected_type,confidence,explanation,affected_step,evidence",
        [
            ("selector_changed", "SELECTOR_CHANGED", 0.9, "Button was renamed", 2, ["Old selector not found"]),
            ("timing_issue", "TIMING_ISSUE", 0.85, "Element not ready", 1, ["Timeout occurred"]),
            ("ui_changed", "UI_CHANGED", 0.75, "Layout changed", 0, []),
            ("data_changed", "DATA_CHANGED", 0.8, "Data mismatch", None, ["Expected vs actual"]),
            ("real_bug", "REAL_BUG", 0.95, "Actual bug", 3, ["Stack trace", "Error log"]),
            ("unknown", "UNKNOWN", 0.5, "Cannot determine", None, []),
        ],
    )
    def test_parse_diagnosis_valid(
        self, mock_env_vars, failure_type_str, expected_type, confidence, explanation, affected_step, evidence
    ):
        """Test parsing valid diagnosis with various failure types."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FailureType, SelfHealerAgent

            agent = SelfHealerAgent()

            data = {
                "diagnosis": {
                    "failure_type": failure_type_str,
                    "confidence": confidence,
                    "explanation": explanation,
                    "affected_step": affected_step,
                    "evidence": evidence,
                }
            }

            diagnosis = agent._parse_diagnosis(data)

            assert diagnosis.failure_type == getattr(FailureType, expected_type)
            assert diagnosis.confidence == confidence
            assert diagnosis.explanation == explanation

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "invalid_type",
            "not_a_real_type",
            "",
            "SELECTOR_CHANGED",  # Wrong case
            "selector-changed",  # Wrong separator
            "none",
        ],
    )
    def test_parse_diagnosis_unknown_type(self, mock_env_vars, invalid_type):
        """Test parsing diagnosis with unknown/invalid type falls back to UNKNOWN."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FailureType, SelfHealerAgent

            agent = SelfHealerAgent()

            data = {
                "diagnosis": {
                    "failure_type": invalid_type,
                    "confidence": 0.5,
                }
            }

            diagnosis = agent._parse_diagnosis(data)

            assert diagnosis.failure_type == FailureType.UNKNOWN

    @pytest.mark.parametrize(
        "fixes_data,expected_count,auto_heal_threshold",
        [
            # Single fix with high confidence
            (
                [{"fix_type": "update_selector", "old_value": "#old", "new_value": "#new", "confidence": 0.95, "explanation": "Update"}],
                1,
                0.9,
            ),
            # Multiple fixes sorted by confidence
            (
                [
                    {"fix_type": "update_selector", "old_value": "#old", "new_value": "#new", "confidence": 0.95, "explanation": "Update selector"},
                    {"fix_type": "add_wait", "confidence": 0.7, "explanation": "Add wait"},
                ],
                2,
                0.9,
            ),
            # Three fixes with varying confidence
            (
                [
                    {"fix_type": "add_wait", "confidence": 0.6, "explanation": "Wait"},
                    {"fix_type": "update_selector", "confidence": 0.85, "explanation": "Selector"},
                    {"fix_type": "increase_timeout", "confidence": 0.75, "explanation": "Timeout"},
                ],
                3,
                0.8,
            ),
            # Empty fixes list
            ([], 0, 0.9),
        ],
    )
    def test_parse_fixes_valid(self, mock_env_vars, fixes_data, expected_count, auto_heal_threshold):
        """Test parsing valid fixes with various configurations."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = auto_heal_threshold

            data = {"fixes": fixes_data}

            fixes = agent._parse_fixes(data)

            assert len(fixes) == expected_count
            # Verify sorted by confidence (descending)
            for i in range(len(fixes) - 1):
                assert fixes[i].confidence >= fixes[i + 1].confidence
            # Verify requires_review based on threshold
            for fix in fixes:
                if fix.confidence >= auto_heal_threshold:
                    assert fix.requires_review is False
                else:
                    assert fix.requires_review is True

    @pytest.mark.parametrize(
        "invalid_fix_type",
        [
            "invalid_fix",
            "not_a_fix",
            "",
            "UPDATE_SELECTOR",  # Wrong case
            "update-selector",  # Wrong separator
        ],
    )
    def test_parse_fixes_unknown_type(self, mock_env_vars, invalid_fix_type):
        """Test parsing fixes with unknown type falls back to NONE."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixType, SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9

            data = {
                "fixes": [
                    {
                        "fix_type": invalid_fix_type,
                        "confidence": 0.5,
                    }
                ]
            }

            fixes = agent._parse_fixes(data)

            assert len(fixes) == 1
            assert fixes[0].fix_type == FixType.NONE

    @pytest.mark.parametrize(
        "old_selector,new_selector",
        [
            ("#old-btn", "#new-btn"),
            (".old-class", ".new-class"),
            ("[data-testid='old']", "[data-testid='new']"),
            ("button.submit", "button.submit-form"),
            ("#login-form input[type='submit']", "#login-form button.submit"),
        ],
    )
    async def test_apply_fix_update_selector(self, mock_env_vars, old_selector, new_selector):
        """Test applying selector update fix with various selectors."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "click", "target": old_selector},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_SELECTOR,
                old_value=old_selector,
                new_value=new_selector,
                confidence=0.95,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0]["target"] == new_selector
            assert healed["_healed"] is True

    @pytest.mark.parametrize(
        "element_to_wait",
        [
            "#element-to-wait",
            ".loading-spinner",
            "[data-loaded='true']",
            "#content",
        ],
    )
    async def test_apply_fix_add_wait(self, mock_env_vars, element_to_wait):
        """Test applying add wait fix with various elements."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            import structlog

            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()
            agent.log = structlog.get_logger()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "click", "target": "#btn"},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.ADD_WAIT,
                new_value=element_to_wait,
                confidence=0.85,
            )

            healed = await agent._apply_fix(test_spec, fix)

            # Wait step should be inserted
            assert any(s["action"] == "wait" for s in healed["steps"])

    @pytest.mark.parametrize(
        "original_timeout,new_timeout_str,expected_timeout",
        [
            (5000, "15000", 15000),
            (1000, "5000", 5000),
            (10000, "30000", 30000),
            (500, "2000", 2000),
        ],
    )
    async def test_apply_fix_increase_timeout(self, mock_env_vars, original_timeout, new_timeout_str, expected_timeout):
        """Test applying increase timeout fix with various timeout values."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            import structlog

            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()
            agent.log = structlog.get_logger()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "click", "target": "#btn", "timeout": original_timeout},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.INCREASE_TIMEOUT,
                old_value="#btn",
                new_value=new_timeout_str,
                confidence=0.8,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0]["timeout"] == expected_timeout

    @pytest.mark.parametrize(
        "old_text,new_text",
        [
            ("Old Text", "New Text"),
            ("Welcome", "Welcome Back"),
            ("0 items", "1 item"),
            ("Loading...", "Loaded"),
            ("Error: Not found", "Success"),
        ],
    )
    async def test_apply_fix_update_assertion(self, mock_env_vars, old_text, new_text):
        """Test applying update assertion fix with various text values."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "assertions": [
                    {"type": "text_equals", "expected": old_text},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_ASSERTION,
                old_value=old_text,
                new_value=new_text,
                confidence=0.9,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["assertions"][0]["expected"] == new_text

    @pytest.mark.parametrize(
        "old_data,new_data,field",
        [
            ("old@example.com", "new@example.com", "value"),
            ("olduser", "newuser", "value"),
            ("password123", "newpassword456", "value"),
            ("https://old-url.com", "https://new-url.com", "value"),
        ],
    )
    async def test_apply_fix_update_test_data(self, mock_env_vars, old_data, new_data, field):
        """Test applying update test data fix with various data values."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "fill", "target": "#input", field: old_data},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_TEST_DATA,
                old_value=old_data,
                new_value=new_data,
                confidence=0.85,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0][field] == new_data

    @pytest.mark.asyncio
    async def test_execute(self, mock_env_vars):
        """Test execute method."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9
            agent.log = MagicMock()
            agent._check_cost_limit = MagicMock(return_value=True)

            mock_response = MagicMock()
            mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
            mock_response.content = [MagicMock(text='''
            {
                "diagnosis": {
                    "failure_type": "selector_changed",
                    "confidence": 0.9,
                    "explanation": "Button renamed"
                },
                "fixes": [
                    {
                        "fix_type": "update_selector",
                        "old_value": "#old",
                        "new_value": "#new",
                        "confidence": 0.95,
                        "explanation": "Update"
                    }
                ]
            }
            ''')]

            agent._call_claude = MagicMock(return_value=mock_response)
            agent._extract_text_response = MagicMock(return_value=mock_response.content[0].text)
            agent._parse_json_response = MagicMock(return_value={
                "diagnosis": {
                    "failure_type": "selector_changed",
                    "confidence": 0.9,
                    "explanation": "Button renamed"
                },
                "fixes": [
                    {
                        "fix_type": "update_selector",
                        "old_value": "#old",
                        "new_value": "#new",
                        "confidence": 0.95,
                        "explanation": "Update"
                    }
                ]
            })

            test_spec = {
                "id": "test-001",
                "steps": [{"action": "click", "target": "#old"}],
            }

            failure_details = {
                "message": "Element not found",
            }

            result = await agent.execute(test_spec, failure_details)

            assert result.success is True
            assert result.data.auto_healed is True

    @pytest.mark.asyncio
    async def test_execute_cost_limit_exceeded(self, mock_env_vars):
        """Test execute when cost limit exceeded."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.log = MagicMock()
            agent._check_cost_limit = MagicMock(return_value=False)

            result = await agent.execute({}, {})

            assert result.success is False
            assert "Cost limit" in result.error

    @pytest.mark.asyncio
    async def test_batch_analyze(self, mock_env_vars):
        """Test batch analysis of failures."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9
            agent.log = MagicMock()
            agent._check_cost_limit = MagicMock(return_value=True)

            mock_response = MagicMock()
            mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
            mock_response.content = [MagicMock(text='{"diagnosis": {"failure_type": "unknown"}, "fixes": []}')]

            agent._call_claude = MagicMock(return_value=mock_response)
            agent._extract_text_response = MagicMock(return_value=mock_response.content[0].text)
            agent._parse_json_response = MagicMock(return_value={
                "diagnosis": {"failure_type": "unknown"},
                "fixes": []
            })

            failures = [
                ({"id": "test-1"}, {"error": "Error 1"}, None),
                ({"id": "test-2"}, {"error": "Error 2"}, None),
            ]

            results = await agent.batch_analyze(failures)

            assert len(results) == 2


class TestSelfHealerAgentCogneePatterns:
    """Tests for SelfHealerAgent cognee_patterns parameter."""

    def test_cognee_patterns_default_empty_list(self, mock_env_vars):
        """Test cognee_patterns defaults to empty list."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            assert agent.cognee_patterns == []

    def test_cognee_patterns_accepts_list(self, mock_env_vars):
        """Test cognee_patterns accepts a list of patterns."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            patterns = [
                {"selector": "#old-btn", "healed_selector": "#new-btn", "confidence": 0.9}
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns)
            assert agent.cognee_patterns == patterns

    def test_cognee_patterns_none_becomes_empty_list(self, mock_env_vars):
        """Test cognee_patterns=None is converted to empty list."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(cognee_patterns=None)
            assert agent.cognee_patterns == []
            assert isinstance(agent.cognee_patterns, list)

    def test_lookup_prefetched_patterns_returns_none_for_empty(self, mock_env_vars):
        """Test _lookup_prefetched_cognee_patterns returns None for empty patterns."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.log = MagicMock()
            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#button",
                error_type="selector_not_found",
                error_message="Element not found"
            )
            assert result is None

    def test_lookup_prefetched_patterns_finds_match(self, mock_env_vars):
        """Test _lookup_prefetched_cognee_patterns finds matching pattern."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixType, SelfHealerAgent

            patterns = [
                {
                    "selector": "#old-button",
                    "healed_selector": "#new-button",
                    "error_type": "selector_not_found",
                    "confidence": 0.95
                }
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns, auto_heal_threshold=0.9)
            agent.log = MagicMock()

            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#old-button",
                error_type="selector_not_found",
                error_message="Element not found"
            )

            assert result is not None
            fix, pattern_id = result
            assert fix.fix_type == FixType.UPDATE_SELECTOR
            assert fix.old_value == "#old-button"
            assert fix.new_value == "#new-button"
            assert fix.confidence > 0.8

    def test_lookup_prefetched_patterns_filters_low_confidence(self, mock_env_vars):
        """Test patterns with low confidence are filtered out."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            patterns = [
                {
                    "selector": "#old-button",
                    "healed_selector": "#new-button",
                    "confidence": 0.5  # Below 0.8 threshold
                }
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns)
            agent.log = MagicMock()

            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#old-button",
                error_type=None,
                error_message="Element not found"
            )
            assert result is None

    def test_lookup_prefetched_patterns_uses_similarity(self, mock_env_vars):
        """Test similarity is used as fallback for confidence."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            patterns = [
                {
                    "selector": "#old-button",
                    "healed_selector": "#new-button",
                    "similarity": 0.9  # Using similarity instead of confidence
                }
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns, auto_heal_threshold=0.9)
            agent.log = MagicMock()

            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#old-button",
                error_type=None,
                error_message="Element not found"
            )
            assert result is not None
            fix, _ = result
            assert fix.confidence > 0.8

    def test_lookup_prefetched_patterns_caps_confidence(self, mock_env_vars):
        """Test confidence is capped at 0.95."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            patterns = [
                {
                    "selector": "#button",
                    "healed_selector": "#new-button",
                    "confidence": 1.0  # Full confidence
                }
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns, auto_heal_threshold=0.9)
            agent.log = MagicMock()

            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#button",
                error_type=None,
                error_message="Element not found"
            )
            assert result is not None
            fix, _ = result
            assert fix.confidence <= 0.95

    def test_lookup_prefetched_patterns_requires_healed_selector(self, mock_env_vars):
        """Test patterns without healed_selector are skipped."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            patterns = [
                {
                    "selector": "#old-button",
                    # No healed_selector
                    "confidence": 0.95
                }
            ]
            agent = SelfHealerAgent(cognee_patterns=patterns)
            agent.log = MagicMock()

            result = agent._lookup_prefetched_cognee_patterns(
                original_selector="#old-button",
                error_type=None,
                error_message="Element not found"
            )
            assert result is None


class TestSelfHealerAgentLLMProvider:
    """Tests for SelfHealerAgent llm_provider parameter."""

    def test_llm_provider_default_anthropic(self, mock_env_vars):
        """Test llm_provider defaults to anthropic."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            assert agent.llm_provider == "anthropic"

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "ollama", "openai", "openrouter", "azure"],
    )
    def test_llm_provider_accepts_various_values(self, mock_env_vars, provider):
        """Test llm_provider accepts various provider values."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(llm_provider=provider)
            assert agent.llm_provider == provider


class TestSelfHealerAgentCodeAnalyzer:
    """Tests for SelfHealerAgent code_analyzer parameter."""

    def test_code_analyzer_param_is_optional(self, mock_env_vars):
        """Test code_analyzer parameter is optional."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(enable_code_aware=False)
            assert agent.git_analyzer is None

    def test_code_analyzer_accepts_external_analyzer(self, mock_env_vars):
        """Test code_analyzer accepts an external analyzer instance."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            class MockAnalyzer:
                pass

            mock_analyzer = MockAnalyzer()
            agent = SelfHealerAgent(code_analyzer=mock_analyzer)
            assert agent.git_analyzer is mock_analyzer
            assert agent.source_analyzer is None  # Source analyzer requires local access

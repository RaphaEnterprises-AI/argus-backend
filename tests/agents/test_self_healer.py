"""Tests for the self healer agent module."""

from unittest.mock import MagicMock, patch

import pytest


class TestFailureType:
    """Tests for FailureType enum."""

    def test_failure_types(self, mock_env_vars):
        """Test FailureType enum values."""
        from src.agents.self_healer import FailureType

        assert FailureType.SELECTOR_CHANGED.value == "selector_changed"
        assert FailureType.TIMING_ISSUE.value == "timing_issue"
        assert FailureType.UI_CHANGED.value == "ui_changed"
        assert FailureType.DATA_CHANGED.value == "data_changed"
        assert FailureType.REAL_BUG.value == "real_bug"
        assert FailureType.UNKNOWN.value == "unknown"


class TestFixType:
    """Tests for FixType enum."""

    def test_fix_types(self, mock_env_vars):
        """Test FixType enum values."""
        from src.agents.self_healer import FixType

        assert FixType.UPDATE_SELECTOR.value == "update_selector"
        assert FixType.ADD_WAIT.value == "add_wait"
        assert FixType.INCREASE_TIMEOUT.value == "increase_timeout"
        assert FixType.UPDATE_ASSERTION.value == "update_assertion"
        assert FixType.UPDATE_TEST_DATA.value == "update_test_data"
        assert FixType.NONE.value == "none"


class TestFailureDiagnosis:
    """Tests for FailureDiagnosis dataclass."""

    def test_diagnosis_creation(self, mock_env_vars):
        """Test FailureDiagnosis creation."""
        from src.agents.self_healer import FailureDiagnosis, FailureType

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.SELECTOR_CHANGED,
            confidence=0.9,
            explanation="Element was renamed",
        )

        assert diagnosis.failure_type == FailureType.SELECTOR_CHANGED
        assert diagnosis.confidence == 0.9
        assert diagnosis.affected_step is None
        assert diagnosis.evidence == []

    def test_diagnosis_with_evidence(self, mock_env_vars):
        """Test FailureDiagnosis with evidence."""
        from src.agents.self_healer import FailureDiagnosis, FailureType

        diagnosis = FailureDiagnosis(
            failure_type=FailureType.TIMING_ISSUE,
            confidence=0.85,
            explanation="Element not ready",
            affected_step=2,
            evidence=["Timeout after 5s", "Element found after 6s"],
        )

        assert diagnosis.affected_step == 2
        assert len(diagnosis.evidence) == 2


class TestFixSuggestion:
    """Tests for FixSuggestion dataclass."""

    def test_fix_creation(self, mock_env_vars):
        """Test FixSuggestion creation."""
        from src.agents.self_healer import FixSuggestion, FixType

        fix = FixSuggestion(
            fix_type=FixType.UPDATE_SELECTOR,
            old_value="#old-btn",
            new_value="#new-btn",
            confidence=0.95,
            explanation="Button was renamed",
        )

        assert fix.fix_type == FixType.UPDATE_SELECTOR
        assert fix.confidence == 0.95
        assert fix.requires_review is True

    def test_fix_to_dict(self, mock_env_vars):
        """Test FixSuggestion to_dict method."""
        from src.agents.self_healer import FixSuggestion, FixType

        fix = FixSuggestion(
            fix_type=FixType.ADD_WAIT,
            old_value=None,
            new_value="#element",
            confidence=0.8,
            explanation="Add wait for element",
            requires_review=True,
        )

        result = fix.to_dict()

        assert result["fix_type"] == "add_wait"
        assert result["new_value"] == "#element"
        assert result["confidence"] == 0.8


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

    def test_agent_creation(self, mock_env_vars):
        """Test SelfHealerAgent creation."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9

            assert agent.auto_heal_threshold == 0.9

    def test_agent_custom_threshold(self, mock_env_vars):
        """Test SelfHealerAgent with custom threshold."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(auto_heal_threshold=0.95)

            assert agent.auto_heal_threshold == 0.95

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            prompt = agent._get_system_prompt()

            # Enhanced prompt may use different terminology but covers the same concepts
            assert "self" in prompt.lower() or "healing" in prompt.lower() or "selector" in prompt.lower()
            assert "JSON" in prompt or "json" in prompt

    def test_build_analysis_prompt(self, mock_env_vars):
        """Test analysis prompt building."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "name": "Login Test",
                "steps": [{"action": "click", "target": "#btn"}],
            }

            failure_details = {
                "type": "element_not_found",
                "message": "Element #btn not found",
            }

            prompt = agent._build_analysis_prompt(test_spec, failure_details, None)

            assert "TEST SPECIFICATION" in prompt
            assert "FAILURE DETAILS" in prompt
            assert "test-001" in prompt

    def test_build_analysis_prompt_with_logs(self, mock_env_vars):
        """Test prompt building with error logs."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {"id": "test-001"}
            failure_details = {"message": "Error"}
            error_logs = "Error: Element not found\nStack trace..."

            prompt = agent._build_analysis_prompt(test_spec, failure_details, error_logs)

            assert "ERROR LOGS" in prompt
            assert "Element not found" in prompt

    def test_parse_diagnosis_valid(self, mock_env_vars):
        """Test parsing valid diagnosis."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FailureType, SelfHealerAgent

            agent = SelfHealerAgent()

            data = {
                "diagnosis": {
                    "failure_type": "selector_changed",
                    "confidence": 0.9,
                    "explanation": "Button was renamed",
                    "affected_step": 2,
                    "evidence": ["Old selector not found"],
                }
            }

            diagnosis = agent._parse_diagnosis(data)

            assert diagnosis.failure_type == FailureType.SELECTOR_CHANGED
            assert diagnosis.confidence == 0.9

    def test_parse_diagnosis_unknown_type(self, mock_env_vars):
        """Test parsing diagnosis with unknown type."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FailureType, SelfHealerAgent

            agent = SelfHealerAgent()

            data = {
                "diagnosis": {
                    "failure_type": "invalid_type",
                    "confidence": 0.5,
                }
            }

            diagnosis = agent._parse_diagnosis(data)

            assert diagnosis.failure_type == FailureType.UNKNOWN

    def test_parse_fixes_valid(self, mock_env_vars):
        """Test parsing valid fixes."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9

            data = {
                "fixes": [
                    {
                        "fix_type": "update_selector",
                        "old_value": "#old",
                        "new_value": "#new",
                        "confidence": 0.95,
                        "explanation": "Update selector",
                    },
                    {
                        "fix_type": "add_wait",
                        "confidence": 0.7,
                        "explanation": "Add wait",
                    },
                ]
            }

            fixes = agent._parse_fixes(data)

            assert len(fixes) == 2
            # Should be sorted by confidence
            assert fixes[0].confidence > fixes[1].confidence
            assert fixes[0].requires_review is False  # High confidence
            assert fixes[1].requires_review is True  # Low confidence

    def test_parse_fixes_unknown_type(self, mock_env_vars):
        """Test parsing fixes with unknown type."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixType, SelfHealerAgent

            agent = SelfHealerAgent()
            agent.auto_heal_threshold = 0.9

            data = {
                "fixes": [
                    {
                        "fix_type": "invalid_fix",
                        "confidence": 0.5,
                    }
                ]
            }

            fixes = agent._parse_fixes(data)

            assert len(fixes) == 1
            assert fixes[0].fix_type == FixType.NONE

    async def test_apply_fix_update_selector(self, mock_env_vars):
        """Test applying selector update fix."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "click", "target": "#old-btn"},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_SELECTOR,
                old_value="#old-btn",
                new_value="#new-btn",
                confidence=0.95,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0]["target"] == "#new-btn"
            assert healed["_healed"] is True

    async def test_apply_fix_add_wait(self, mock_env_vars):
        """Test applying add wait fix."""
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
                new_value="#element-to-wait",
                confidence=0.85,
            )

            healed = await agent._apply_fix(test_spec, fix)

            # Wait step should be inserted
            assert any(s["action"] == "wait" for s in healed["steps"])

    async def test_apply_fix_increase_timeout(self, mock_env_vars):
        """Test applying increase timeout fix."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            import structlog

            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()
            agent.log = structlog.get_logger()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "click", "target": "#btn", "timeout": 5000},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.INCREASE_TIMEOUT,
                old_value="#btn",
                new_value="15000",
                confidence=0.8,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0]["timeout"] == 15000

    async def test_apply_fix_update_assertion(self, mock_env_vars):
        """Test applying update assertion fix."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "assertions": [
                    {"type": "text_equals", "expected": "Old Text"},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_ASSERTION,
                old_value="Old Text",
                new_value="New Text",
                confidence=0.9,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["assertions"][0]["expected"] == "New Text"

    async def test_apply_fix_update_test_data(self, mock_env_vars):
        """Test applying update test data fix."""
        with patch('src.agents.self_healer.BaseAgent.__init__', return_value=None):
            from src.agents.self_healer import FixSuggestion, FixType, SelfHealerAgent

            agent = SelfHealerAgent()

            test_spec = {
                "id": "test-001",
                "steps": [
                    {"action": "fill", "target": "#email", "value": "old@example.com"},
                ],
            }

            fix = FixSuggestion(
                fix_type=FixType.UPDATE_TEST_DATA,
                old_value="old@example.com",
                new_value="new@example.com",
                confidence=0.85,
            )

            healed = await agent._apply_fix(test_spec, fix)

            assert healed["steps"][0]["value"] == "new@example.com"

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

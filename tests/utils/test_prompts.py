"""Tests for the prompt templates module."""


import pytest


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_prompt_template_creation(self, mock_env_vars):
        """Test PromptTemplate creation."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}!",
            required_vars=["name"],
        )

        assert template.name == "test_prompt"
        assert template.required_vars == ["name"]
        assert template.optional_vars == []

    def test_prompt_template_with_optional_vars(self, mock_env_vars):
        """Test PromptTemplate with optional variables."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}! {greeting}",
            required_vars=["name"],
            optional_vars=["greeting"],
        )

        assert template.optional_vars == ["greeting"]

    def test_prompt_template_with_description(self, mock_env_vars):
        """Test PromptTemplate with description."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}!",
            required_vars=["name"],
            description="A greeting template",
        )

        assert template.description == "A greeting template"

    def test_render_success(self, mock_env_vars):
        """Test successful render."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}! Welcome to {place}.",
            required_vars=["name", "place"],
        )

        result = template.render(name="Alice", place="Wonderland")

        assert result == "Hello Alice! Welcome to Wonderland."

    def test_render_with_optional_vars(self, mock_env_vars):
        """Test render with optional variables."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}!{extra}",
            required_vars=["name"],
            optional_vars=["extra"],
        )

        # Without optional
        result1 = template.render(name="Alice")
        assert result1 == "Hello Alice!"

        # With optional
        result2 = template.render(name="Alice", extra=" Welcome!")
        assert result2 == "Hello Alice! Welcome!"

    def test_render_missing_required_vars(self, mock_env_vars):
        """Test render with missing required variables."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}! Welcome to {place}.",
            required_vars=["name", "place"],
        )

        with pytest.raises(ValueError) as exc_info:
            template.render(name="Alice")

        assert "place" in str(exc_info.value)

    def test_validate_all_present(self, mock_env_vars):
        """Test validate when all variables present."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}!",
            required_vars=["name"],
        )

        missing = template.validate(name="Alice")

        assert missing == []

    def test_validate_missing_vars(self, mock_env_vars):
        """Test validate with missing variables."""
        from src.utils.prompts import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="Hello {name}! {greeting}",
            required_vars=["name", "greeting"],
        )

        missing = template.validate(name="Alice")

        assert "greeting" in missing


class TestPredefinedPrompts:
    """Tests for predefined prompt templates."""

    def test_prompts_dictionary_exists(self, mock_env_vars):
        """Test that PROMPTS dictionary exists."""
        from src.utils.prompts import PROMPTS

        assert isinstance(PROMPTS, dict)
        assert len(PROMPTS) > 0

    def test_analyze_codebase_prompt(self, mock_env_vars):
        """Test analyze_codebase prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["analyze_codebase"]

        assert template.name == "analyze_codebase"
        assert "codebase_path" in template.required_vars
        assert "app_url" in template.required_vars

    def test_plan_tests_prompt(self, mock_env_vars):
        """Test plan_tests prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["plan_tests"]

        assert template.name == "plan_tests"
        assert "app_url" in template.required_vars
        assert "surfaces_json" in template.required_vars

    def test_verify_screenshot_prompt(self, mock_env_vars):
        """Test verify_screenshot prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["verify_screenshot"]

        assert template.name == "verify_screenshot"
        assert "verification_task" in template.required_vars

    def test_diagnose_failure_prompt(self, mock_env_vars):
        """Test diagnose_failure prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["diagnose_failure"]

        assert template.name == "diagnose_failure"
        assert "test_spec_json" in template.required_vars
        assert "failure_json" in template.required_vars

    def test_generate_report_prompt(self, mock_env_vars):
        """Test generate_report prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["generate_report"]

        assert template.name == "generate_report"
        assert "total_tests" in template.required_vars

    def test_execute_test_prompt(self, mock_env_vars):
        """Test execute_test prompt template."""
        from src.utils.prompts import PROMPTS

        template = PROMPTS["execute_test"]

        assert template.name == "execute_test"
        assert "test_name" in template.required_vars
        assert "app_url" in template.required_vars

    def test_all_prompts_renderable(self, mock_env_vars):
        """Test that all prompts can be rendered with sample data."""
        from src.utils.prompts import PROMPTS

        sample_values = {
            "codebase_path": "/path/to/code",
            "app_url": "http://localhost:3000",
            "changed_files_section": "",
            "file_contents_section": "",
            "codebase_summary_section": "",
            "surfaces_json": "[]",
            "max_tests": "10",
            "verification_task": "Check login page",
            "expected_elements": "Login form",
            "test_spec_json": "{}",
            "failure_json": "{}",
            "error_logs_section": "",
            "total_tests": "10",
            "passed": "8",
            "failed": "2",
            "pass_rate": "80",
            "failures_json": "[]",
            "test_name": "Login Test",
            "steps_text": "1. Go to login",
            "assertions_text": "1. Verify URL",
        }

        for name, template in PROMPTS.items():
            # Get required values for this template
            kwargs = {var: sample_values.get(var, f"value_{var}") for var in template.required_vars}
            for var in template.optional_vars:
                kwargs[var] = sample_values.get(var, "")

            result = template.render(**kwargs)
            assert len(result) > 0, f"Prompt {name} rendered empty"


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_get_prompt_success(self, mock_env_vars):
        """Test get_prompt with valid name."""
        from src.utils.prompts import get_prompt

        result = get_prompt(
            "verify_screenshot",
            verification_task="Check login",
        )

        assert "Check login" in result

    def test_get_prompt_unknown_name(self, mock_env_vars):
        """Test get_prompt with unknown name."""
        from src.utils.prompts import get_prompt

        with pytest.raises(KeyError) as exc_info:
            get_prompt("nonexistent_prompt")

        assert "nonexistent_prompt" in str(exc_info.value)

    def test_get_prompt_missing_vars(self, mock_env_vars):
        """Test get_prompt with missing variables."""
        from src.utils.prompts import get_prompt

        with pytest.raises(ValueError):
            get_prompt("verify_screenshot")  # Missing verification_task


class TestListPrompts:
    """Tests for list_prompts function."""

    def test_list_prompts(self, mock_env_vars):
        """Test list_prompts returns all prompts."""
        from src.utils.prompts import PROMPTS, list_prompts

        prompts = list_prompts()

        assert len(prompts) == len(PROMPTS)

        for prompt_info in prompts:
            assert "name" in prompt_info
            assert "description" in prompt_info
            assert "required_vars" in prompt_info
            assert "optional_vars" in prompt_info


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_prompt_builder_creation(self, mock_env_vars):
        """Test PromptBuilder creation."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()

        assert builder._parts == []

    def test_add_context(self, mock_env_vars):
        """Test add_context method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        result = builder.add_context("You are a testing agent")

        assert result is builder  # Returns self for chaining
        assert "You are a testing agent" in builder._parts

    def test_add_section(self, mock_env_vars):
        """Test add_section method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_section("TASK", "Execute this test")

        prompt = builder.build()
        assert "TASK:" in prompt
        assert "Execute this test" in prompt

    def test_add_json_block(self, mock_env_vars):
        """Test add_json_block method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        data = {"key": "value", "number": 42}
        builder.add_json_block(data)

        prompt = builder.build()
        assert "```json" in prompt
        assert '"key"' in prompt
        assert '"value"' in prompt

    def test_add_json_block_with_title(self, mock_env_vars):
        """Test add_json_block with title."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        data = {"test": "data"}
        builder.add_json_block(data, title="TEST DATA")

        prompt = builder.build()
        assert "TEST DATA:" in prompt
        assert "```json" in prompt

    def test_add_instructions(self, mock_env_vars):
        """Test add_instructions method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_instructions(["Step one", "Step two", "Step three"])

        prompt = builder.build()
        assert "INSTRUCTIONS:" in prompt
        assert "1. Step one" in prompt
        assert "2. Step two" in prompt
        assert "3. Step three" in prompt

    def test_add_list(self, mock_env_vars):
        """Test add_list method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_list("ITEMS", ["Item A", "Item B"])

        prompt = builder.build()
        assert "ITEMS:" in prompt
        assert "- Item A" in prompt
        assert "- Item B" in prompt

    def test_add_response_format(self, mock_env_vars):
        """Test add_response_format method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_response_format('{"status": "passed"}')

        prompt = builder.build()
        assert "Respond with JSON:" in prompt
        assert '{"status": "passed"}' in prompt

    def test_build(self, mock_env_vars):
        """Test build method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_context("Context")
        builder.add_section("TITLE", "Content")

        prompt = builder.build()

        assert isinstance(prompt, str)
        assert "Context" in prompt
        assert "TITLE:" in prompt

    def test_clear(self, mock_env_vars):
        """Test clear method."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        builder.add_context("Some context")
        builder.clear()

        assert builder._parts == []
        assert builder.build() == ""

    def test_chaining(self, mock_env_vars):
        """Test method chaining."""
        from src.utils.prompts import PromptBuilder

        builder = PromptBuilder()
        prompt = (
            builder
            .add_context("You are a testing agent")
            .add_section("TASK", "Run tests")
            .add_json_block({"test": "spec"}, title="SPEC")
            .add_instructions(["Do this", "Do that"])
            .add_list("TOOLS", ["playwright", "selenium"])
            .add_response_format('{"result": "value"}')
            .build()
        )

        assert "You are a testing agent" in prompt
        assert "TASK:" in prompt
        assert "SPEC:" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "TOOLS:" in prompt
        assert "Respond with JSON:" in prompt

    def test_complex_prompt_building(self, mock_env_vars):
        """Test building a complex prompt."""
        from src.utils.prompts import PromptBuilder

        test_spec = {
            "name": "Login Test",
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "fill", "target": "#email", "value": "test@example.com"},
            ],
        }

        builder = PromptBuilder()
        prompt = (
            builder
            .add_context("You are an autonomous E2E testing agent.")
            .add_section("TEST NAME", test_spec["name"])
            .add_json_block(test_spec["steps"], title="STEPS")
            .add_instructions([
                "Navigate to the page",
                "Fill in the form",
                "Verify the result",
            ])
            .add_response_format('{"status": "passed", "observations": []}')
            .build()
        )

        # Verify all parts are present
        assert "autonomous E2E testing agent" in prompt
        assert "Login Test" in prompt
        assert "/login" in prompt
        assert "#email" in prompt
        assert "Navigate to the page" in prompt
        assert '{"status": "passed"' in prompt

"""Natural Language test creation - testRigor style.

Converts plain English descriptions into executable test specifications.
No coding required - just describe what you want to test.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

from ..config import get_settings

logger = structlog.get_logger()


@dataclass
class TestStep:
    """A single step in a test."""
    action: str  # goto, click, fill, type, wait, assert, etc.
    target: Optional[str] = None  # Selector, URL, or element description
    value: Optional[str] = None  # Input value
    description: str = ""  # Human-readable description


@dataclass
class TestAssertion:
    """A test assertion."""
    type: str  # element_visible, text_contains, url_matches, etc.
    target: str
    expected: str
    description: str = ""


@dataclass
class GeneratedTest:
    """A test generated from natural language."""
    id: str
    name: str
    description: str
    original_prompt: str
    steps: list[TestStep] = field(default_factory=list)
    assertions: list[TestAssertion] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    priority: str = "medium"
    estimated_duration_seconds: int = 30

    def to_spec(self) -> dict:
        """Convert to test specification format."""
        return {
            "id": self.id,
            "name": self.name,
            "type": "ui",
            "priority": self.priority,
            "preconditions": self.preconditions,
            "steps": [
                {
                    "action": s.action,
                    "target": s.target,
                    "value": s.value,
                }
                for s in self.steps
            ],
            "assertions": [
                {
                    "type": a.type,
                    "target": a.target,
                    "expected": a.expected,
                }
                for a in self.assertions
            ],
            "tags": self.tags,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "original_prompt": self.original_prompt,
            "steps": [
                {
                    "action": s.action,
                    "target": s.target,
                    "value": s.value,
                    "description": s.description,
                }
                for s in self.steps
            ],
            "assertions": [
                {
                    "type": a.type,
                    "target": a.target,
                    "expected": a.expected,
                    "description": a.description,
                }
                for a in self.assertions
            ],
            "preconditions": self.preconditions,
            "tags": self.tags,
            "priority": self.priority,
            "estimated_duration_seconds": self.estimated_duration_seconds,
        }


class NLPTestCreator:
    """
    Creates tests from plain English descriptions.

    Like testRigor - no code needed, just describe what to test.

    Usage:
        creator = NLPTestCreator(app_url="http://localhost:3000")

        # Simple test
        test = await creator.create(
            "Login with email test@example.com and password secret123, "
            "then verify the dashboard shows a welcome message"
        )

        # Multiple tests from user story
        tests = await creator.create_from_story('''
            As a user, I want to:
            1. Register a new account
            2. Login with my credentials
            3. Update my profile
            4. Logout successfully
        ''')

        # Execute the generated test
        result = await executor.run(test.to_spec())
    """

    def __init__(
        self,
        app_url: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self.app_url = app_url
        self.model = model
        self.log = logger.bind(component="nlp_test_creator")

    async def create(
        self,
        description: str,
        context: Optional[str] = None,
        test_data: Optional[dict] = None,
    ) -> GeneratedTest:
        """
        Create a test from a plain English description.

        Args:
            description: Plain English description of what to test
            context: Optional context about the app (routes, components)
            test_data: Optional test data to use (credentials, etc.)

        Returns:
            GeneratedTest ready to execute
        """
        self.log.info("Creating test from description", description=description[:100])

        context_text = ""
        if context:
            context_text = f"\n\nAPP CONTEXT:\n{context}"

        data_text = ""
        if test_data:
            data_text = f"\n\nTEST DATA AVAILABLE:\n{json.dumps(test_data, indent=2)}"

        prompt = f"""You are a QA automation expert. Convert this plain English test description into an executable test specification.

TEST DESCRIPTION:
{description}

APP URL: {self.app_url}
{context_text}
{data_text}

Generate a complete test with specific, executable steps. Be precise with selectors and actions.

IMPORTANT GUIDELINES:
1. Use semantic selectors when possible (button text, labels, placeholders)
2. Include waits after navigation and form submissions
3. Add assertions to verify expected outcomes
4. Use realistic timing (wait for page loads)

Respond with JSON:
{{
    "id": "unique-test-id",
    "name": "Short descriptive name",
    "description": "What this test verifies",
    "priority": "critical|high|medium|low",
    "preconditions": ["any setup needed"],
    "steps": [
        {{
            "action": "goto|click|fill|type|wait|press|hover|select|screenshot",
            "target": "URL, selector, or element description",
            "value": "input value if needed",
            "description": "what this step does"
        }}
    ],
    "assertions": [
        {{
            "type": "element_visible|text_contains|url_matches|value_equals|element_count",
            "target": "what to check",
            "expected": "expected value",
            "description": "what we're verifying"
        }}
    ],
    "tags": ["relevant", "tags"],
    "estimated_duration_seconds": 30
}}

SELECTOR EXAMPLES:
- "text=Login" - button with text "Login"
- "placeholder=Email" - input with placeholder "Email"
- "[data-testid=submit]" - element with test ID
- "h1:has-text('Welcome')" - h1 containing "Welcome"
- "form >> input[type=email]" - email input inside form

ACTION EXAMPLES:
- {{"action": "goto", "target": "/login"}}
- {{"action": "fill", "target": "placeholder=Email", "value": "test@example.com"}}
- {{"action": "click", "target": "text=Submit"}}
- {{"action": "wait", "value": "2000"}}
- {{"action": "press", "target": "placeholder=Search", "value": "Enter"}}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Build GeneratedTest
            steps = [
                TestStep(
                    action=s.get("action", ""),
                    target=s.get("target"),
                    value=s.get("value"),
                    description=s.get("description", ""),
                )
                for s in data.get("steps", [])
            ]

            assertions = [
                TestAssertion(
                    type=a.get("type", ""),
                    target=a.get("target", ""),
                    expected=a.get("expected", ""),
                    description=a.get("description", ""),
                )
                for a in data.get("assertions", [])
            ]

            test = GeneratedTest(
                id=data.get("id", f"test-{hash(description) % 10000}"),
                name=data.get("name", "Generated Test"),
                description=data.get("description", description),
                original_prompt=description,
                steps=steps,
                assertions=assertions,
                preconditions=data.get("preconditions", []),
                tags=data.get("tags", []),
                priority=data.get("priority", "medium"),
                estimated_duration_seconds=data.get("estimated_duration_seconds", 30),
            )

            self.log.info(
                "Test created",
                test_id=test.id,
                steps=len(steps),
                assertions=len(assertions),
            )

            return test

        except Exception as e:
            self.log.error("Test creation failed", error=str(e))
            raise

    async def create_multiple(
        self,
        descriptions: list[str],
        context: Optional[str] = None,
    ) -> list[GeneratedTest]:
        """Create multiple tests from a list of descriptions."""
        tests = []
        for desc in descriptions:
            test = await self.create(desc, context)
            tests.append(test)
        return tests

    async def create_from_story(
        self,
        user_story: str,
        context: Optional[str] = None,
    ) -> list[GeneratedTest]:
        """
        Create multiple tests from a user story.

        Args:
            user_story: A user story or feature description
            context: Optional app context

        Returns:
            List of generated tests covering the story
        """
        self.log.info("Creating tests from user story")

        prompt = f"""You are a QA automation expert. Analyze this user story and identify all test scenarios.

USER STORY:
{user_story}

APP URL: {self.app_url}

Break this down into individual test scenarios. For each scenario, provide:
1. A clear test description
2. What user action it tests
3. What outcome it verifies

Respond with JSON:
{{
    "scenarios": [
        {{
            "name": "Test name",
            "description": "Plain English test description that can be converted to steps",
            "priority": "critical|high|medium|low",
            "tags": ["relevant", "tags"]
        }}
    ]
}}

Make sure each scenario is:
- Specific and actionable
- Independently executable
- Focused on one user flow
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())
            scenarios = data.get("scenarios", [])

            self.log.info("Identified scenarios", count=len(scenarios))

            # Generate detailed tests for each scenario
            tests = []
            for scenario in scenarios:
                test = await self.create(
                    scenario["description"],
                    context,
                )
                test.name = scenario.get("name", test.name)
                test.priority = scenario.get("priority", test.priority)
                test.tags = scenario.get("tags", test.tags)
                tests.append(test)

            return tests

        except Exception as e:
            self.log.error("Story parsing failed", error=str(e))
            raise

    async def enhance_test(
        self,
        test: GeneratedTest,
        feedback: str,
    ) -> GeneratedTest:
        """
        Enhance a test based on feedback.

        Args:
            test: Existing test to enhance
            feedback: What to improve or add

        Returns:
            Enhanced test
        """
        current_spec = json.dumps(test.to_spec(), indent=2)

        prompt = f"""You are a QA automation expert. Enhance this test based on the feedback.

CURRENT TEST:
{current_spec}

FEEDBACK:
{feedback}

Update the test to address the feedback. Keep existing steps that are still valid.

Respond with the complete updated test in JSON format (same structure as input).
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content.strip())

            # Rebuild test
            steps = [
                TestStep(
                    action=s.get("action", ""),
                    target=s.get("target"),
                    value=s.get("value"),
                    description=s.get("description", ""),
                )
                for s in data.get("steps", [])
            ]

            assertions = [
                TestAssertion(
                    type=a.get("type", ""),
                    target=a.get("target", ""),
                    expected=a.get("expected", ""),
                    description=a.get("description", ""),
                )
                for a in data.get("assertions", [])
            ]

            enhanced = GeneratedTest(
                id=test.id,
                name=data.get("name", test.name),
                description=data.get("description", test.description),
                original_prompt=test.original_prompt,
                steps=steps,
                assertions=assertions,
                preconditions=data.get("preconditions", test.preconditions),
                tags=data.get("tags", test.tags),
                priority=data.get("priority", test.priority),
            )

            self.log.info("Test enhanced", test_id=enhanced.id)
            return enhanced

        except Exception as e:
            self.log.error("Test enhancement failed", error=str(e))
            raise


class ConversationalTestBuilder:
    """
    Interactive test builder that works conversationally.

    Like having a conversation with a QA engineer.
    """

    def __init__(self, app_url: str):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self.app_url = app_url
        self.creator = NLPTestCreator(app_url)
        self.conversation_history = []
        self.current_tests = []
        self.log = logger.bind(component="conversational_test_builder")

    async def chat(self, message: str) -> str:
        """
        Have a conversation about testing.

        Can handle:
        - "Test that login works with valid credentials"
        - "Add a check for error message on invalid password"
        - "What tests do I have so far?"
        - "Make the login test more thorough"
        """
        self.conversation_history.append({"role": "user", "content": message})

        system = f"""You are a helpful QA assistant helping build tests for {self.app_url}.

You can:
1. Create new tests from descriptions
2. Modify existing tests
3. Explain what tests exist
4. Suggest additional test coverage

Current tests: {len(self.current_tests)}

When asked to create a test, respond with:
ACTION: CREATE_TEST
DESCRIPTION: <the test description>

When asked to modify a test, respond with:
ACTION: MODIFY_TEST
TEST_ID: <id>
CHANGES: <what to change>

For other questions, just respond conversationally.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1000,
            system=system,
            messages=self.conversation_history,
        )

        reply = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": reply})

        # Parse actions
        if "ACTION: CREATE_TEST" in reply:
            # Extract description and create test
            lines = reply.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("DESCRIPTION:"):
                    desc = line.replace("DESCRIPTION:", "").strip()
                    # Get any continuation lines
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("ACTION:") or not lines[j].strip():
                            break
                        desc += " " + lines[j].strip()

                    test = await self.creator.create(desc)
                    self.current_tests.append(test)
                    reply += f"\n\n✅ Created test: {test.name} ({len(test.steps)} steps)"
                    break

        return reply

    def get_all_tests(self) -> list[GeneratedTest]:
        """Get all tests created in this session."""
        return self.current_tests

    def export_tests(self) -> list[dict]:
        """Export all tests as specifications."""
        return [t.to_spec() for t in self.current_tests]


# Convenience function
def create_nlp_test_creator(app_url: str) -> NLPTestCreator:
    """Factory function for NLPTestCreator."""
    return NLPTestCreator(app_url=app_url)

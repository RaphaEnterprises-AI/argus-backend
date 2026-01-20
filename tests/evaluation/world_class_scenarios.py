"""
World-Class Evaluation Scenarios.

Test scenarios aligned with industry benchmarks:
- SWE-bench: Real code understanding and modification tasks
- WebArena: Multi-step web navigation with state
- BFCL: Function calling with complex schemas
- TAU-bench: Long-context multi-turn conversations
- Bloom: Behavioral elicitation scenarios

Each scenario includes:
- Difficulty stratification (easy/medium/hard/expert)
- Multiple evaluation attempts for pass@k
- State management for multi-turn
- Cost tracking for efficiency metrics
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from .world_class_metrics import EvalDomain, TaskDifficulty


@dataclass
class EvalScenario:
    """A single evaluation scenario with full context."""
    id: str
    name: str
    description: str
    domain: EvalDomain
    difficulty: TaskDifficulty

    # Input data
    initial_state: dict[str, Any]
    input_prompt: str

    # Expected outcomes
    success_criteria: list[dict[str, Any]]  # List of criteria to check
    expected_steps: int  # Minimum steps for completion
    timeout_seconds: int = 300

    # For multi-turn scenarios
    conversation_turns: list[dict[str, str]] = field(default_factory=list)
    requires_state_management: bool = False

    # For function calling scenarios
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)

    # Grading
    partial_credit_allowed: bool = True
    human_baseline_success_rate: float = 0.95  # Default human success rate


# =============================================================================
# CODE UNDERSTANDING SCENARIOS (SWE-bench style)
# =============================================================================

CODE_SCENARIOS = [
    EvalScenario(
        id="code_001",
        name="Identify Authentication Testable Surfaces",
        description="Analyze authentication code to find testable surfaces and functions",
        domain=EvalDomain.CODE_UNDERSTANDING,
        difficulty=TaskDifficulty.MEDIUM,
        initial_state={
            "codebase": {
                "auth.py": '''
def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "expired"}
    except:
        return {}  # BUG: Silent failure, should raise or return error

def login(username: str, password: str) -> str:
    user = db.get_user(username)
    if user and check_password(password, user.password_hash):
        return create_token(user.id)
    return None  # BUG: No rate limiting, timing attack possible
''',
            },
        },
        input_prompt="Analyze this authentication code and identify all testable surfaces including functions, error paths, and test scenarios.",
        success_criteria=[
            {"type": "identifies", "element": "verify_token"},
            {"type": "identifies", "element": "login"},
            {"type": "provides", "element": "test_scenarios"},
        ],
        expected_steps=3,
        human_baseline_success_rate=0.92,
    ),

    EvalScenario(
        id="code_002",
        name="Extract Testable Surfaces",
        description="Identify all testable components from a React application",
        domain=EvalDomain.CODE_UNDERSTANDING,
        difficulty=TaskDifficulty.EASY,
        initial_state={
            "codebase": {
                "ProductList.tsx": '''
export function ProductList({ category, onSelect }) {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    fetchProducts(category).then(setProducts).finally(() => setLoading(false));
  }, [category]);

  const filteredProducts = products.filter(p =>
    p.name.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div data-testid="product-list">
      <input
        data-testid="filter-input"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter products"
      />
      {loading && <Spinner data-testid="loading-spinner" />}
      {filteredProducts.map(product => (
        <ProductCard
          key={product.id}
          product={product}
          onClick={() => onSelect(product)}
          data-testid={`product-${product.id}`}
        />
      ))}
    </div>
  );
}
''',
            },
        },
        input_prompt="Identify all testable surfaces in this React component. Include UI elements, state changes, and async operations.",
        success_criteria=[
            {"type": "identifies", "element": "ProductList"},  # Component name
            {"type": "identifies", "element": "filter"},  # Filter functionality
            {"type": "identifies", "element": "product"},  # Product list
            {"type": "documents", "element": "surfaces", "count_min": 3},  # At least 3 surfaces found
        ],
        expected_steps=2,
        human_baseline_success_rate=0.98,
    ),

    EvalScenario(
        id="code_003",
        name="Complex State Machine Analysis",
        description="Understand and document a state machine implementation",
        domain=EvalDomain.CODE_UNDERSTANDING,
        difficulty=TaskDifficulty.HARD,
        initial_state={
            "codebase": {
                "order_state_machine.py": '''
class OrderStateMachine:
    STATES = ["pending", "confirmed", "processing", "shipped", "delivered", "cancelled", "refunded"]

    TRANSITIONS = {
        "pending": ["confirmed", "cancelled"],
        "confirmed": ["processing", "cancelled"],
        "processing": ["shipped", "cancelled"],
        "shipped": ["delivered", "returned"],
        "delivered": ["refunded"],
        "returned": ["refunded"],
        "cancelled": [],
        "refunded": [],
    }

    def __init__(self, order_id: str):
        self.order_id = order_id
        self.state = "pending"
        self.history = []
        self.metadata = {}

    def can_transition(self, target_state: str) -> bool:
        return target_state in self.TRANSITIONS.get(self.state, [])

    def transition(self, target_state: str, actor: str, reason: str = None) -> bool:
        if not self.can_transition(target_state):
            raise InvalidTransitionError(f"Cannot transition from {self.state} to {target_state}")

        self.history.append({
            "from": self.state,
            "to": target_state,
            "actor": actor,
            "reason": reason,
            "timestamp": datetime.utcnow(),
        })
        self.state = target_state
        self._trigger_side_effects(target_state)
        return True

    def _trigger_side_effects(self, state: str):
        handlers = {
            "confirmed": self._send_confirmation_email,
            "shipped": self._send_shipping_notification,
            "delivered": self._request_review,
            "cancelled": self._process_refund,
            "refunded": self._update_inventory,
        }
        if state in handlers:
            handlers[state]()
''',
            },
        },
        input_prompt="Analyze this state machine. Document all states, valid transitions, side effects, and identify potential edge cases for testing.",
        success_criteria=[
            {"type": "identifies", "element": "OrderStateMachine"},  # Main class
            {"type": "identifies", "element": "transition"},  # Transition method
            {"type": "identifies", "element": "state"},  # State management
            {"type": "documents", "element": "surfaces", "count_min": 4},  # At least 4 surfaces
        ],
        expected_steps=4,
        human_baseline_success_rate=0.85,
    ),

    EvalScenario(
        id="code_004",
        name="Security Vulnerability Analysis",
        description="Identify security vulnerabilities in authentication code",
        domain=EvalDomain.CODE_UNDERSTANDING,
        difficulty=TaskDifficulty.EXPERT,
        initial_state={
            "file": "auth_handler.py",
            "language": "python",
            "code": '''
import hashlib
import sqlite3
import os

class AuthHandler:
    def __init__(self, db_path):
        self.db = sqlite3.connect(db_path)
        self.secret = "hardcoded_secret_key_123"  # Used for token signing

    def authenticate(self, username, password):
        # Hash password and check against database
        hashed = hashlib.md5(password.encode()).hexdigest()
        query = f"SELECT * FROM users WHERE username = '{username}' AND password_hash = '{hashed}'"
        cursor = self.db.execute(query)
        user = cursor.fetchone()

        if user:
            return self._generate_token(user[0])
        return None

    def _generate_token(self, user_id):
        token = hashlib.sha256(f"{user_id}{self.secret}".encode()).hexdigest()
        return token

    def verify_token(self, token, user_id):
        expected = self._generate_token(user_id)
        return token == expected

    def reset_password(self, email):
        # Generate reset token and send email
        reset_token = os.urandom(8).hex()
        query = f"UPDATE users SET reset_token = '{reset_token}' WHERE email = '{email}'"
        self.db.execute(query)
        self.db.commit()
        return f"Password reset link: /reset?token={reset_token}&email={email}"
''',
        },
        input_prompt="Perform a security audit of this authentication handler. Identify all vulnerabilities, classify by severity (critical/high/medium/low), and suggest fixes.",
        success_criteria=[
            {"type": "identifies_vuln", "name": "SQL Injection", "severity": "critical", "location": "authenticate"},
            {"type": "identifies_vuln", "name": "SQL Injection", "severity": "critical", "location": "reset_password"},
            {"type": "identifies_vuln", "name": "Weak Hashing (MD5)", "severity": "high", "location": "authenticate"},
            {"type": "identifies_vuln", "name": "Hardcoded Secret", "severity": "high", "location": "__init__"},
            {"type": "identifies_vuln", "name": "Information Exposure", "severity": "medium", "location": "reset_password"},
            {"type": "suggests_fix", "for": "SQL Injection", "fix_type": "parameterized_queries"},
            {"type": "suggests_fix", "for": "Weak Hashing", "fix_type": "bcrypt_or_argon2"},
        ],
        expected_steps=7,
        human_baseline_success_rate=0.68,
    ),
]


# =============================================================================
# WEB NAVIGATION SCENARIOS (WebArena style)
# =============================================================================

WEB_SCENARIOS = [
    EvalScenario(
        id="web_001",
        name="E-commerce Search and Filter",
        description="Search for products, apply filters, and add to cart",
        domain=EvalDomain.WEB_NAVIGATION,
        difficulty=TaskDifficulty.MEDIUM,
        initial_state={
            "url": "https://shop.example.com",
            "logged_in": True,
            "cart_items": 0,
        },
        input_prompt="Search for 'wireless headphones', filter by price under $100 and rating 4+, then add the top result to cart",
        success_criteria=[
            {"type": "action", "action": "search", "value": "wireless headphones"},
            {"type": "action", "action": "filter", "field": "price", "value": "<100"},
            {"type": "action", "action": "filter", "field": "rating", "value": ">=4"},
            {"type": "action", "action": "add_to_cart"},
            {"type": "state", "cart_items": ">0"},
        ],
        expected_steps=5,
        requires_state_management=True,
        human_baseline_success_rate=0.95,
    ),

    EvalScenario(
        id="web_002",
        name="Multi-page Form Submission",
        description="Complete a multi-page checkout form with validation",
        domain=EvalDomain.WEB_NAVIGATION,
        difficulty=TaskDifficulty.HARD,
        initial_state={
            "url": "https://shop.example.com/checkout",
            "cart_total": 99.99,
            "user_data": {
                "name": "Test User",
                "email": "test@example.com",
                "address": "123 Test St",
                "card": "4111111111111111",
            },
        },
        input_prompt="Complete the checkout process: fill shipping info, select standard shipping, enter payment details, and confirm order",
        success_criteria=[
            {"type": "page", "url_contains": "/checkout/shipping"},
            {"type": "form_filled", "fields": ["name", "address", "city", "zip"]},
            {"type": "page", "url_contains": "/checkout/payment"},
            {"type": "form_filled", "fields": ["card_number", "expiry", "cvv"]},
            {"type": "page", "url_contains": "/checkout/confirmation"},
            {"type": "element_visible", "selector": "[data-testid='order-confirmation']"},
        ],
        expected_steps=8,
        requires_state_management=True,
        timeout_seconds=180,
        human_baseline_success_rate=0.88,
    ),

    EvalScenario(
        id="web_003",
        name="Error Recovery Flow",
        description="Handle validation errors and retry form submission",
        domain=EvalDomain.WEB_NAVIGATION,
        difficulty=TaskDifficulty.EXPERT,
        initial_state={
            "url": "https://app.example.com/settings",
            "errors_will_occur": True,
        },
        input_prompt="Update user profile with email 'new@example.com'. The first submission will fail with validation error - handle the error and retry with corrected data.",
        success_criteria=[
            {"type": "action", "action": "fill", "field": "email", "value": "new@example.com"},
            {"type": "action", "action": "submit"},
            {"type": "handles", "error_type": "validation_error"},
            {"type": "action", "action": "correct_and_retry"},
            {"type": "state", "profile_updated": True},
        ],
        expected_steps=6,
        requires_state_management=True,
        human_baseline_success_rate=0.78,
    ),
]


# =============================================================================
# FUNCTION CALLING SCENARIOS (BFCL style)
# =============================================================================

FUNCTION_CALLING_SCENARIOS = [
    EvalScenario(
        id="func_001",
        name="Single Tool Call",
        description="Make a simple API call with correct parameters",
        domain=EvalDomain.FUNCTION_CALLING,
        difficulty=TaskDifficulty.EASY,
        initial_state={},
        input_prompt="Get the weather forecast for San Francisco for the next 3 days",
        available_tools=[
            {
                "name": "get_weather",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "days": {"type": "integer", "description": "Number of forecast days"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "days"],
                },
            },
        ],
        expected_tool_calls=["get_weather"],
        success_criteria=[
            {"type": "tool_called", "name": "get_weather"},
            {"type": "param_correct", "tool": "get_weather", "param": "location", "contains": "San Francisco"},
            {"type": "param_correct", "tool": "get_weather", "param": "days", "value": 3},
        ],
        expected_steps=1,
        human_baseline_success_rate=0.99,
    ),

    EvalScenario(
        id="func_002",
        name="Parallel Tool Calls",
        description="Execute multiple independent tool calls in parallel",
        domain=EvalDomain.FUNCTION_CALLING,
        difficulty=TaskDifficulty.MEDIUM,
        initial_state={},
        input_prompt="Get the stock prices for AAPL, GOOGL, and MSFT, then calculate the total portfolio value assuming 10 shares of each",
        available_tools=[
            {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "calculate_portfolio",
                "description": "Calculate portfolio value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "holdings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "shares": {"type": "integer"},
                                    "price": {"type": "number"},
                                },
                            },
                        },
                    },
                    "required": ["holdings"],
                },
            },
        ],
        expected_tool_calls=["get_stock_price", "get_stock_price", "get_stock_price", "calculate_portfolio"],
        success_criteria=[
            {"type": "tool_called", "name": "get_stock_price", "count": 3},
            {"type": "parallel_calls", "tools": ["get_stock_price"], "min_parallel": 2},
            {"type": "tool_called", "name": "calculate_portfolio"},
            {"type": "correct_dependency", "depends_on": "get_stock_price", "tool": "calculate_portfolio"},
        ],
        expected_steps=2,  # Parallel fetch, then calculate
        human_baseline_success_rate=0.92,
    ),

    EvalScenario(
        id="func_003",
        name="Tool Selection with Ambiguity",
        description="Choose correct tool when multiple could apply",
        domain=EvalDomain.FUNCTION_CALLING,
        difficulty=TaskDifficulty.HARD,
        initial_state={},
        input_prompt="Find the user with email john.doe@company.com and update their role to 'admin'",
        available_tools=[
            {
                "name": "search_users",
                "description": "Search users by various criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "field": {"type": "string", "enum": ["name", "email", "department"]},
                    },
                },
            },
            {
                "name": "get_user_by_email",
                "description": "Get user by exact email match",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["email"],
                },
            },
            {
                "name": "update_user",
                "description": "Update user properties",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "updates": {"type": "object"},
                    },
                    "required": ["user_id", "updates"],
                },
            },
            {
                "name": "set_user_role",
                "description": "Set user role (requires user_id and role)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "role": {"type": "string", "enum": ["user", "admin", "moderator"]},
                    },
                    "required": ["user_id", "role"],
                },
            },
        ],
        expected_tool_calls=["get_user_by_email", "set_user_role"],
        success_criteria=[
            {"type": "tool_selected", "preferred": "get_user_by_email", "over": "search_users"},
            {"type": "tool_selected", "preferred": "set_user_role", "over": "update_user"},
            {"type": "correct_sequence", "sequence": ["get_user_by_email", "set_user_role"]},
        ],
        expected_steps=2,
        human_baseline_success_rate=0.85,
    ),

    EvalScenario(
        id="func_004",
        name="Conditional Tool Chain with Error Handling",
        description="Execute conditional tool calls with fallback logic",
        domain=EvalDomain.FUNCTION_CALLING,
        difficulty=TaskDifficulty.EXPERT,
        initial_state={
            "context": "Production environment - handle errors gracefully",
        },
        input_prompt="""Create a new user 'Alice Smith' with email 'alice@example.com'.
First check if the email is already registered. If the email exists, update the existing user's name instead of creating a duplicate.
If email doesn't exist, create the new user. Finally, send a welcome email regardless of whether it was a create or update.""",
        available_tools=[
            {
                "name": "check_email_exists",
                "description": "Check if an email is already registered",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                    },
                    "required": ["email"],
                },
            },
            {
                "name": "create_user",
                "description": "Create a new user account",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
            },
            {
                "name": "update_user",
                "description": "Update existing user by email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "updates": {"type": "object"},
                    },
                    "required": ["email", "updates"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email to a user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "template": {"type": "string", "enum": ["welcome", "password_reset", "notification"]},
                    },
                    "required": ["to", "template"],
                },
            },
        ],
        expected_tool_calls=["check_email_exists", "create_user", "send_email"],  # Primary path
        success_criteria=[
            {"type": "tool_called", "name": "check_email_exists"},
            {"type": "conditional_logic", "description": "Either create_user or update_user based on check"},
            {"type": "tool_called", "name": "send_email"},
        ],
        expected_steps=3,
        human_baseline_success_rate=0.72,
    ),
]


# =============================================================================
# MULTI-TURN SCENARIOS (TAU-bench style)
# =============================================================================

MULTI_TURN_SCENARIOS = [
    EvalScenario(
        id="turn_001",
        name="Context Retention Across Turns",
        description="Maintain context in a multi-turn conversation",
        domain=EvalDomain.MULTI_TURN_REASONING,
        difficulty=TaskDifficulty.MEDIUM,
        initial_state={"user_name": "Alice", "previous_orders": []},
        input_prompt="Start a product support conversation",
        conversation_turns=[
            {"role": "user", "content": "Hi, I bought a laptop last week and need help"},
            {"role": "assistant", "expected": "acknowledge_and_ask_order"},
            {"role": "user", "content": "Order #12345. The battery drains too fast."},
            {"role": "assistant", "expected": "lookup_order_and_troubleshoot"},
            {"role": "user", "content": "I already tried those steps. Can I get a replacement?"},
            {"role": "assistant", "expected": "initiate_replacement_with_context"},
        ],
        success_criteria=[
            {"type": "context_retained", "key": "order_number", "value": "12345"},
            {"type": "context_retained", "key": "issue", "value": "battery"},
            {"type": "no_repeated_questions", "about": ["order_number", "product"]},
            {"type": "escalation_appropriate", "after_troubleshooting": True},
        ],
        expected_steps=6,
        requires_state_management=True,
        human_baseline_success_rate=0.90,
    ),

    EvalScenario(
        id="turn_002",
        name="Goal Refinement Through Dialogue",
        description="Refine ambiguous goal through clarifying questions",
        domain=EvalDomain.MULTI_TURN_REASONING,
        difficulty=TaskDifficulty.HARD,
        initial_state={},
        input_prompt="Help user book a trip",
        conversation_turns=[
            {"role": "user", "content": "I want to book a trip"},
            {"role": "assistant", "expected": "ask_destination"},
            {"role": "user", "content": "Somewhere warm, maybe beach"},
            {"role": "assistant", "expected": "ask_dates_and_preferences"},
            {"role": "user", "content": "Next month, about a week, under $2000"},
            {"role": "assistant", "expected": "suggest_options_with_criteria"},
            {"role": "user", "content": "The second option looks good"},
            {"role": "assistant", "expected": "confirm_and_book_with_all_details"},
        ],
        success_criteria=[
            {"type": "clarification_asked", "for": ["destination", "dates", "budget"]},
            {"type": "constraints_tracked", "constraints": ["warm", "beach", "week", "under_2000"]},
            {"type": "options_matched_criteria", "all_constraints": True},
            {"type": "final_confirmation_complete", "includes": ["destination", "dates", "price"]},
        ],
        expected_steps=8,
        requires_state_management=True,
        human_baseline_success_rate=0.82,
    ),
]


# =============================================================================
# SELF-HEALING SCENARIOS (E2E Testing specific)
# =============================================================================

SELF_HEALING_SCENARIOS = [
    EvalScenario(
        id="heal_001",
        name="Selector Migration Detection",
        description="Detect and heal when selectors change due to refactoring",
        domain=EvalDomain.SELF_HEALING,
        difficulty=TaskDifficulty.MEDIUM,
        initial_state={
            "original_selector": "button#submit-btn",
            "current_html": '''
                <form data-testid="login-form">
                    <button data-testid="login-submit" type="submit">Sign In</button>
                </form>
            ''',
            "git_diff": '''
                - <button id="submit-btn">Sign In</button>
                + <button data-testid="login-submit">Sign In</button>
            ''',
        },
        input_prompt="The test step 'click button#submit-btn' failed. Analyze and provide a healed selector.",
        success_criteria=[
            {"type": "diagnosis", "failure_type": "selector_changed"},
            {"type": "new_selector", "uses": "data-testid"},
            {"type": "confidence", "min": 0.85},
            {"type": "reasoning", "mentions": "git_diff"},
        ],
        expected_steps=2,
        human_baseline_success_rate=0.95,
    ),

    EvalScenario(
        id="heal_002",
        name="Timing Issue Detection",
        description="Identify and fix timing-related test failures",
        domain=EvalDomain.SELF_HEALING,
        difficulty=TaskDifficulty.HARD,
        initial_state={
            "failed_step": "assert element #results visible",
            "timeout_ms": 5000,
            "network_logs": [
                {"url": "/api/search", "duration_ms": 4800, "status": 200},
            ],
            "historical_runs": [
                {"duration_ms": 1200, "passed": True},
                {"duration_ms": 1500, "passed": True},
                {"duration_ms": 4900, "passed": False},
            ],
        },
        input_prompt="Test 'search_results_displayed' failed with timeout. Analyze logs and history to diagnose.",
        success_criteria=[
            {"type": "diagnosis", "failure_type": "timing_issue"},
            {"type": "root_cause", "identified": "api_slowdown"},
            {"type": "recommendation", "fix_type": "increase_timeout"},
            {"type": "recommendation", "new_timeout_ms": {"min": 8000, "max": 15000}},
        ],
        expected_steps=3,
        human_baseline_success_rate=0.88,
    ),
]


# =============================================================================
# ALL SCENARIOS
# =============================================================================

ALL_SCENARIOS = (
    CODE_SCENARIOS +
    WEB_SCENARIOS +
    FUNCTION_CALLING_SCENARIOS +
    MULTI_TURN_SCENARIOS +
    SELF_HEALING_SCENARIOS
)


def get_scenarios_by_domain(domain: EvalDomain) -> list[EvalScenario]:
    """Get scenarios for a specific domain."""
    return [s for s in ALL_SCENARIOS if s.domain == domain]


def get_scenarios_by_difficulty(difficulty: TaskDifficulty) -> list[EvalScenario]:
    """Get scenarios for a specific difficulty."""
    return [s for s in ALL_SCENARIOS if s.difficulty == difficulty]


def get_scenario_distribution() -> dict[str, dict[str, int]]:
    """Get count of scenarios by domain and difficulty."""
    distribution = {}

    for domain in EvalDomain:
        distribution[domain.value] = {
            "total": 0,
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "expert": 0,
        }

    for scenario in ALL_SCENARIOS:
        distribution[scenario.domain.value]["total"] += 1
        distribution[scenario.domain.value][scenario.difficulty.value] += 1

    return distribution

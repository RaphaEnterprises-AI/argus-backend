"""
Real-World Test Cases for Agent Evaluation.

These test cases exercise the actual AI capabilities of each agent
using realistic scenarios that mirror production use.

Test Case Categories:
1. Code Analysis - Understanding codebases
2. Test Planning - Generating comprehensive test plans
3. UI Execution - Browser automation accuracy
4. Self-Healing - Fixing broken selectors/tests
5. NLP Understanding - Natural language to test conversion
6. Visual AI - Screenshot comparison and regression detection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    name: str
    category: str
    difficulty: Difficulty
    description: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    grading_rubric: dict[str, float]  # metric -> weight
    timeout_seconds: int = 120


# =============================================================================
# CODE ANALYSIS TEST CASES
# =============================================================================

CODE_ANALYSIS_CASES = [
    TestCase(
        id="ca_001",
        name="Simple React Login Component",
        category="code_analysis",
        difficulty=Difficulty.EASY,
        description="Analyze a simple React login form and identify testable surfaces",
        input_data={
            "code": '''
import React, { useState } from 'react';

export function LoginForm({ onSubmit }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields');
      return;
    }
    try {
      await onSubmit({ email, password });
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <form onSubmit={handleSubmit} data-testid="login-form">
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        data-testid="email-input"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        data-testid="password-input"
      />
      {error && <div data-testid="error-message">{error}</div>}
      <button type="submit" data-testid="submit-button">Login</button>
    </form>
  );
}
''',
            "file_path": "src/components/LoginForm.jsx",
            "framework": "react",
        },
        expected_output={
            "testable_surfaces": [
                {"type": "form", "name": "login-form"},
                {"type": "input", "name": "email-input"},
                {"type": "input", "name": "password-input"},
                {"type": "button", "name": "submit-button"},
                {"type": "validation", "name": "error-message"},
            ],
            "suggested_tests": [
                "empty_field_validation",
                "successful_login",
                "failed_login_error",
                "email_format_validation",
            ],
            "framework_detected": "react",
        },
        grading_rubric={
            "identified_all_elements": 0.4,
            "suggested_relevant_tests": 0.3,
            "detected_framework": 0.2,
            "no_false_positives": 0.1,
        },
    ),

    TestCase(
        id="ca_002",
        name="E-commerce Cart with State Management",
        category="code_analysis",
        difficulty=Difficulty.MEDIUM,
        description="Analyze complex shopping cart with Redux state",
        input_data={
            "code": '''
// cartSlice.js
import { createSlice } from '@reduxjs/toolkit';

const cartSlice = createSlice({
  name: 'cart',
  initialState: { items: [], total: 0, discount: null },
  reducers: {
    addItem: (state, action) => {
      const existing = state.items.find(i => i.id === action.payload.id);
      if (existing) {
        existing.quantity += 1;
      } else {
        state.items.push({ ...action.payload, quantity: 1 });
      }
      state.total = calculateTotal(state.items, state.discount);
    },
    removeItem: (state, action) => {
      state.items = state.items.filter(i => i.id !== action.payload);
      state.total = calculateTotal(state.items, state.discount);
    },
    applyDiscount: (state, action) => {
      state.discount = action.payload;
      state.total = calculateTotal(state.items, state.discount);
    },
    clearCart: (state) => {
      state.items = [];
      state.total = 0;
      state.discount = null;
    },
  },
});

// CartComponent.jsx
export function Cart() {
  const { items, total, discount } = useSelector(state => state.cart);
  const dispatch = useDispatch();

  return (
    <div data-testid="cart-container">
      {items.map(item => (
        <CartItem key={item.id} item={item} onRemove={() => dispatch(removeItem(item.id))} />
      ))}
      {discount && <div data-testid="discount-badge">{discount.code}: -{discount.percent}%</div>}
      <div data-testid="cart-total">${total.toFixed(2)}</div>
      <button data-testid="checkout-btn" disabled={items.length === 0}>Checkout</button>
    </div>
  );
}
''',
            "file_path": "src/features/cart/",
            "framework": "react",
        },
        expected_output={
            "testable_surfaces": [
                {"type": "state", "name": "cart_items"},
                {"type": "state", "name": "cart_total"},
                {"type": "action", "name": "addItem"},
                {"type": "action", "name": "removeItem"},
                {"type": "action", "name": "applyDiscount"},
                {"type": "action", "name": "clearCart"},
                {"type": "component", "name": "cart-container"},
                {"type": "button", "name": "checkout-btn"},
            ],
            "suggested_tests": [
                "add_item_to_empty_cart",
                "add_duplicate_item_increases_quantity",
                "remove_item_updates_total",
                "apply_discount_code",
                "checkout_disabled_when_empty",
                "clear_cart_resets_state",
            ],
            "complexity": "medium",
        },
        grading_rubric={
            "identified_state_management": 0.3,
            "identified_all_actions": 0.3,
            "suggested_edge_cases": 0.2,
            "understood_business_logic": 0.2,
        },
    ),

    TestCase(
        id="ca_003",
        name="Complex API with Authentication",
        category="code_analysis",
        difficulty=Difficulty.HARD,
        description="Analyze FastAPI backend with OAuth2 and rate limiting",
        input_data={
            "code": '''
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = await get_user(payload.get("sub"))
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# routes.py
@router.post("/orders")
@limiter.limit("10/minute")
async def create_order(
    order: OrderCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if user.balance < order.total:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    db_order = Order(**order.dict(), user_id=user.id)
    db.add(db_order)
    await db.commit()

    # Send notification
    await send_order_confirmation(user.email, db_order)

    return {"order_id": db_order.id, "status": "created"}

@router.get("/orders/{order_id}")
async def get_order(order_id: int, user: User = Depends(get_current_user)):
    order = await db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.user_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    return order
''',
            "file_path": "src/api/",
            "framework": "fastapi",
        },
        expected_output={
            "testable_surfaces": [
                {"type": "endpoint", "method": "POST", "path": "/orders"},
                {"type": "endpoint", "method": "GET", "path": "/orders/{order_id}"},
                {"type": "auth", "name": "oauth2_token_validation"},
                {"type": "auth", "name": "token_expiration"},
                {"type": "rate_limit", "name": "10_per_minute"},
                {"type": "authorization", "name": "order_access_control"},
            ],
            "suggested_tests": [
                "create_order_success",
                "create_order_insufficient_balance",
                "create_order_rate_limited",
                "get_order_owner_access",
                "get_order_admin_access",
                "get_order_unauthorized",
                "expired_token_rejected",
                "invalid_token_rejected",
            ],
            "security_considerations": [
                "token_validation",
                "authorization_checks",
                "rate_limiting",
            ],
        },
        grading_rubric={
            "identified_security_surfaces": 0.3,
            "identified_error_cases": 0.25,
            "understood_authorization": 0.25,
            "suggested_security_tests": 0.2,
        },
    ),
]

# =============================================================================
# TEST PLANNING TEST CASES
# =============================================================================

TEST_PLANNING_CASES = [
    TestCase(
        id="tp_001",
        name="Login Flow Test Plan",
        category="test_planning",
        difficulty=Difficulty.EASY,
        description="Generate test plan for a login flow",
        input_data={
            "app_url": "https://example.com",
            "testable_surfaces": [
                {"type": "page", "path": "/login"},
                {"type": "form", "id": "login-form"},
                {"type": "input", "name": "email"},
                {"type": "input", "name": "password"},
                {"type": "button", "text": "Sign In"},
            ],
            "requirements": "Test successful login, failed login, and validation",
        },
        expected_output={
            "tests": [
                {
                    "name": "successful_login",
                    "priority": "critical",
                    "steps": ["goto /login", "fill email", "fill password", "click Sign In"],
                    "assertions": ["url changed to /dashboard", "welcome message visible"],
                },
                {
                    "name": "invalid_credentials",
                    "priority": "high",
                    "steps": ["goto /login", "fill invalid email", "fill wrong password", "click Sign In"],
                    "assertions": ["error message visible", "still on /login"],
                },
                {
                    "name": "empty_fields_validation",
                    "priority": "medium",
                    "steps": ["goto /login", "click Sign In without filling"],
                    "assertions": ["validation errors shown"],
                },
            ],
            "estimated_duration_minutes": 5,
        },
        grading_rubric={
            "covered_happy_path": 0.3,
            "covered_error_cases": 0.3,
            "logical_step_order": 0.2,
            "appropriate_assertions": 0.2,
        },
    ),

    TestCase(
        id="tp_002",
        name="E-commerce Checkout Flow",
        category="test_planning",
        difficulty=Difficulty.HARD,
        description="Generate comprehensive checkout test plan",
        input_data={
            "app_url": "https://shop.example.com",
            "testable_surfaces": [
                {"type": "page", "path": "/cart"},
                {"type": "page", "path": "/checkout"},
                {"type": "page", "path": "/checkout/shipping"},
                {"type": "page", "path": "/checkout/payment"},
                {"type": "page", "path": "/checkout/confirmation"},
                {"type": "form", "id": "shipping-form"},
                {"type": "form", "id": "payment-form"},
                {"type": "button", "text": "Place Order"},
            ],
            "requirements": """
            Test complete checkout flow including:
            - Cart with items
            - Shipping address entry
            - Payment processing (card and PayPal)
            - Order confirmation
            - Edge cases: empty cart, invalid card, expired session
            """,
        },
        expected_output={
            "test_count_minimum": 8,
            "must_cover": [
                "complete_checkout_success",
                "empty_cart_checkout_blocked",
                "invalid_shipping_address",
                "invalid_credit_card",
                "paypal_payment_flow",
                "session_timeout_handling",
                "order_confirmation_details",
            ],
            "should_have_priorities": True,
            "should_have_test_data": True,
        },
        grading_rubric={
            "comprehensive_coverage": 0.3,
            "edge_cases_identified": 0.25,
            "payment_variations": 0.2,
            "proper_prioritization": 0.15,
            "test_data_included": 0.1,
        },
    ),
]

# =============================================================================
# SELF-HEALING TEST CASES
# =============================================================================

SELF_HEALING_CASES = [
    TestCase(
        id="sh_001",
        name="Selector Changed - Button ID",
        category="self_healing",
        difficulty=Difficulty.EASY,
        description="Heal a test when button ID changed",
        input_data={
            "failed_test": {
                "name": "login_submit",
                "failed_step": "click button#login-btn",
                "error": "Element not found: button#login-btn",
            },
            "current_html": '''
            <form id="login-form">
                <input type="email" id="email" />
                <input type="password" id="password" />
                <button id="submit-login" type="submit">Sign In</button>
            </form>
            ''',
            "git_diff": '''
            - <button id="login-btn" type="submit">Sign In</button>
            + <button id="submit-login" type="submit">Sign In</button>
            ''',
        },
        expected_output={
            "healing_type": "UPDATE_SELECTOR",
            "old_selector": "button#login-btn",
            "new_selector": "button#submit-login",
            "confidence": 0.95,
            "reasoning": "Button ID changed from 'login-btn' to 'submit-login' in recent commit",
        },
        grading_rubric={
            "correct_new_selector": 0.5,
            "high_confidence": 0.2,
            "used_git_history": 0.2,
            "clear_reasoning": 0.1,
        },
    ),

    TestCase(
        id="sh_002",
        name="Timing Issue - Slow API",
        category="self_healing",
        difficulty=Difficulty.MEDIUM,
        description="Heal a test failing due to slow API response",
        input_data={
            "failed_test": {
                "name": "load_user_profile",
                "failed_step": "assert element #user-name is visible",
                "error": "Timeout waiting for #user-name",
                "step_duration_ms": 5000,
            },
            "network_logs": [
                {"url": "/api/user/profile", "duration_ms": 4800, "status": 200},
            ],
            "previous_runs": [
                {"duration_ms": 1200, "passed": True},
                {"duration_ms": 1500, "passed": True},
                {"duration_ms": 4900, "passed": False},
            ],
        },
        expected_output={
            "healing_type": "INCREASE_TIMEOUT",
            "current_timeout_ms": 5000,
            "suggested_timeout_ms": 10000,
            "root_cause": "API response time increased",
            "confidence": 0.85,
        },
        grading_rubric={
            "identified_timing_issue": 0.4,
            "suggested_appropriate_timeout": 0.3,
            "analyzed_network_logs": 0.2,
            "considered_historical_data": 0.1,
        },
    ),

    TestCase(
        id="sh_003",
        name="UI Redesign - Multiple Changes",
        category="self_healing",
        difficulty=Difficulty.HARD,
        description="Heal multiple selectors after UI redesign",
        input_data={
            "failed_test": {
                "name": "checkout_flow",
                "failures": [
                    {"step": "click .checkout-btn", "error": "Element not found"},
                    {"step": "fill #card-number", "error": "Element not found"},
                    {"step": "click .place-order", "error": "Element not found"},
                ],
            },
            "current_html": '''
            <div class="cart-summary">
                <button data-testid="proceed-to-checkout">Checkout</button>
            </div>
            <form data-testid="payment-form">
                <input data-testid="credit-card-input" placeholder="Card Number" />
                <button data-testid="submit-order">Complete Purchase</button>
            </form>
            ''',
            "design_system_update": True,
        },
        expected_output={
            "healing_type": "UPDATE_SELECTOR",
            "changes": [
                {
                    "old": ".checkout-btn",
                    "new": "[data-testid='proceed-to-checkout']",
                },
                {
                    "old": "#card-number",
                    "new": "[data-testid='credit-card-input']",
                },
                {
                    "old": ".place-order",
                    "new": "[data-testid='submit-order']",
                },
            ],
            "confidence": 0.90,
            "pattern_detected": "Migration to data-testid attributes",
        },
        grading_rubric={
            "all_selectors_updated": 0.4,
            "used_stable_selectors": 0.25,
            "detected_pattern": 0.2,
            "high_confidence": 0.15,
        },
    ),
]

# =============================================================================
# NLP UNDERSTANDING TEST CASES
# =============================================================================

NLP_UNDERSTANDING_CASES = [
    TestCase(
        id="nlp_001",
        name="Simple Natural Language Test",
        category="nlp_understanding",
        difficulty=Difficulty.EASY,
        description="Convert natural language to test steps",
        input_data={
            "natural_language": "Log in with email 'test@example.com' and password 'secret123'",
            "app_context": {"login_page": "/login"},
        },
        expected_output={
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "fill", "target": "email", "value": "test@example.com"},
                {"action": "fill", "target": "password", "value": "secret123"},
                {"action": "click", "target": "submit"},
            ],
        },
        grading_rubric={
            "correct_action_sequence": 0.4,
            "extracted_test_data": 0.3,
            "inferred_navigation": 0.2,
            "complete_steps": 0.1,
        },
    ),

    TestCase(
        id="nlp_002",
        name="Complex User Flow Description",
        category="nlp_understanding",
        difficulty=Difficulty.HARD,
        description="Convert complex natural language scenario to test",
        input_data={
            "natural_language": """
            As a returning customer, I want to:
            1. Search for 'wireless headphones'
            2. Filter by price under $100
            3. Sort by customer rating
            4. Add the top-rated item to my cart
            5. Apply promo code 'SAVE20'
            6. Verify the discount is applied correctly
            7. Proceed to checkout but don't complete the purchase
            """,
            "app_context": {
                "search_page": "/search",
                "cart_page": "/cart",
                "checkout_page": "/checkout",
            },
        },
        expected_output={
            "test_name": "returning_customer_search_and_cart",
            "step_count_minimum": 10,
            "must_include_actions": ["search", "filter", "sort", "add_to_cart", "apply_promo"],
            "must_include_assertions": ["discount_applied", "cart_updated"],
        },
        grading_rubric={
            "understood_user_intent": 0.3,
            "correct_action_sequence": 0.25,
            "included_assertions": 0.2,
            "handled_conditional_logic": 0.15,
            "test_data_extracted": 0.1,
        },
    ),
]

# =============================================================================
# VISUAL AI TEST CASES
# =============================================================================

VISUAL_AI_CASES = [
    TestCase(
        id="va_001",
        name="Detect Layout Regression",
        category="visual_ai",
        difficulty=Difficulty.MEDIUM,
        description="Detect significant layout change vs minor styling",
        input_data={
            "baseline_description": "Two-column layout with sidebar on left, content on right",
            "current_description": "Single-column layout, sidebar collapsed into hamburger menu",
            "viewport": "desktop",
        },
        expected_output={
            "is_regression": True,
            "difference_type": "LAYOUT",
            "severity": "high",
            "description": "Major layout change: sidebar navigation restructured",
        },
        grading_rubric={
            "correctly_identified_regression": 0.4,
            "correct_difference_type": 0.3,
            "appropriate_severity": 0.2,
            "clear_description": 0.1,
        },
    ),

    TestCase(
        id="va_002",
        name="Ignore Dynamic Content",
        category="visual_ai",
        difficulty=Difficulty.MEDIUM,
        description="Correctly ignore timestamps and dynamic data",
        input_data={
            "baseline_elements": [
                {"type": "timestamp", "text": "Updated: Jan 1, 2025 10:00 AM"},
                {"type": "count", "text": "15 items"},
                {"type": "heading", "text": "Dashboard"},
            ],
            "current_elements": [
                {"type": "timestamp", "text": "Updated: Jan 20, 2026 3:45 PM"},
                {"type": "count", "text": "23 items"},
                {"type": "heading", "text": "Dashboard"},
            ],
        },
        expected_output={
            "is_regression": False,
            "ignored_differences": ["timestamp", "count"],
            "reasoning": "Dynamic content changes are expected and not regressions",
        },
        grading_rubric={
            "correctly_no_regression": 0.4,
            "identified_dynamic_content": 0.3,
            "clear_reasoning": 0.2,
            "no_false_positives": 0.1,
        },
    ),
]

# =============================================================================
# ORCHESTRATION TEST CASES
# =============================================================================

ORCHESTRATION_CASES = [
    TestCase(
        id="orch_001",
        name="Parallel Test Execution",
        category="orchestration",
        difficulty=Difficulty.MEDIUM,
        description="Verify parallel execution with proper resource management",
        input_data={
            "test_count": 10,
            "max_parallel": 3,
            "test_durations_ms": [1000, 2000, 500, 1500, 800, 1200, 900, 1100, 700, 1300],
        },
        expected_output={
            "executed_in_parallel": True,
            "max_concurrent": 3,
            "total_duration_less_than_serial": True,
            "all_tests_completed": True,
        },
        grading_rubric={
            "respected_parallelism_limit": 0.3,
            "efficient_scheduling": 0.3,
            "all_tests_ran": 0.2,
            "proper_resource_cleanup": 0.2,
        },
    ),

    TestCase(
        id="orch_002",
        name="Failure Recovery and Healing",
        category="orchestration",
        difficulty=Difficulty.HARD,
        description="Verify orchestrator handles failures and triggers healing",
        input_data={
            "test_plan": [
                {"id": "t1", "should_pass": True},
                {"id": "t2", "should_fail": True, "healable": True},
                {"id": "t3", "should_pass": True},
                {"id": "t4", "should_fail": True, "healable": False},
                {"id": "t5", "should_pass": True},
            ],
        },
        expected_output={
            "total_executed": 5,
            "initial_passed": 3,
            "initial_failed": 2,
            "healing_attempted": 2,
            "healed_successfully": 1,
            "final_passed": 4,
            "continued_after_unhealed_failure": True,
        },
        grading_rubric={
            "executed_all_tests": 0.2,
            "triggered_healing_correctly": 0.3,
            "handled_unhealable_failure": 0.2,
            "continued_execution": 0.2,
            "correct_final_counts": 0.1,
        },
    ),
]


# =============================================================================
# ALL TEST CASES
# =============================================================================

ALL_TEST_CASES = (
    CODE_ANALYSIS_CASES +
    TEST_PLANNING_CASES +
    SELF_HEALING_CASES +
    NLP_UNDERSTANDING_CASES +
    VISUAL_AI_CASES +
    ORCHESTRATION_CASES
)


def get_test_cases_by_category(category: str) -> list[TestCase]:
    """Get test cases filtered by category."""
    return [tc for tc in ALL_TEST_CASES if tc.category == category]


def get_test_cases_by_difficulty(difficulty: Difficulty) -> list[TestCase]:
    """Get test cases filtered by difficulty."""
    return [tc for tc in ALL_TEST_CASES if tc.difficulty == difficulty]

"""
Input and Output Validators for Guardrails.

Provides specialized validation components:
- Prompt injection detection
- Input sanitization
- Output filtering (PII, hallucination)
"""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    sanitized_input: Any | None = None
    confidence: float = 1.0


class PromptInjectionDetector:
    """
    Detects potential prompt injection attacks.

    Based on common injection patterns observed in production systems.

    RAP-331: Enhanced with additional adversarial patterns including:
    - Unicode/homoglyph obfuscation attempts
    - Base64 encoded payloads
    - Markdown/HTML injection
    - Context manipulation via fake conversation history
    """

    # Patterns indicating instruction override attempts
    OVERRIDE_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|guidelines?)",
        r"disregard\s+(your|all|any)\s+(instructions?|rules?|guidelines?)",
        r"forget\s+(everything|all)\s+(you\s+)?know",
        r"you\s+are\s+now\s+(a|an|in)\s+(different|new|unrestricted)",
        r"pretend\s+(you\s+are|to\s+be)\s+a",
        r"act\s+as\s+if\s+you\s+(have\s+no|don't\s+have)",
        r"your\s+new\s+instructions?\s+(are|is)",
        r"override\s+(all\s+)?(safety|security)\s+(protocols?|measures?)",
        # RAP-331: Additional override patterns
        r"from\s+now\s+on,?\s+(you\s+)?(will|must|should)",
        r"stop\s+being\s+(an?\s+)?ai",
        r"reset\s+(your|all)\s+(context|memory|instructions?)",
        r"new\s+conversation\s*:\s*you\s+are",
    ]

    # Patterns indicating system prompt extraction
    EXTRACTION_PATTERNS = [
        r"(print|show|display|reveal|tell)\s+(me\s+)?(your|the)\s+(system\s+)?prompt",
        r"what\s+(are|is)\s+your\s+(instructions?|rules?|system\s+prompt)",
        r"repeat\s+(your|the)\s+(instructions?|prompt)",
        r"output\s+everything\s+(before|above)",
        # RAP-331: Additional extraction patterns
        r"(echo|write|dump)\s+(your|the)\s+system",
        r"(show|reveal)\s+(hidden|secret|initial)\s+(prompt|instructions?)",
        r"what\s+were\s+you\s+told\s+(to\s+do|initially|before)",
        r"(list|enumerate)\s+your\s+(rules?|constraints?|limitations?)",
    ]

    # Patterns indicating role manipulation
    ROLE_PATTERNS = [
        r"you\s+are\s+(now\s+)?(DAN|jailbroken|unrestricted)",
        r"entering?\s+(developer|admin|debug)\s+mode",
        r"bypass\s+(all\s+)?(restrictions?|filters?|safety)",
        r"\[?(system|user|assistant)\]?\s*:",
        r"</?(?:system|user|assistant|instruction)>",
        # RAP-331: Additional role patterns
        r"roleplay\s+as\s+(an?\s+)?(evil|malicious|unrestricted)",
        r"(simulate|emulate)\s+(being\s+)?(hacked|compromised|jailbroken)",
        r"activate\s+(god|admin|sudo|root)\s+mode",
        r"disable\s+(safety|content\s+)?(filter|moderation|policy)",
        r"hypothetically,?\s+(if\s+)?you\s+(had|were)\s+(no|without)\s+(restrictions?|limits?)",
    ]

    # Delimiter injection patterns
    DELIMITER_PATTERNS = [
        r"```+\s*\n?\s*(system|instruction)",
        r"={5,}",  # Separator injection
        r"-{5,}",
        r"END\s+OF\s+(PROMPT|INSTRUCTIONS?)",
        # RAP-331: Additional delimiter patterns
        r"\[\[SYSTEM\]\]",
        r"###\s*(SYSTEM|INSTRUCTION|OVERRIDE)",
        r"<\|.*?\|>",  # Token-like delimiters
        r"\\n\\n---\\n\\nSYSTEM:",  # Escaped newlines with fake system
    ]

    # RAP-331: New category - Obfuscation patterns
    OBFUSCATION_PATTERNS = [
        r"base64:\s*[A-Za-z0-9+/=]{20,}",  # Embedded base64
        r"(?:&#x?[0-9a-fA-F]+;){5,}",  # HTML entity sequences
        r"\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}",  # Unicode escapes
        r"(?:[\u200b-\u200f\u202a-\u202e\u2060-\u206f]){2,}",  # Zero-width/formatting chars
        r"(?:eval|exec|import)\s*\(",  # Code execution attempts
        r"(?:javascript|data|vbscript):",  # Protocol handlers
    ]

    # RAP-331: New category - Fake context injection
    CONTEXT_INJECTION_PATTERNS = [
        r"(?:user|human|customer)\s*said:\s*[\"']?yes",  # Fake user agreement
        r"(?:assistant|ai|bot)\s*(?:said|responded):\s*[\"']?(?:ok|sure|certainly)",  # Fake AI agreement
        r"previous\s+conversation:\s*\n",  # Fake history injection
        r"context:\s*the\s+user\s+has\s+(?:permission|authorization|admin)",  # Fake permissions
        r"\[internal\s+note\]",  # Fake internal notes
        r"\[admin\s+override\]",  # Fake admin commands
    ]

    def __init__(self):
        self.log = logger.bind(component="prompt_injection_detector")

        # Compile patterns for efficiency
        self.all_patterns = []
        for pattern_list in [
            self.OVERRIDE_PATTERNS,
            self.EXTRACTION_PATTERNS,
            self.ROLE_PATTERNS,
            self.DELIMITER_PATTERNS,
            self.OBFUSCATION_PATTERNS,  # RAP-331
            self.CONTEXT_INJECTION_PATTERNS,  # RAP-331
        ]:
            for pattern in pattern_list:
                self.all_patterns.append(
                    (pattern, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
                )

    def detect(self, text: str) -> ValidationResult:
        """
        Check text for prompt injection patterns.

        Returns ValidationResult with is_valid=False if injection detected.

        Confidence scores by pattern category:
        - Override patterns: 0.95 (highest - direct instruction override)
        - Role patterns: 0.90 (jailbreak/role manipulation)
        - Context injection: 0.88 (fake history/permissions)
        - Extraction patterns: 0.85 (system prompt extraction)
        - Obfuscation patterns: 0.80 (encoded/hidden payloads)
        - Delimiter patterns: 0.75 (boundary manipulation)
        """
        issues = []
        max_confidence = 0.0

        for pattern_str, compiled in self.all_patterns:
            match = compiled.search(text)
            if match:
                matched_text = match.group()
                # Truncate long matches for logging
                truncated = matched_text[:50] + "..." if len(matched_text) > 50 else matched_text
                issue = f"Injection pattern detected: '{truncated}'"
                issues.append(issue)

                # Different pattern categories have different confidence scores
                if pattern_str in self.OVERRIDE_PATTERNS:
                    max_confidence = max(max_confidence, 0.95)
                elif pattern_str in self.ROLE_PATTERNS:
                    max_confidence = max(max_confidence, 0.90)
                elif pattern_str in self.CONTEXT_INJECTION_PATTERNS:  # RAP-331
                    max_confidence = max(max_confidence, 0.88)
                elif pattern_str in self.EXTRACTION_PATTERNS:
                    max_confidence = max(max_confidence, 0.85)
                elif pattern_str in self.OBFUSCATION_PATTERNS:  # RAP-331
                    max_confidence = max(max_confidence, 0.80)
                else:  # DELIMITER_PATTERNS
                    max_confidence = max(max_confidence, 0.75)

        if issues:
            self.log.warning(
                "Prompt injection detected",
                issues=issues,
                confidence=max_confidence,
                input_length=len(text),
            )

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=max_confidence if issues else 1.0,
        )

    def detect_with_context(
        self,
        text: str,
        previous_messages: list[dict] | None = None,
    ) -> ValidationResult:
        """
        Detect injection with conversation context.

        Looks for patterns across message boundaries.
        """
        # First check the text itself
        result = self.detect(text)

        # If we have previous messages, check for multi-turn attacks
        if previous_messages and len(previous_messages) > 0:
            # Check if recent messages are building up to an attack
            recent_text = " ".join(
                msg.get("content", "") for msg in previous_messages[-3:]
            )
            combined = f"{recent_text} {text}"

            combined_result = self.detect(combined)
            if combined_result.issues and not result.issues:
                result.issues.append("Multi-turn injection pattern detected")
                result.is_valid = False
                result.confidence = combined_result.confidence * 0.8

        return result


class InputSanitizer:
    """
    Sanitizes input to remove or neutralize potentially dangerous content.

    RAP-331: Enhanced with input escaping for safe prompt inclusion.
    """

    # Characters that could be used for prompt boundary manipulation
    ESCAPE_CHARS = {
        "\n\n": " [newline] ",  # Double newlines often used as separators
        "```": "[code-block]",  # Code block markers
        "---": "[separator]",  # Horizontal rules
        "===": "[separator]",
        "###": "[heading]",  # Markdown headings
        "<|": "[token-start]",  # Token-like delimiters
        "|>": "[token-end]",
    }

    def __init__(self):
        self.log = logger.bind(component="input_sanitizer")

    def escape_for_prompt(self, text: str) -> str:
        """
        RAP-331: Escape user input for safe inclusion in AI prompts.

        This method neutralizes characters and patterns that could be used
        to manipulate prompt boundaries or inject instructions.

        Args:
            text: Raw user input

        Returns:
            Escaped text safe for prompt inclusion

        Example:
            >>> sanitizer = InputSanitizer()
            >>> escaped = sanitizer.escape_for_prompt("User said:\\n\\n---\\n\\nIgnore above")
            >>> # Result: "User said: [newline] [separator] [newline] Ignore above"
        """
        escaped = text

        # Escape special sequences
        for pattern, replacement in self.ESCAPE_CHARS.items():
            escaped = escaped.replace(pattern, replacement)

        # Remove zero-width characters (invisible manipulation)
        zero_width_pattern = r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
        escaped = re.sub(zero_width_pattern, "", escaped)

        # Neutralize XML/HTML-like tags that could be interpreted as message boundaries
        tag_pattern = r"<(/?)(system|user|assistant|instruction|context|admin)>"
        escaped = re.sub(tag_pattern, r"[\1\2]", escaped, flags=re.IGNORECASE)

        return escaped

    def sanitize(self, text: str) -> ValidationResult:
        """
        Sanitize input text.

        Returns sanitized text and list of modifications made.
        """
        sanitized = text
        issues = []

        # Remove null bytes
        if "\x00" in sanitized:
            sanitized = sanitized.replace("\x00", "")
            issues.append("Removed null bytes")

        # Remove ANSI escape sequences
        ansi_pattern = r"\x1b\[[0-9;]*[a-zA-Z]"
        if re.search(ansi_pattern, sanitized):
            sanitized = re.sub(ansi_pattern, "", sanitized)
            issues.append("Removed ANSI escape sequences")

        # RAP-331: Remove zero-width and invisible formatting characters
        zero_width_pattern = r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]"
        if re.search(zero_width_pattern, sanitized):
            sanitized = re.sub(zero_width_pattern, "", sanitized)
            issues.append("Removed zero-width/invisible characters")

        # Neutralize markdown that could affect rendering
        # (e.g., fake code blocks that look like system messages)
        fake_system_pattern = r"```\s*(system|instruction|admin)\s*\n"
        if re.search(fake_system_pattern, sanitized, re.IGNORECASE):
            sanitized = re.sub(
                fake_system_pattern,
                "```text\n[sanitized] ",
                sanitized,
                flags=re.IGNORECASE,
            )
            issues.append("Neutralized fake system code blocks")

        # RAP-331: Neutralize XML-like message boundary tags
        tag_pattern = r"<(/?)(system|user|assistant|instruction|context)>"
        if re.search(tag_pattern, sanitized, re.IGNORECASE):
            sanitized = re.sub(tag_pattern, r"[\1\2]", sanitized, flags=re.IGNORECASE)
            issues.append("Neutralized message boundary tags")

        # Limit extreme repetition (potential DoS)
        repetition_pattern = r"(.)\1{100,}"
        if re.search(repetition_pattern, sanitized):
            sanitized = re.sub(repetition_pattern, r"\1\1\1...[truncated]", sanitized)
            issues.append("Truncated extreme character repetition")

        return ValidationResult(
            is_valid=True,
            issues=issues,
            sanitized_input=sanitized,
        )

    def sanitize_for_shell(self, command: str) -> ValidationResult:
        """
        Sanitize input intended for shell execution.

        More aggressive sanitization for security.
        """
        sanitized = command
        issues = []

        # Check for command chaining that could bypass intent
        dangerous_chains = [
            (r";\s*rm\s+-rf", "Blocked rm -rf after chain"),
            (r"\|\s*sh\b", "Blocked pipe to sh"),
            (r"\|\s*bash\b", "Blocked pipe to bash"),
            (r"\$\(.*rm\s", "Blocked command substitution with rm"),
            (r"`.*rm\s", "Blocked backtick with rm"),
        ]

        for pattern, message in dangerous_chains:
            if re.search(pattern, sanitized, re.IGNORECASE):
                issues.append(message)
                # Don't modify, just flag - let guardrail stack decide
                return ValidationResult(
                    is_valid=False,
                    issues=issues,
                    sanitized_input=None,
                )

        return ValidationResult(
            is_valid=True,
            issues=issues,
            sanitized_input=sanitized,
        )


class OutputFilter:
    """
    Filters agent output to remove sensitive information.
    """

    # Patterns for sensitive data
    PII_PATTERNS = {
        "api_key": r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?",
        "password": r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
        "secret": r"(?:secret|token|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?",
        "aws_key": r"(?:AKIA|ABIA|ACCA)[A-Z0-9]{16}",
        "private_key": r"-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----",
        "jwt": r"eyJ[a-zA-Z0-9\-_]+\.eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+",
    }

    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PHONE_PATTERN = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
    CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"

    def __init__(self, redact_pii: bool = True, redact_credentials: bool = True):
        self.log = logger.bind(component="output_filter")
        self.redact_pii = redact_pii
        self.redact_credentials = redact_credentials

    def filter(self, text: str) -> tuple[str, list[str]]:
        """
        Filter output text for sensitive content.

        Returns:
            Tuple of (filtered_text, list of redaction types applied)
        """
        filtered = text
        redactions = []

        if self.redact_credentials:
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, filtered, re.IGNORECASE):
                    filtered = re.sub(
                        pattern,
                        f"[REDACTED_{pii_type.upper()}]",
                        filtered,
                        flags=re.IGNORECASE,
                    )
                    redactions.append(f"{pii_type} redacted")

        if self.redact_pii:
            # Email - partial redaction
            if re.search(self.EMAIL_PATTERN, filtered):
                filtered = re.sub(
                    self.EMAIL_PATTERN,
                    lambda m: f"{m.group()[:2]}***@{m.group().split('@')[1]}",
                    filtered,
                )
                redactions.append("emails partially redacted")

            # Phone numbers
            if re.search(self.PHONE_PATTERN, filtered):
                filtered = re.sub(
                    self.PHONE_PATTERN,
                    "[REDACTED_PHONE]",
                    filtered,
                )
                redactions.append("phone numbers redacted")

            # SSN
            if re.search(self.SSN_PATTERN, filtered):
                filtered = re.sub(
                    self.SSN_PATTERN,
                    "[REDACTED_SSN]",
                    filtered,
                )
                redactions.append("SSN redacted")

            # Credit cards
            if re.search(self.CREDIT_CARD_PATTERN, filtered):
                filtered = re.sub(
                    self.CREDIT_CARD_PATTERN,
                    "[REDACTED_CARD]",
                    filtered,
                )
                redactions.append("credit card numbers redacted")

        if redactions:
            self.log.info(
                "Output filtered",
                redactions=redactions,
                original_length=len(text),
                filtered_length=len(filtered),
            )

        return filtered, redactions

    def check_hallucination_indicators(self, text: str) -> list[str]:
        """
        Check for common hallucination indicators.

        Returns list of potential hallucination warnings.
        """
        warnings = []

        # Fake URLs
        fake_url_patterns = [
            r"https?://example\.com/real-api",
            r"https?://api\.fake",
            r"https?://.*\.test/",
        ]
        for pattern in fake_url_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                warnings.append("Contains potentially fake/example URLs")
                break

        # Hedging language that might indicate uncertainty
        hedging_patterns = [
            r"I(?:'m| am) not (?:entirely )?sure",
            r"(?:probably|likely|possibly|maybe) (?:would|could|should)",
            r"(?:I )?(?:believe|think|assume) (?:this|that|it)",
        ]
        hedge_count = 0
        for pattern in hedging_patterns:
            hedge_count += len(re.findall(pattern, text, re.IGNORECASE))

        if hedge_count >= 3:
            warnings.append("High uncertainty language detected")

        # Contradictions within same response
        contradiction_pairs = [
            (r"this (?:is|will) work", r"this (?:won't|will not) work"),
            (r"you should", r"you should not"),
            (r"it is possible", r"it is (?:not |im)possible"),
        ]
        for affirm, negate in contradiction_pairs:
            if re.search(affirm, text, re.IGNORECASE) and re.search(
                negate, text, re.IGNORECASE
            ):
                warnings.append("Potential self-contradiction detected")
                break

        return warnings

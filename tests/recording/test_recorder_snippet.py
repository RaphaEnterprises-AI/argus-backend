"""Tests for recorder snippet generator."""

import pytest
from src.recording.recorder_snippet import (
    RecorderConfig,
    RecorderSnippetGenerator,
    generate_snippet,
)


# =============================================================================
# RecorderConfig Tests
# =============================================================================


class TestRecorderConfig:
    """Tests for RecorderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RecorderConfig()

        assert config.block_class == "rr-block"
        assert config.ignore_class == "rr-ignore"
        assert config.mask_text_class == "rr-mask"
        assert config.mask_all_inputs is False
        assert config.auto_start is True
        assert config.upload_interval_ms == 10000

    def test_custom_config(self):
        """Test custom configuration."""
        config = RecorderConfig(
            mask_all_inputs=True,
            upload_interval_ms=5000,
            max_events_per_upload=500
        )

        assert config.mask_all_inputs is True
        assert config.upload_interval_ms == 5000
        assert config.max_events_per_upload == 500

    def test_default_sampling(self):
        """Test default sampling configuration."""
        config = RecorderConfig()

        assert config.sampling["mousemove"] is True
        assert config.sampling["mouseInteraction"] is True
        assert config.sampling["scroll"] == 150
        assert config.sampling["input"] == "last"

    def test_custom_sampling(self):
        """Test custom sampling configuration."""
        config = RecorderConfig(
            sampling={
                "mousemove": False,
                "scroll": 200
            }
        )

        assert config.sampling["mousemove"] is False
        assert config.sampling["scroll"] == 200


# =============================================================================
# RecorderSnippetGenerator Tests
# =============================================================================


class TestRecorderSnippetGenerator:
    """Tests for RecorderSnippetGenerator class."""

    def test_create_generator(self):
        """Test creating generator with default config."""
        generator = RecorderSnippetGenerator()
        assert generator.config is not None

    def test_create_generator_custom_config(self):
        """Test creating generator with custom config."""
        config = RecorderConfig(mask_all_inputs=True)
        generator = RecorderSnippetGenerator(config)
        assert generator.config.mask_all_inputs is True


class TestInlineSnippet:
    """Tests for inline JavaScript snippet generation."""

    def test_generate_inline_basic(self):
        """Test basic inline snippet generation."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "<script>" in snippet
        assert "</script>" in snippet
        assert "rrweb" in snippet

    def test_generate_inline_with_credentials(self):
        """Test inline snippet includes API key and project ID."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet(
            api_key="test-api-key",
            project_id="proj-123"
        )

        assert "test-api-key" in snippet
        assert "proj-123" in snippet

    def test_generate_inline_has_rrweb_cdn(self):
        """Test snippet loads rrweb from CDN."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "cdn.jsdelivr.net" in snippet
        assert "rrweb" in snippet

    def test_generate_inline_has_record_call(self):
        """Test snippet calls rrweb.record()."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "rrweb.record(" in snippet

    def test_generate_inline_has_upload_function(self):
        """Test snippet has upload function."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "uploadEvents" in snippet
        assert "fetch" in snippet

    def test_generate_inline_has_unload_handler(self):
        """Test snippet handles page unload."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "beforeunload" in snippet
        assert "sendBeacon" in snippet

    def test_generate_inline_uses_config(self):
        """Test snippet uses configuration options."""
        config = RecorderConfig(
            mask_all_inputs=True,
            upload_interval_ms=5000
        )
        generator = RecorderSnippetGenerator(config)
        snippet = generator.generate_inline_snippet()

        assert "maskAllInputs: true" in snippet
        assert "5000" in snippet


class TestNPMSnippet:
    """Tests for NPM/ES module snippet generation."""

    def test_generate_npm_basic(self):
        """Test basic NPM snippet generation."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_npm_snippet()

        assert "import" in snippet
        assert "from 'rrweb'" in snippet
        assert "export" in snippet

    def test_generate_npm_has_init_function(self):
        """Test NPM snippet has init function."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_npm_snippet()

        assert "initArgusRecording" in snippet

    def test_generate_npm_has_typescript_types(self):
        """Test NPM snippet includes TypeScript types."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_npm_snippet()

        assert ": any[]" in snippet or "any[]" in snippet

    def test_generate_npm_with_credentials(self):
        """Test NPM snippet includes credentials."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_npm_snippet(
            api_key="npm-key",
            project_id="npm-proj"
        )

        assert "npm-key" in snippet
        assert "npm-proj" in snippet


class TestReactHook:
    """Tests for React hook generation."""

    def test_generate_react_hook(self):
        """Test React hook generation."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_react_hook()

        assert "useEffect" in snippet
        assert "useRef" in snippet
        assert "useArgusRecording" in snippet

    def test_generate_react_hook_returns_session_id(self):
        """Test React hook returns session ID."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_react_hook()

        assert "sessionIdRef.current" in snippet
        assert "return sessionIdRef" in snippet

    def test_generate_react_hook_has_cleanup(self):
        """Test React hook has cleanup."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_react_hook()

        # Should have return cleanup function in useEffect
        assert "clearInterval" in snippet
        assert "stopFn" in snippet

    def test_generate_react_hook_enabled_param(self):
        """Test React hook has enabled parameter."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_react_hook()

        assert "enabled = true" in snippet or "enabled: boolean" in snippet


class TestGTMSnippet:
    """Tests for Google Tag Manager snippet generation."""

    def test_generate_gtm_snippet(self):
        """Test GTM snippet generation."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_gtm_snippet()

        assert "<script>" in snippet
        assert "GTM" in snippet or "rrweb" in snippet

    def test_generate_gtm_has_variable_placeholders(self):
        """Test GTM snippet uses GTM variable syntax."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_gtm_snippet()

        assert "{{Argus API Key}}" in snippet or "Argus" in snippet


# =============================================================================
# generate_snippet Convenience Function Tests
# =============================================================================


class TestGenerateSnippetFunction:
    """Tests for generate_snippet convenience function."""

    def test_generate_inline_format(self):
        """Test generate inline format."""
        snippet = generate_snippet(format="inline")
        assert "<script>" in snippet

    def test_generate_npm_format(self):
        """Test generate npm format."""
        snippet = generate_snippet(format="npm")
        assert "import" in snippet

    def test_generate_react_format(self):
        """Test generate react format."""
        snippet = generate_snippet(format="react")
        assert "useEffect" in snippet

    def test_generate_gtm_format(self):
        """Test generate gtm format."""
        snippet = generate_snippet(format="gtm")
        assert "<script>" in snippet

    def test_generate_with_config(self):
        """Test generate with custom config."""
        config = RecorderConfig(mask_all_inputs=True)
        snippet = generate_snippet(
            format="inline",
            config=config
        )
        assert "maskAllInputs: true" in snippet

    def test_generate_invalid_format(self):
        """Test generate raises error for invalid format."""
        with pytest.raises(ValueError) as exc_info:
            generate_snippet(format="invalid")

        assert "Unknown snippet format" in str(exc_info.value)


# =============================================================================
# Configuration Propagation Tests
# =============================================================================


class TestConfigPropagation:
    """Tests for configuration propagation to snippets."""

    def test_block_class_in_snippet(self):
        """Test block_class is in snippet."""
        config = RecorderConfig(block_class="custom-block")
        generator = RecorderSnippetGenerator(config)
        snippet = generator.generate_inline_snippet()

        assert "custom-block" in snippet

    def test_ignore_class_in_snippet(self):
        """Test ignore_class is in snippet."""
        config = RecorderConfig(ignore_class="custom-ignore")
        generator = RecorderSnippetGenerator(config)
        snippet = generator.generate_inline_snippet()

        assert "custom-ignore" in snippet

    def test_mask_text_selector_in_snippet(self):
        """Test mask_text_selector is in snippet."""
        config = RecorderConfig(mask_text_selector=".sensitive-field")
        generator = RecorderSnippetGenerator(config)
        snippet = generator.generate_inline_snippet()

        assert ".sensitive-field" in snippet

    def test_upload_endpoint_in_snippet(self):
        """Test upload_endpoint is in snippet."""
        config = RecorderConfig(upload_endpoint="/custom/upload")
        generator = RecorderSnippetGenerator(config)
        snippet = generator.generate_inline_snippet()

        assert "/custom/upload" in snippet


# =============================================================================
# Security Tests
# =============================================================================


class TestSnippetSecurity:
    """Tests for security aspects of generated snippets."""

    def test_password_masking_enabled(self):
        """Test password fields are masked by default."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "input[type=password]" in snippet

    def test_api_key_header(self):
        """Test API key is sent in header."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "X-Argus-API-Key" in snippet

    def test_keepalive_for_uploads(self):
        """Test fetch uses keepalive for reliability."""
        generator = RecorderSnippetGenerator()
        snippet = generator.generate_inline_snippet()

        assert "keepalive: true" in snippet

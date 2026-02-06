"""End-to-end tests for HealingService."""
import pytest
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

# Test imports work
def test_imports():
    """Test all healing service imports work."""
    from src.services.healing_service import (
        HealingService,
        TestFailure,
        HealingResult,
        DeploymentMode,
        get_healing_service,
    )
    assert HealingService is not None
    assert TestFailure is not None
    assert HealingResult is not None
    assert DeploymentMode is not None

def test_test_failure_dataclass():
    """Test TestFailure dataclass."""
    from src.services.healing_service import TestFailure
    
    # Minimal creation
    failure = TestFailure(error_type="selector_not_found")
    assert failure.error_type == "selector_not_found"
    assert failure.selector is None
    
    # Full creation
    failure = TestFailure(
        error_type="selector_not_found",
        selector="#login-btn",
        error_message="Element not found",
        page_url="https://example.com/login",
        test_id="test_123",
        failure_id="fail_456",
    )
    assert failure.selector == "#login-btn"

def test_healing_result_dataclass():
    """Test HealingResult dataclass."""
    from src.services.healing_service import HealingResult
    
    # Success result
    result = HealingResult(
        success=True,
        fix_applied={"new_selector": "[data-testid='login']"},
        strategy_used="cognee_pattern",
        confidence=0.95,
    )
    assert result.success is True
    assert result.strategy_used == "cognee_pattern"
    
    # Failure result
    result = HealingResult(
        success=False,
        error="No fix found",
    )
    assert result.success is False

def test_deployment_mode_enum():
    """Test DeploymentMode enum values."""
    from src.services.healing_service import DeploymentMode
    
    assert DeploymentMode.AUTO.value == "auto"
    assert DeploymentMode.LOCAL.value == "local"
    assert DeploymentMode.REMOTE.value == "remote"

def test_healing_service_instantiation():
    """Test HealingService can be instantiated."""
    from src.services.healing_service import HealingService
    
    service = HealingService()
    assert service is not None
    assert hasattr(service, 'heal')

@pytest.mark.asyncio
async def test_heal_method_exists_and_callable():
    """Test heal method exists and has correct signature."""
    from src.services.healing_service import HealingService, TestFailure
    
    service = HealingService()
    
    # Check method exists
    assert hasattr(service, 'heal')
    assert inspect.iscoroutinefunction(service.heal)
    
    # Check signature
    sig = inspect.signature(service.heal)
    params = list(sig.parameters.keys())
    assert 'test_id' in params
    assert 'org_id' in params
    assert 'project_id' in params
    assert 'failure' in params

@pytest.mark.asyncio
async def test_heal_with_mocked_dependencies():
    """Test heal method with mocked dependencies."""
    from src.services.healing_service import HealingService, TestFailure, HealingResult
    
    service = HealingService()
    
    # Mock the internal methods using their actual names
    with patch.object(service, '_try_cognee_pattern', new_callable=AsyncMock) as mock_cognee:
        with patch.object(service, '_try_code_aware_healing', new_callable=AsyncMock) as mock_code:
            with patch.object(service, '_try_llm_healing', new_callable=AsyncMock) as mock_healer:
                with patch.object(service, '_store_successful_pattern', new_callable=AsyncMock) as mock_store:
                    mock_cognee.return_value = None
                    mock_code.return_value = None
                    mock_healer.return_value = HealingResult(
                        success=True,
                        fix_applied={"fix_type": "update_selector", "old_value": "#test-btn", "new_value": "[data-testid='test']"},
                        strategy_used="llm_fallback",
                        confidence=0.8,
                    )
                    mock_store.return_value = None
                    
                    failure = TestFailure(
                        error_type="selector_not_found",
                        selector="#test-btn",
                        error_message="Element not found",
                    )
                    
                    result = await service.heal(
                        test_id="test_1",
                        org_id="org_1",
                        project_id="proj_1",
                        failure=failure,
                    )
                    
                    assert result.success is True
                    assert result.strategy_used == "llm_fallback"

@pytest.mark.asyncio
async def test_heal_cognee_pattern_match():
    """Test heal when Cognee finds a matching pattern."""
    from src.services.healing_service import HealingService, TestFailure, HealingResult
    
    service = HealingService()
    
    # Simulate Cognee pattern found
    cognee_result = HealingResult(
        success=True,
        fix_applied={"fix_type": "update_selector", "old_value": "#login", "new_value": "[data-testid='login-btn']"},
        strategy_used="cognee_pattern",
        confidence=0.92,
        pattern_id="pattern_123",
    )
    
    with patch.object(service, '_try_cognee_pattern', new_callable=AsyncMock) as mock_cognee:
        mock_cognee.return_value = cognee_result
        
        failure = TestFailure(
            error_type="selector_not_found",
            selector="#login",
            error_message="Element #login not found",
        )
        
        result = await service.heal(
            test_id="test_1",
            org_id="org_1",
            project_id="proj_1",
            failure=failure,
        )
        
        assert result.success is True
        assert result.strategy_used == "cognee_pattern"
        assert result.pattern_id == "pattern_123"

@pytest.mark.asyncio
async def test_heal_code_aware_fallback():
    """Test heal falls back to code-aware healing when Cognee has no match."""
    from src.services.healing_service import HealingService, TestFailure, HealingResult
    
    service = HealingService()
    
    code_aware_result = HealingResult(
        success=True,
        fix_applied={"fix_type": "update_selector", "old_value": "#submit", "new_value": "[data-testid='submit-form']"},
        strategy_used="code_aware",
        confidence=0.99,
        code_context={"commit_sha": "abc123", "file_changed": "src/components/Form.tsx"},
    )
    
    with patch.object(service, '_try_cognee_pattern', new_callable=AsyncMock) as mock_cognee:
        with patch.object(service, '_try_code_aware_healing', new_callable=AsyncMock) as mock_code:
            with patch.object(service, '_store_successful_pattern', new_callable=AsyncMock) as mock_store:
                mock_cognee.return_value = None
                mock_code.return_value = code_aware_result
                mock_store.return_value = None
                
                failure = TestFailure(
                    error_type="selector_not_found",
                    selector="#submit",
                    error_message="Element #submit not found",
                )
                
                result = await service.heal(
                    test_id="test_1",
                    org_id="org_1",
                    project_id="proj_1",
                    failure=failure,
                )
                
                assert result.success is True
                assert result.strategy_used == "code_aware"
                assert result.code_context is not None
                # Verify store was called for code-aware success
                mock_store.assert_called_once()

@pytest.mark.asyncio
async def test_heal_all_strategies_fail():
    """Test heal returns failure when all strategies fail."""
    from src.services.healing_service import HealingService, TestFailure
    
    service = HealingService()
    
    with patch.object(service, '_try_cognee_pattern', new_callable=AsyncMock) as mock_cognee:
        with patch.object(service, '_try_code_aware_healing', new_callable=AsyncMock) as mock_code:
            with patch.object(service, '_try_llm_healing', new_callable=AsyncMock) as mock_llm:
                mock_cognee.return_value = None
                mock_code.return_value = None
                mock_llm.return_value = None
                
                failure = TestFailure(
                    error_type="selector_not_found",
                    selector="#unknown-element",
                    error_message="Cannot find element",
                )
                
                result = await service.heal(
                    test_id="test_1",
                    org_id="org_1",
                    project_id="proj_1",
                    failure=failure,
                )
                
                assert result.success is False
                assert "exhausted" in result.error.lower()

@pytest.mark.asyncio
async def test_get_healing_service_factory():
    """Test get_healing_service factory function."""
    from src.services.healing_service import get_healing_service, HealingService
    
    service = get_healing_service(org_id="org_1", project_id="proj_1")
    assert isinstance(service, HealingService)
    assert service.org_id == "org_1"
    assert service.project_id == "proj_1"

@pytest.mark.asyncio
async def test_store_pattern():
    """Test store_pattern method."""
    from src.services.healing_service import HealingService, HealingPatternCreate
    
    service = HealingService(org_id="org_1", project_id="proj_1")
    
    # Mock Supabase client
    mock_supabase = MagicMock()
    mock_supabase.is_configured = True
    mock_supabase.request = AsyncMock(return_value={
        "data": [{"id": "pattern_123", "success_count": 1}],
        "error": None,
    })
    
    with patch.object(service, '_supabase', mock_supabase):
        pattern = HealingPatternCreate(
            original_selector="#old-selector",
            healed_selector="[data-testid='new-selector']",
            error_type="selector_not_found",
            project_id="proj_1",
        )
        
        result = await service.store_pattern(pattern, store_in_cognee=False)
        
        # Should successfully store or update pattern
        assert mock_supabase.request.called

def test_healing_pattern_create_model():
    """Test HealingPatternCreate Pydantic model."""
    from src.services.healing_service import HealingPatternCreate
    
    pattern = HealingPatternCreate(
        original_selector="#test",
        healed_selector="[data-testid='test']",
        error_type="selector_not_found",
        project_id="proj_1",
        page_url="https://example.com",
        element_context={"tag": "button"},
        metadata={"test_id": "test_123"},
    )
    
    assert pattern.original_selector == "#test"
    assert pattern.healed_selector == "[data-testid='test']"
    assert pattern.error_type == "selector_not_found"
    assert pattern.project_id == "proj_1"

def test_healing_pattern_result_model():
    """Test HealingPatternResult Pydantic model."""
    from src.services.healing_service import HealingPatternResult
    
    result = HealingPatternResult(
        success=True,
        pattern_id="pattern_123",
        fingerprint="abc123",
        is_new=True,
        message="Pattern created",
    )
    
    assert result.success is True
    assert result.pattern_id == "pattern_123"
    assert result.is_new is True

@pytest.mark.asyncio
async def test_test_failure_to_dict():
    """Test TestFailure.to_dict() method."""
    from src.services.healing_service import TestFailure
    
    failure = TestFailure(
        error_type="selector_not_found",
        selector="#test-btn",
        error_message="Element not found",
        page_url="https://example.com/page",
        test_id="test_123",
        failure_id="fail_456",
    )
    
    d = failure.to_dict()
    assert d["error_type"] == "selector_not_found"
    assert d["selector"] == "#test-btn"
    assert d["error_message"] == "Element not found"
    assert d["page_url"] == "https://example.com/page"
    assert d["test_id"] == "test_123"
    assert d["failure_id"] == "fail_456"

@pytest.mark.asyncio
async def test_healing_result_to_dict():
    """Test HealingResult.to_dict() method."""
    from src.services.healing_service import HealingResult
    
    result = HealingResult(
        success=True,
        fix_applied={"fix_type": "update_selector", "new_value": "[data-testid='test']"},
        strategy_used="cognee_pattern",
        confidence=0.95,
        pattern_id="pattern_123",
    )
    
    d = result.to_dict()
    assert d["success"] is True
    assert d["fix_applied"]["fix_type"] == "update_selector"
    assert d["strategy_used"] == "cognee_pattern"
    assert d["confidence"] == 0.95
    assert d["pattern_id"] == "pattern_123"

@pytest.mark.asyncio
async def test_generate_fingerprint():
    """Test _generate_fingerprint method."""
    from src.services.healing_service import HealingService
    
    service = HealingService()
    
    # Same inputs should produce same fingerprint
    fp1 = service._generate_fingerprint("#selector", "error_type")
    fp2 = service._generate_fingerprint("#selector", "error_type")
    assert fp1 == fp2
    
    # Different inputs should produce different fingerprints
    fp3 = service._generate_fingerprint("#different", "error_type")
    assert fp1 != fp3
    
    # Fingerprint should be 32 characters (truncated sha256)
    assert len(fp1) == 32

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

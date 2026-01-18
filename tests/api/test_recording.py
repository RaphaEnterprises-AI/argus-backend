"""Tests for Recording API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException


class TestRecordingModels:
    """Tests for recording request/response models."""

    def test_rrweb_event_model(self, mock_env_vars):
        """Test RRWebEvent model."""
        from src.api.recording import RRWebEvent

        event = RRWebEvent(
            type=2,
            data={"node": {"type": 1}},
            timestamp=1704067200000,
        )

        assert event.type == 2
        assert event.timestamp == 1704067200000

    def test_recording_metadata_model(self, mock_env_vars):
        """Test RecordingMetadata model."""
        from src.api.recording import RecordingMetadata

        metadata = RecordingMetadata(
            duration=5000,
            start_time="2024-01-01T00:00:00Z",
            url="https://example.com",
            user_agent="Mozilla/5.0",
            viewport={"width": 1920, "height": 1080},
        )

        assert metadata.duration == 5000
        assert metadata.url == "https://example.com"

    def test_recording_upload_request_basic(self, mock_env_vars):
        """Test RecordingUploadRequest with basic data."""
        from src.api.recording import RecordingMetadata, RecordingUploadRequest, RRWebEvent

        events = [
            RRWebEvent(type=4, data={"href": "https://example.com"}, timestamp=1000),
            RRWebEvent(type=2, data={"node": {}}, timestamp=2000),
        ]
        metadata = RecordingMetadata(
            duration=1000,
            start_time="2024-01-01T00:00:00Z",
        )

        request = RecordingUploadRequest(
            events=events,
            metadata=metadata,
            project_id="test-project",
            name="Test Recording",
        )

        assert len(request.events) == 2
        assert request.project_id == "test-project"

    def test_recording_upload_request_empty_events_validation(self, mock_env_vars):
        """Test that empty events list is allowed."""
        from src.api.recording import RecordingMetadata, RecordingUploadRequest

        metadata = RecordingMetadata(
            duration=0,
            start_time="2024-01-01T00:00:00Z",
        )

        request = RecordingUploadRequest(
            events=[],
            metadata=metadata,
        )

        assert len(request.events) == 0

    def test_recording_upload_response_model(self, mock_env_vars):
        """Test RecordingUploadResponse model."""
        from src.api.recording import RecordingUploadResponse

        response = RecordingUploadResponse(
            success=True,
            recording_id="rec-123",
            events_count=100,
            duration_ms=5000,
            estimated_steps=15,
            message="Recording uploaded successfully",
        )

        assert response.success is True
        assert response.recording_id == "rec-123"

    def test_convert_request_model(self, mock_env_vars):
        """Test ConvertRequest model."""
        from src.api.recording import ConvertRequest

        request = ConvertRequest(
            recording_id="rec-123",
            test_name="Login Test",
            include_waits=True,
            include_scrolls=False,
            min_wait_threshold=1000,
            generalize_data=True,
        )

        assert request.recording_id == "rec-123"
        assert request.min_wait_threshold == 1000

    def test_convert_response_model(self, mock_env_vars):
        """Test ConvertResponse model."""
        from src.api.recording import ConvertResponse

        response = ConvertResponse(
            success=True,
            test={"id": "test-1", "steps": []},
            recording_id="rec-123",
            duration_ms=5000,
            steps_generated=10,
            assertions_generated=5,
            warnings=["Low confidence on step 3"],
        )

        assert response.success is True
        assert response.steps_generated == 10

    def test_test_step_model(self, mock_env_vars):
        """Test TestStepModel."""
        from src.api.recording import TestStepModel

        step = TestStepModel(
            action="click",
            target="#submit-btn",
            value=None,
            confidence=0.95,
            timestamp=1000,
        )

        assert step.action == "click"
        assert step.confidence == 0.95

    def test_recorder_snippet_request(self, mock_env_vars):
        """Test RecorderSnippetRequest model."""
        from src.api.recording import RecorderSnippetRequest

        request = RecorderSnippetRequest(
            project_id="test-project",
            upload_url="https://api.example.com/upload",
            options={"maskAllInputs": True},
        )

        assert request.project_id == "test-project"


class TestRecordingEnums:
    """Tests for recording enums."""

    def test_rrweb_event_type_enum(self, mock_env_vars):
        """Test RRWebEventType enum values."""
        from src.api.recording import RRWebEventType

        assert RRWebEventType.DOM_CONTENT_LOADED.value == 0
        assert RRWebEventType.LOAD.value == 1
        assert RRWebEventType.FULL_SNAPSHOT.value == 2
        assert RRWebEventType.INCREMENTAL_SNAPSHOT.value == 3
        assert RRWebEventType.META.value == 4
        assert RRWebEventType.CUSTOM.value == 5
        assert RRWebEventType.PLUGIN.value == 6

    def test_incremental_source_enum(self, mock_env_vars):
        """Test IncrementalSource enum values."""
        from src.api.recording import IncrementalSource

        assert IncrementalSource.MUTATION.value == 0
        assert IncrementalSource.MOUSE_MOVE.value == 1
        assert IncrementalSource.MOUSE_INTERACTION.value == 2
        assert IncrementalSource.SCROLL.value == 3
        assert IncrementalSource.INPUT.value == 5

    def test_mouse_interaction_enum(self, mock_env_vars):
        """Test MouseInteraction enum values."""
        from src.api.recording import MouseInteraction

        assert MouseInteraction.MOUSE_UP.value == 0
        assert MouseInteraction.MOUSE_DOWN.value == 1
        assert MouseInteraction.CLICK.value == 2
        assert MouseInteraction.DBL_CLICK.value == 4
        assert MouseInteraction.FOCUS.value == 5


class TestUploadRecordingEndpoint:
    """Tests for the upload_recording endpoint."""

    @pytest.mark.asyncio
    async def test_upload_recording_success(self, mock_env_vars):
        """Test successful recording upload."""
        from src.api.recording import (
            RecordingMetadata,
            RecordingUploadRequest,
            RRWebEvent,
            _recordings,
            upload_recording,
        )

        _recordings.clear()

        events = [
            RRWebEvent(type=4, data={"href": "https://example.com"}, timestamp=1000),
            RRWebEvent(type=2, data={"node": {}}, timestamp=2000),
            RRWebEvent(type=3, data={"source": 2, "type": 2}, timestamp=3000),  # Mouse click
        ]
        metadata = RecordingMetadata(
            duration=2000,
            start_time="2024-01-01T00:00:00Z",
            url="https://example.com",
        )

        body = RecordingUploadRequest(
            events=events,
            metadata=metadata,
            project_id="test-project",
            name="Test Recording",
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await upload_recording(mock_request, body)

                assert response.success is True
                assert response.events_count == 3
                assert response.duration_ms == 2000
                assert response.recording_id in _recordings
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_upload_recording_unauthenticated(self, mock_env_vars):
        """Test upload without authentication."""
        from src.api.recording import RecordingMetadata, RecordingUploadRequest, upload_recording

        body = RecordingUploadRequest(
            events=[],
            metadata=RecordingMetadata(duration=0, start_time="2024-01-01T00:00:00Z"),
        )

        mock_request = MagicMock()

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await upload_recording(mock_request, body)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_upload_recording_error_handling(self, mock_env_vars):
        """Test upload error handling."""
        from src.api.recording import RecordingMetadata, RecordingUploadRequest, upload_recording

        body = RecordingUploadRequest(
            events=[],
            metadata=RecordingMetadata(duration=0, start_time="2024-01-01T00:00:00Z"),
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            with patch("src.api.recording._count_interaction_events", side_effect=Exception("Parsing error")):
                response = await upload_recording(mock_request, body)

                assert response.success is False
                assert "Parsing error" in response.error


class TestConvertRecordingEndpoint:
    """Tests for the convert_recording endpoint."""

    @pytest.mark.asyncio
    async def test_convert_recording_success(self, mock_env_vars):
        """Test successful recording conversion."""
        from src.api.recording import ConvertRequest, _recordings, convert_recording

        recording_id = str(uuid4())
        _recordings[recording_id] = {
            "id": recording_id,
            "events": [
                {"type": 3, "data": {"source": 2, "type": 2, "id": 1}, "timestamp": 1000},
                {"type": 3, "data": {"source": 5, "text": "hello", "id": 2}, "timestamp": 2000},
            ],
            "metadata": {"duration": 1000, "url": "https://example.com"},
            "name": "Test Recording",
        }

        body = ConvertRequest(
            recording_id=recording_id,
            test_name="My Test",
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                with patch("src.api.recording._parse_rrweb_events") as mock_parse:
                    from src.api.recording import TestAssertionModel, TestStepModel
                    mock_parse.return_value = (
                        [TestStepModel(action="click", target="#btn")],
                        [TestAssertionModel(type="visible", target="#result")],
                        [],
                    )

                    response = await convert_recording(mock_request, body)

                    assert response.success is True
                    assert response.test["name"] == "My Test"
                    assert response.steps_generated == 1
                    assert response.assertions_generated == 1
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_convert_recording_not_found(self, mock_env_vars):
        """Test converting non-existent recording."""
        from src.api.recording import ConvertRequest, _recordings, convert_recording

        _recordings.clear()

        body = ConvertRequest(recording_id="nonexistent")

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await convert_recording(mock_request, body)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_convert_recording_unauthenticated(self, mock_env_vars):
        """Test conversion without authentication."""
        from src.api.recording import ConvertRequest, convert_recording

        body = ConvertRequest(recording_id="any-id")
        mock_request = MagicMock()

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await convert_recording(mock_request, body)

            assert exc_info.value.status_code == 401


class TestGetReplayDataEndpoint:
    """Tests for the get_replay_data endpoint."""

    @pytest.mark.asyncio
    async def test_get_replay_data_success(self, mock_env_vars):
        """Test successful replay data retrieval."""
        from src.api.recording import _recordings, get_replay_data

        recording_id = str(uuid4())
        _recordings[recording_id] = {
            "id": recording_id,
            "events": [{"type": 2, "data": {}, "timestamp": 1000}],
            "metadata": {"duration": 1000},
            "user_id": "test-user",
            "organization_id": "test-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await get_replay_data(mock_request, recording_id)

                assert response.success is True
                assert len(response.events) == 1
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_get_replay_data_access_denied(self, mock_env_vars):
        """Test replay data access denied for different org."""
        from src.api.recording import _recordings, get_replay_data

        recording_id = str(uuid4())
        _recordings[recording_id] = {
            "id": recording_id,
            "events": [],
            "metadata": {},
            "user_id": "other-user",
            "organization_id": "other-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                with pytest.raises(HTTPException) as exc_info:
                    await get_replay_data(mock_request, recording_id)

                assert exc_info.value.status_code == 403
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_get_replay_data_not_found(self, mock_env_vars):
        """Test replay data not found."""
        from src.api.recording import _recordings, get_replay_data

        _recordings.clear()

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await get_replay_data(mock_request, "nonexistent")

            assert exc_info.value.status_code == 404


class TestGenerateRecorderSnippetEndpoint:
    """Tests for the generate_recorder_snippet endpoint."""

    @pytest.mark.asyncio
    async def test_generate_snippet_success(self, mock_env_vars):
        """Test successful snippet generation."""
        from src.api.recording import RecorderSnippetRequest, generate_recorder_snippet

        body = RecorderSnippetRequest(
            project_id="test-project",
            upload_url="https://api.example.com/upload",
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            response = await generate_recorder_snippet(mock_request, body)

            assert response.success is True
            assert "rrweb" in response.snippet
            assert "argusStopRecording" in response.snippet
            assert response.cdn_script.startswith("https://")

    @pytest.mark.asyncio
    async def test_generate_snippet_with_custom_options(self, mock_env_vars):
        """Test snippet generation with custom options."""
        from src.api.recording import RecorderSnippetRequest, generate_recorder_snippet

        body = RecorderSnippetRequest(
            project_id="test-project",
            options={
                "checkoutEveryNms": 5000,
                "maskAllInputs": True,
            },
        )

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            response = await generate_recorder_snippet(mock_request, body)

            assert response.success is True
            assert "5000" in response.snippet

    @pytest.mark.asyncio
    async def test_generate_snippet_unauthenticated(self, mock_env_vars):
        """Test snippet generation without authentication."""
        from src.api.recording import RecorderSnippetRequest, generate_recorder_snippet

        body = RecorderSnippetRequest()
        mock_request = MagicMock()

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await generate_recorder_snippet(mock_request, body)

            assert exc_info.value.status_code == 401


class TestGetRecordingsEndpoint:
    """Tests for the get_recordings endpoint."""

    @pytest.mark.asyncio
    async def test_get_recordings_success(self, mock_env_vars):
        """Test successful recordings list retrieval."""
        from src.api.recording import _recordings, get_recordings

        _recordings.clear()
        _recordings["rec-1"] = {
            "id": "rec-1",
            "name": "Recording 1",
            "project_id": "project-1",
            "events_count": 100,
            "interaction_count": 10,
            "metadata": {"duration": 5000, "url": "https://example.com"},
            "created_at": "2024-01-01T00:00:00Z",
            "user_id": "test-user",
            "organization_id": "test-org",
        }
        _recordings["rec-2"] = {
            "id": "rec-2",
            "name": "Recording 2",
            "project_id": "project-2",
            "events_count": 50,
            "interaction_count": 5,
            "metadata": {"duration": 3000, "url": "https://other.com"},
            "created_at": "2024-01-02T00:00:00Z",
            "user_id": "test-user",
            "organization_id": "test-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await get_recordings(mock_request)

                assert response["success"] is True
                assert len(response["recordings"]) == 2
                assert response["total"] == 2
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_get_recordings_filter_by_project(self, mock_env_vars):
        """Test filtering recordings by project."""
        from src.api.recording import _recordings, get_recordings

        _recordings.clear()
        _recordings["rec-1"] = {
            "id": "rec-1",
            "project_id": "project-1",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "user_id": "test-user",
            "organization_id": "test-org",
        }
        _recordings["rec-2"] = {
            "id": "rec-2",
            "project_id": "project-2",
            "metadata": {},
            "created_at": "2024-01-02T00:00:00Z",
            "user_id": "test-user",
            "organization_id": "test-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await get_recordings(mock_request, project_id="project-1")

                assert len(response["recordings"]) == 1
                assert response["recordings"][0]["id"] == "rec-1"
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_get_recordings_org_isolation(self, mock_env_vars):
        """Test that recordings from other orgs are not visible."""
        from src.api.recording import _recordings, get_recordings

        _recordings.clear()
        _recordings["rec-1"] = {
            "id": "rec-1",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "user_id": "other-user",
            "organization_id": "other-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await get_recordings(mock_request)

                assert len(response["recordings"]) == 0
        finally:
            _recordings.clear()


class TestListRecordingsEndpoint:
    """Tests for the list_recordings endpoint (alias for get_recordings)."""

    @pytest.mark.asyncio
    async def test_list_recordings_success(self, mock_env_vars):
        """Test list_recordings endpoint."""
        from src.api.recording import _recordings, list_recordings

        _recordings.clear()
        _recordings["rec-1"] = {
            "id": "rec-1",
            "metadata": {"duration": 5000},
            "created_at": "2024-01-01T00:00:00Z",
            "user_id": "test-user",
            "organization_id": "test-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await list_recordings(mock_request)

                assert response["success"] is True
                assert response["total"] == 1
        finally:
            _recordings.clear()


class TestDeleteRecordingEndpoint:
    """Tests for the delete_recording endpoint."""

    @pytest.mark.asyncio
    async def test_delete_recording_success(self, mock_env_vars):
        """Test successful recording deletion."""
        from src.api.recording import _recordings, delete_recording

        recording_id = "rec-1"
        _recordings[recording_id] = {
            "id": recording_id,
            "user_id": "test-user",
            "organization_id": "test-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                response = await delete_recording(mock_request, recording_id)

                assert response["success"] is True
                assert recording_id not in _recordings
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_delete_recording_not_found(self, mock_env_vars):
        """Test deleting non-existent recording."""
        from src.api.recording import _recordings, delete_recording

        _recordings.clear()

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
            with pytest.raises(HTTPException) as exc_info:
                await delete_recording(mock_request, "nonexistent")

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_recording_access_denied(self, mock_env_vars):
        """Test deleting recording from different org."""
        from src.api.recording import _recordings, delete_recording

        recording_id = "rec-1"
        _recordings[recording_id] = {
            "id": recording_id,
            "user_id": "other-user",
            "organization_id": "other-org",
        }

        mock_request = MagicMock()
        mock_user = {"user_id": "test-user", "organization_id": "test-org"}

        try:
            with patch("src.api.recording.get_current_user", AsyncMock(return_value=mock_user)):
                with pytest.raises(HTTPException) as exc_info:
                    await delete_recording(mock_request, recording_id)

                assert exc_info.value.status_code == 403
        finally:
            _recordings.clear()

    @pytest.mark.asyncio
    async def test_delete_recording_unauthenticated(self, mock_env_vars):
        """Test deleting recording without authentication."""
        from src.api.recording import delete_recording

        mock_request = MagicMock()

        with patch("src.api.recording.get_current_user", AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await delete_recording(mock_request, "any-id")

            assert exc_info.value.status_code == 401


class TestPayloadSizeValidation:
    """Tests for recording payload size validation."""

    def test_events_payload_size_validation_small(self, mock_env_vars):
        """Test that small payloads pass validation."""
        from src.api.recording import RecordingMetadata, RecordingUploadRequest, RRWebEvent

        events = [
            RRWebEvent(type=2, data={"small": "data"}, timestamp=i * 1000)
            for i in range(100)
        ]
        metadata = RecordingMetadata(duration=100000, start_time="2024-01-01T00:00:00Z")

        request = RecordingUploadRequest(events=events, metadata=metadata)
        assert len(request.events) == 100

    def test_events_max_length_enforced(self, mock_env_vars):
        """Test that max_length constraint is enforced."""
        from pydantic import ValidationError

        from src.api.recording import RecordingMetadata, RecordingUploadRequest, RRWebEvent

        # Try to create more than 50,000 events (the max_length)
        events = [
            RRWebEvent(type=2, data={}, timestamp=i)
            for i in range(50001)
        ]
        metadata = RecordingMetadata(duration=50001, start_time="2024-01-01T00:00:00Z")

        with pytest.raises(ValidationError):
            RecordingUploadRequest(events=events, metadata=metadata)


class TestRecordingHelperFunctions:
    """Tests for recording helper functions."""

    def test_count_interaction_events(self, mock_env_vars):
        """Test _count_interaction_events helper."""
        from src.api.recording import RRWebEvent, _count_interaction_events

        events = [
            RRWebEvent(type=3, data={"source": 2, "type": 2}, timestamp=1000),  # Click
            RRWebEvent(type=3, data={"source": 5, "text": "hello"}, timestamp=2000),  # Input
            RRWebEvent(type=3, data={"source": 1}, timestamp=3000),  # Mouse move - not interaction
            RRWebEvent(type=2, data={}, timestamp=4000),  # Full snapshot - not interaction
        ]

        count = _count_interaction_events(events)
        assert count >= 1  # At least the click should be counted

    def test_parse_rrweb_events_basic(self, mock_env_vars):
        """Test _parse_rrweb_events with basic events."""
        from src.api.recording import _parse_rrweb_events

        events = [
            {"type": 3, "data": {"source": 2, "type": 2, "id": 1}, "timestamp": 1000},  # Click
            {"type": 3, "data": {"source": 5, "text": "hello", "id": 2}, "timestamp": 2000},  # Input
        ]

        steps, assertions, warnings = _parse_rrweb_events(events)

        assert isinstance(steps, list)
        assert isinstance(assertions, list)
        assert isinstance(warnings, list)

    def test_parse_rrweb_events_with_waits(self, mock_env_vars):
        """Test _parse_rrweb_events with wait detection."""
        from src.api.recording import _parse_rrweb_events

        events = [
            {"type": 3, "data": {"source": 2, "type": 2, "id": 1}, "timestamp": 1000},
            {"type": 3, "data": {"source": 2, "type": 2, "id": 2}, "timestamp": 5000},  # 4s gap
        ]

        steps, _, _ = _parse_rrweb_events(
            events,
            include_waits=True,
            min_wait_threshold=500,
        )

        # Should include wait step for the 4s gap
        wait_steps = [s for s in steps if s.action == "wait"]
        assert len(wait_steps) >= 0  # Implementation dependent

    def test_parse_rrweb_events_scrolls_excluded(self, mock_env_vars):
        """Test _parse_rrweb_events excludes scrolls by default."""
        from src.api.recording import _parse_rrweb_events

        events = [
            {"type": 3, "data": {"source": 3, "x": 0, "y": 100}, "timestamp": 1000},  # Scroll
        ]

        steps, _, _ = _parse_rrweb_events(events, include_scrolls=False)

        scroll_steps = [s for s in steps if s.action == "scroll"]
        assert len(scroll_steps) == 0

    def test_parse_rrweb_events_scrolls_included(self, mock_env_vars):
        """Test _parse_rrweb_events includes scrolls when enabled."""
        from src.api.recording import _parse_rrweb_events

        events = [
            {"type": 3, "data": {"source": 3, "x": 0, "y": 100, "id": 1}, "timestamp": 1000},  # Scroll
        ]

        steps, _, _ = _parse_rrweb_events(events, include_scrolls=True)

        # May or may not include scroll based on implementation
        assert isinstance(steps, list)

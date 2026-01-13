# Changelog

## [2.4.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.3.0...v2.4.0) (2026-01-13)

### 🚀 Major Chat Experience Improvements

This release delivers significant improvements to the chat interface, making test execution more reliable and visual feedback more comprehensive.

### Features

* **Artifacts API** - New `/api/v1/artifacts/{id}` endpoint to fetch screenshot content by artifact ID ([972cdce](https://github.com/RaphaEnterprises-AI/argus-backend/commit/972cdce))
  - `GET /api/v1/artifacts/{id}` - Returns base64 encoded screenshot
  - `GET /api/v1/artifacts/{id}/raw` - Returns raw image bytes for direct embedding
  - `POST /api/v1/artifacts/resolve` - Bulk resolve multiple artifact references

* **Cancel/Stop Functionality** - Backend cancellation support for long-running operations
  - `DELETE /api/v1/chat/cancel/{thread_id}` - Cancel ongoing chat execution
  - LangGraph state-based cancellation via `should_continue` flag
  - Frontend stop button now properly terminates backend operations

* **Session Screenshots Panel** - New collapsible panel showing all screenshots from a chat session
  - Accumulates screenshots across all tool invocations
  - Deduplicates identical screenshots
  - Expandable gallery view for easy review

### Bug Fixes

* **Screenshots now display correctly** - Fixed issue where screenshots returned artifact IDs instead of image data. Frontend now fetches actual content from artifacts API.

* **Stop button actually stops** - Fixed `should_continue()` in LangGraph to check cancellation flag before proceeding with tool calls.

* **Multiple screenshots visible** - Fixed issue where only one screenshot was shown when multiple were captured. Added `extractScreenshotsFromResult()` helper to collect all screenshots.

* **Chat history includes tool results** - Added `ToolMessage` serialization to `/api/v1/chat/history/{thread_id}` endpoint. Tool execution results with `_artifact_refs` are now included in history.

### Technical Details

* Added `should_continue: bool` field to `ChatState` TypedDict
* Created message serialization helpers: `_serialize_message()` and `_serialize_message_content()`
* Frontend proxy route at `/api/streaming/cancel/[threadId]` for CORS-safe cancellation
* Cloudflare R2 integration for screenshot artifact storage and retrieval

---

## [2.3.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.2.0...v2.3.0) (2026-01-13)


### Features

* add test defaults and complete user settings API ([cf5a23e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cf5a23e53b7461c78e310974902befb49734b91d))


### Bug Fixes

* correct release-please config to use packages format ([cbe6a46](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cbe6a463aef5146fd9a8fb62b6b97cb6f012d382))


### Documentation

* add API reference and settings architecture documentation ([cdb059f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cdb059fcf0e93447c7ad1f74de17918fd3248598))

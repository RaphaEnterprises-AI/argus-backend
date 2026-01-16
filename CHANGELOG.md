# Changelog

## [2.7.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.6.2...v2.7.0) (2026-01-16)


### Features

* add secret migration script and database migrations ([dd4533f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/dd4533fb5f4e8122904f3eaa96a9c1484dd681c6))
* **api:** register MCP sessions router in FastAPI app ([89557c0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/89557c0bea37ce23600f7597f3baa42de25ef91e))
* **browser-pool:** add Kubernetes browser pool with production-grade JWT auth ([8e5169c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8e5169c774095dc4c8182fe2cca9b45429842189))
* **browser:** add Python client for browser pool with JWT auth ([39c24df](https://github.com/RaphaEnterprises-AI/argus-backend/commit/39c24df189e95e39dfe51fa556e8fc5ccae67538))
* **mcp:** add MCP session management API and dashboard visibility ([8fda84e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8fda84e6334b91e891c8d6ac77659521a4c437dc))


### Bug Fixes

* **api:** additional mcp_sessions fixes and add rpc method ([77bf06d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/77bf06daf4c9f11e9ce883960ba0b3c35b9d4fd8))
* **api:** fix remaining bugs in mcp_sessions.py ([a83d4d7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a83d4d7e71130b6bd72c880fbb2e09f8d1789812))
* **api:** use dict access for user object in mcp_sessions ([bffe347](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bffe347fd047280e64cf92e6164468570a0208c0))
* **auth:** handle SecretStr in JWT token generation ([58074f6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/58074f699c3cccc0fbf003d4a8a691425b7b7115))
* **security:** comprehensive SOC2 security audit fixes ([99b3484](https://github.com/RaphaEnterprises-AI/argus-backend/commit/99b34842439487df506fbfc6a8e2cdb5a7da8c71))
* update deployment workflow for new docs structure ([742aaf3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/742aaf31871855160baba196535b429fdf744a57))

## [2.6.2](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.6.1...v2.6.2) (2026-01-14)


### Bug Fixes

* **tests:** use dynamic API_VERSION instead of hardcoded version ([a228b40](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a228b408f994f99ee2b4be846004052eb5c4aef2))

## [2.6.1](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.6.0...v2.6.1) (2026-01-14)


### Bug Fixes

* **ci:** enable auto-merge for release PRs and improve Claude Code Review ([60e51fa](https://github.com/RaphaEnterprises-AI/argus-backend/commit/60e51fa072f8f2fa99fa5768376c73c41990ec80))
* **ci:** enable auto-trigger for Claude Code Review on release PRs ([4e5dd89](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4e5dd8919df9698ed2f39b61445d62f15cf23f46))
* **ci:** simplify Claude Code Review to match argus dashboard pattern ([b4433d6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b4433d6451954d894746e181fe8b30ed0323564a))

## [2.6.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.5.0...v2.6.0) (2026-01-14)


### Features

* **orchestrator:** upgrade to LangGraph 1.0 patterns and align frontend-backend ([63ada81](https://github.com/RaphaEnterprises-AI/argus-backend/commit/63ada812cd9361289824a5bd9e3d71226608967b))


### Bug Fixes

* add missing Query import in discovery.py ([c2c4019](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c2c40199e07a272a39b806bf4d0dcad46f0cc16c))
* resolve E2E API bugs and infrastructure issues ([4e2fcba](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4e2fcba39a6a404d0c335b5bb30acdb9cb0dacae))
* resolve supabase migration schema drift issues ([5799493](https://github.com/RaphaEnterprises-AI/argus-backend/commit/57994931c712be73f061430ec9d239999ff87b2a))
* **visual-ai:** resolve screenshot capture and database compatibility issues ([7a9bc23](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7a9bc233e73c5f286a2e2812d0a1933269499600))

## [2.5.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.4.0...v2.5.0) (2026-01-14)


### Features

* **chat:** add structured test preview response for createTest tool ([4f88d2e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4f88d2ea5a0e235c13eccc15bc03def7cbc385a2))


### Bug Fixes

* add comprehensive debug logging to JWT verification ([ee64c67](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ee64c674baa0921ca10d589c03e29e8d4d1c475c))
* make organization slug optional and auto-generate from name ([0030492](https://github.com/RaphaEnterprises-AI/argus-backend/commit/003049235b387d81e22f6e17f511ee4fdbdb8563))
* sync Clerk organizations to Supabase on first access ([07cb876](https://github.com/RaphaEnterprises-AI/argus-backend/commit/07cb8768d2e2fd29999d5af7842119d5d77286d7))
* translate Clerk org IDs to Supabase UUIDs in API keys endpoints ([2ad9aaf](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2ad9aaff3a9382556ab13a2e962128e485b619be))

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

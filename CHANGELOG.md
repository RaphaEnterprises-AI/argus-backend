# Changelog

## [2.7.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.6.2...v2.7.0) (2026-01-16)


### Features

* add /api/v1/auth/me endpoint for user info ([cf2019a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cf2019a3f3a55cc20d0989fefb7e475aa0189b8e))
* add auto-indexing for semantic search ([f2a5800](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f2a5800760945d7dbf8fb6f3f686a72d346c4561))
* add Clerk JWT verification for dashboard SSO ([d105ea8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d105ea8e6568dfc96cc0339c26c3afd6267f1180))
* add CLI script to create API keys ([637575d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/637575d7e83c5149084c0b265dfe755b722cfc6f))
* add Cloudflare R2 storage for screenshots and artifacts ([7709524](https://github.com/RaphaEnterprises-AI/argus-backend/commit/77095249bab93e6f419de4f924f0ab2b77b758e6))
* add code-aware healing, analyzers, and comprehensive test suite ([e078620](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e0786201d466ecb91e802a72dfce6184317e6225))
* add Crawlee microservice, Visual AI, and Discovery Intelligence ([3b4ed10](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3b4ed10e4677774b76bb239c4dd83a3019e60af5))
* add debug-token endpoint and improve Clerk JWT claim extraction ([99fa47e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/99fa47e0b2e4795f5d4c198c3077bc6dbf41e0ad))
* add enterprise API endpoints for team, API keys, audit, and healing ([402f140](https://github.com/RaphaEnterprises-AI/argus-backend/commit/402f140958c6ab0742ec5fa909dc22fe6de9196e))
* add frontend API aliases and missing AI intelligence endpoints ([4674e84](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4674e848a3afbcf04246b52123b963549dc659d8))
* add GET /sessions and GET /patterns endpoints to discovery API ([db24ce9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/db24ce9d681c578e33916c6c1e9573633955c7c0))
* add Hyperdrive for Supabase connection pooling ([8c83c64](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8c83c64a3326144e2613e900fe72c7c6b71ba3dc))
* add MCP quality tools, CI coverage tables, and enhanced landing page ([39ae8f8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/39ae8f84815ceac751d1f4ad8569575fb363a56f))
* add multi-tenancy, AI cost tracking, and Gemini 2.5/3.0 support ([72fa413](https://github.com/RaphaEnterprises-AI/argus-backend/commit/72fa413639a4110635fa8d8be598ae69ddccc3dc))
* add OAuth2 Device Flow authentication for MCP servers ([a4e7bda](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a4e7bdabf70ba259a074b72a574e7ae30d2c6694))
* add Railway deployment and Brain-Dashboard integration ([87cee11](https://github.com/RaphaEnterprises-AI/argus-backend/commit/87cee1195c8d6c98830220150edd7709b2d60085))
* add real-time activity tracking and mobile responsive sidebar ([9d9f645](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9d9f645d02d236fe1abeeb2e5a15352a0582541d))
* Add scheduling, notifications, parameterized tests & database migrations ([d826c18](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d826c186c01836577133f18b88c6c4baad74cf1d))
* add screenshot capture and fix timeouts for E2E testing ([73884fb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/73884fbf0556723ad64b71d845f631b5c7624330))
* add secret migration script and database migrations ([dac7894](https://github.com/RaphaEnterprises-AI/argus-backend/commit/dac7894fab5a7f20e4a9eb3f042270a293237837))
* add status page, error pages, and migration fixes ([fd9f6e6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fd9f6e62d9dd979f5662e5814ab3631e0b0fd0bf))
* add Supabase persistence for notifications and parameterized tests APIs ([c9bafa5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c9bafa53b4eb6d6d35e9048b447ef402d093d25e))
* add Supabase persistence to scheduling API ([caa7de7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/caa7de7bb5e35c268617a532e86f9957a094c61a))
* add test defaults and complete user settings API ([acdc5cd](https://github.com/RaphaEnterprises-AI/argus-backend/commit/acdc5cd0d8f58986a744f7179e748e3de8c4e754))
* add Upstash Redis caching layer for performance optimization ([970234a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/970234a5796b7443879b9ccba88358b5b0876982))
* add verify action to worker, model registry, and integration tests ([2b17b4a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2b17b4a745c2c3ed33c26376e4e22c673d944690))
* AI-powered test healing with retry functionality ([e2c47ec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e2c47ec63dc899e9a7554831c9ec107c0a070490))
* **api:** register MCP sessions router in FastAPI app ([18eee40](https://github.com/RaphaEnterprises-AI/argus-backend/commit/18eee40ebcf764ca0eb81eeadbe2f57967a60dfa))
* **browser-pool:** add Kubernetes browser pool with production-grade JWT auth ([1137be6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1137be6ab129fd60bfce5bf7334d165a6fd9fe2f))
* **browser:** add Python client for browser pool with JWT auth ([60d109c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/60d109c9c879e9a41df458b5ef248ed1d4a2b8b3))
* **chat:** add structured test preview response for createTest tool ([e1ff7dc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e1ff7dc4bd947361cfc55a2fa44db95278277989))
* **dashboard:** add frontend with performance and UUID validation fixes ([320caf5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/320caf518c23d47964f83641d6daa371f312d7f9))
* **dashboard:** add Vercel Speed Insights for performance monitoring ([f32d930](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f32d930aa8c0d0b54ce28703866ffa707ce691a5))
* deploy Cloudflare Worker with queue handler and KV cache ([fd48087](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fd48087b63ef4004478039df44df023e67ff9e08))
* enterprise security layer for SOC2 compliance ([e00d970](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e00d9701b715607542d605b43e5e22fc5a813d43))
* fetch user email from Clerk Backend API when not in JWT ([41e460b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/41e460b44c627f10c8e717db5c59f9bcfc92ac4b))
* implement complete user management system ([4437660](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4437660325672766410cec26baebe800cf4d5fd5))
* implement full LangGraph 1.0 feature suite for production-ready orchestration ([1c2c003](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1c2c003f6122491c827b220cf6ceb2709466b104))
* **mcp:** add Argus MCP Server for AI IDE integration ([7db9319](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7db93197a56124eadba3e6aa9a37b5779b14ad8a))
* **mcp:** add MCP session management API and dashboard visibility ([fe7904b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fe7904b454485249ef12fa67be2240b2dbeb275a))
* **orchestrator:** upgrade to LangGraph 1.0 patterns and align frontend-backend ([7a2aa90](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7a2aa90c9086b5943aae519195967f8206b3c89e))
* **self-healing:** implement enhanced 95%+ pass rate system ([a3ce72d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a3ce72da7368b903314f9ed6afdc3cdd4ed39f75))
* Test execution hardening with retry logic and error categorization ([96d7ddd](https://github.com/RaphaEnterprises-AI/argus-backend/commit/96d7dddd0585effa5b5c489ea46c90ce6d68c649))
* upgrade to bge-large-en-v1.5 embeddings (1024 dims) ([cc22004](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cc22004df9e870a804648da8bdccb6cd3943702d))
* **worker:** upgrade to latest AI models and dependencies v2.1.0 ([9a25704](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9a2570470211626daf0e38b2c73bea2b6e1fafc9))


### Bug Fixes

* actually send invitation emails via Resend ([8ddee36](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8ddee360429c979bdbc1449b28c38941168b4089))
* add aiosmtplib to Dockerfile fallback dependencies ([b56abeb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b56abebe1963fde8dcd87ace24715b7c32d24e45))
* add API key support to all organization access checks ([c43d101](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c43d101a5843c6c449c400eb3e07a67e92b710c3))
* add comprehensive debug logging to JWT verification ([3bf6a6c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3bf6a6c69e28ebf4d48a84c8529130ace94cddea))
* add email fallback for org access verification ([64bda62](https://github.com/RaphaEnterprises-AI/argus-backend/commit/64bda62af229f29b7fc272e01738d8d447208d4c))
* add email-validator dependency for EmailStr support ([55b1ce4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/55b1ce4c6d31cd362aeff541f835d48bfe00c788))
* Add missing backend dependencies for Docker build ([9d30698](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9d30698ab2afaffb82a6c627e36aee2cf81589a8))
* add missing columns to audit_logs table and fix duration_ms type ([1a8e3d7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1a8e3d779d674c0b46a2dabe2f0f5b4f1d88af1f))
* add missing Query import in discovery.py ([088aa94](https://github.com/RaphaEnterprises-AI/argus-backend/commit/088aa948f1c47db34dceed76e9cfcc2db4ff921d))
* add missing supabase integration module for audit logs ([6c7168d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6c7168df90ab441cadf58e945f7e2cc762ce74df))
* add PyJWT[crypto] dependency for Clerk RS256 verification ([c4b920c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c4b920ce6029e1a84c74b8a8059b1d8380cf0e5c))
* add PyJWT[crypto] to Dockerfile fallback dependencies ([b93576c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b93576c531b100c9966f0ae162ef6a960a4c4f91))
* add robust JSON parsing to orchestrator nodes ([fde1a1b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fde1a1ba98e02be3ae200d400f78df6faf5f0f47))
* **api:** additional mcp_sessions fixes and add rpc method ([3e4bf24](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3e4bf24a5b15676dc0d76b3da5767fe9b9f7ddfe))
* **api:** fix remaining bugs in mcp_sessions.py ([b314cc2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b314cc2b67dddb8d409ca01871c3733848888bd4))
* **api:** use dict access for user object in mcp_sessions ([9ee520d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9ee520d5fc402622c6f3404cbabd0e971eada233))
* **auth:** handle SecretStr in JWT token generation ([4d329c0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4d329c0a8187966c22b5c5b54e98b2b399a54096))
* auto-detect Clerk JWKS URL from token issuer claim ([b85ceff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b85ceff7997c398a124047d6e5a8b8857b1df700))
* backend chat improvements - artifacts API, cancel, tool serialization ([33ee271](https://github.com/RaphaEnterprises-AI/argus-backend/commit/33ee271ae39eede6086861b76f71fd372afcd66f))
* **ci:** enable auto-merge for release PRs and improve Claude Code Review ([35a2f14](https://github.com/RaphaEnterprises-AI/argus-backend/commit/35a2f14f1e27018bbbb315e31d1e0e9940072dd3))
* **ci:** enable auto-trigger for Claude Code Review on release PRs ([bf9cd6f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bf9cd6fe5b122cad99a0bcad6081693ee9ad812d))
* **ci:** simplify Claude Code Review to match argus dashboard pattern ([4bbfd6d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4bbfd6de9197832e50c81e806fc6f38ef95c5e3c))
* context overflow and duplicate message display ([21ebfac](https://github.com/RaphaEnterprises-AI/argus-backend/commit/21ebfac1c1a6f297bf3e91b567078cd070b5baef))
* convert Rollbar timestamps to ISO format strings ([df1168d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df1168d605c4300e14756fed4309ddeaaa233d6a))
* correct AI SDK error part format (must be string, not object) ([10bbf91](https://github.com/RaphaEnterprises-AI/argus-backend/commit/10bbf910fb4a1121a41d2e268b81f27ee0560062))
* correct API key prefix mismatch in auth.py ([a39eec9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a39eec9df65bbff6ceb36fd7e5a849bf1b8e25ec))
* correct AuthMethod string comparison for API key auth ([bc6d934](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bc6d9342daf3da33f402c178f77daef54fb9f45b))
* correct email service method name (send_invitation) ([599de99](https://github.com/RaphaEnterprises-AI/argus-backend/commit/599de99887132972e28085dc91c75d5e6faa3e42))
* correct flow inference for Crawlee discovery sessions ([e9d7d3c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e9d7d3c36c99b3232e16810470498688e871ab90))
* correct release-please config to use packages format ([0115fa4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0115fa485a116c282c7da3efa54215e208e2d10f))
* **dashboard:** additional optimizations for Vercel function size ([c64a4f4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c64a4f4801f9d9645471f7aeca199db806bea4e6))
* **dashboard:** move outputFileTracingExcludes to root config level ([21c20e8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/21c20e8e63be8125a6bf8227bfcd0d4785e0b2f1))
* **dashboard:** resolve Vercel serverless function size limit (250MB) ([4e43e0c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4e43e0c0368045861c9f0d533c8ae5314e21da77))
* handle flat payload structures in FullStory webhook ([bdba821](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bdba8211ca99987c8b13889ea67fc76284fc826d))
* handle missing Supabase tables gracefully in quality API ([fcead03](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fcead036e857939c800e34729016b087a8a24cf0))
* handle None values in discovery response formatting ([bfaaca3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bfaaca37d95c54973f6f85680dd044bfa195f413))
* handle NULL created_by in API key authentication ([4dffe3e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4dffe3e8f73d256b363a3f590e90daf73a6f6a9a))
* improve AI SDK streaming reliability and JSON formatting ([657a78b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/657a78bfb8d4918b67dc9f51f807884c9e41e1d7))
* improve Railway deployment reliability ([0a321d1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0a321d1f12281ed2de2333530844f7132c020c1b))
* make anthropic_api_key optional for healthcheck ([c7311b5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c7311b5981acf16d0f53c4ad2737697ea55b582e))
* make Clerk JWT verification more robust with graceful fallbacks ([e0cb3bb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e0cb3bbcdfb129c1b2526bec6973777c516a2d0d))
* make email nullable in user_profiles and fix onboarding_step type ([be498a6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/be498a699588a855a92ae6562fbfc3db8c87844e))
* make organization slug optional and auto-generate from name ([148bd4f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/148bd4fc510f3c38d49903623e8638f06567305f))
* null safety in vectorize and seed default project ([bd293a7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bd293a78bb39242845c3cc131a3ca1e913109936))
* pass text as array to Workers AI embedding API ([77e5d7a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/77e5d7ac0b3ceba4f6a515b0cce7dc9a8598fa9f))
* remove unused CognitiveEngine import in predictive_quality ([5c893b3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5c893b3edad078ad72afebdc5da5fa14c346daf3))
* rename security audit migration to fix duplicate timestamp ([598b069](https://github.com/RaphaEnterprises-AI/argus-backend/commit/598b06916cbe71cdaf37148f17f6dc03008ef868))
* resolve 'default' org_id to user's actual organization ([8fb39b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8fb39b452e9ca623a509ee72d5f9eb64a9801314))
* resolve E2E API bugs and infrastructure issues ([2dc871f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2dc871f2c3ec845e9bb9135b2e849be26855d1f2))
* resolve production bugs from Railway testing ([bdfe8f5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bdfe8f5e4ab61abe0e061a2e228f9f44a41ba1dc))
* resolve supabase migration schema drift issues ([4e6d6f3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4e6d6f384598eeddaf1acee640b34bf4dcb74b2c))
* **security:** comprehensive SOC2 security audit fixes ([43a8311](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43a8311def50edbb3d3506e2910f6ab166458ace))
* simplify Dockerfile for Railway deployment ([5a58c3b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5a58c3be961ec3e6ed75b26ed65da391cdbec40d))
* simplify Vectorize v2 query payload ([64e4d8a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/64e4d8a383e3869c3b249ecbe85cdf92547c8570))
* skip org membership verification for API key auth ([a138fb6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a138fb67a0f4ce5ac82a0b6a90ea1a0455060c57))
* stream tool results in AI SDK format (a: prefix) ([ed3bd65](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ed3bd657b27cfd1a88f1a229029563a6452b79c6))
* support API key authentication for organization access ([468c4ae](https://github.com/RaphaEnterprises-AI/argus-backend/commit/468c4aed04fb139da91823191ab6c795bb8427f1))
* support comma-separated CORS_ALLOWED_ORIGINS environment variable ([8cb55be](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8cb55bed5667c3d66199c8d6af579848e1ac38e8))
* sync Clerk organizations to Supabase on first access ([8074b69](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8074b69c25fcabb4ebaae570749afc4e11e4fd29))
* **tests:** use dynamic API_VERSION instead of hardcoded version ([4afeda8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4afeda83ca079a29cfe447cc8754b1b7a154575e))
* translate Clerk org IDs to Supabase UUIDs in API keys endpoints ([00bef21](https://github.com/RaphaEnterprises-AI/argus-backend/commit/00bef213abd164a4e07a3597c2f957d6a7c59c56))
* update authenticate_api_key to use correct Supabase client ([3f7e38f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3f7e38f26a23eca007ce1c03062ac7a0ce779860))
* update chat stream to use Vercel AI SDK compatible format ([708fe35](https://github.com/RaphaEnterprises-AI/argus-backend/commit/708fe35e82e8717bddf785de94a90081ca368ada))
* update deployment workflow for new docs structure ([9aa661f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9aa661f656dbf2ec85042b9108ae89c2cf343973))
* update Dockerfile with v2.0.0 LangGraph dependencies ([33c8a2b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/33c8a2b240aad35a3a69772ca4efec82457ec8ad))
* update get_current_user to use request.state.user from middleware ([8e2e895](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8e2e8958d8b39812d3c0ced319105e9046a81460))
* URL encode datetime in Supabase query URLs ([31595b0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/31595b07e916a234d8354eeeb7997a9c13ebc3b3))
* use claude-3-haiku for test generation ([288038d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/288038d83cf401a0a31273c6db533abf191c49db))
* use correct Claude Haiku 4.5 model ID ([fc58de2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fc58de297302e887a07c7565b0f99d699adc3343))
* use correct Claude model name ([043d0f4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/043d0f44ce9f1b2740eec5fae5f21049a5d65cef))
* use correct column name overall_risk_score ([6b2df81](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6b2df81f5cf0797b28d455eff980185f5bd217ac))
* use security_audit_logs table for API request logging ([e1e9f6b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e1e9f6b9ede47d8f2f1903957b1f726b89bb7c54))
* use string values for Vectorize v2 API parameters ([bbf6664](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bbf66645a553d0cf0cf018d728b5bf3912f2f744))
* use Vectorize v2 API endpoint ([d4b692d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4b692dec10a80ec409103a782c0627616a9ba28))
* **visual-ai:** resolve screenshot capture and database compatibility issues ([9cb6b30](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9cb6b3072f28abf99744688ed067c9340053018f))


### Documentation

* add API reference and settings architecture documentation ([94ff420](https://github.com/RaphaEnterprises-AI/argus-backend/commit/94ff4209294aac9d41f83935696039896cb09130))
* add comprehensive architecture documentation ([522cb1f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/522cb1fed93cb6806c1a746eeac3e36fcc72a68c))
* add comprehensive autonomous testing roadmap ([38d3497](https://github.com/RaphaEnterprises-AI/argus-backend/commit/38d3497a59f3c36321e0791492b2e738ac616f59))
* add comprehensive UX enhancement plan ([d89bb6e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d89bb6e152d193705eaa2ad7121150b5b969adea))
* add OpenAPI specification export ([70ea055](https://github.com/RaphaEnterprises-AI/argus-backend/commit/70ea055a05fad8fc91a6433bda54d6177f0b82b9))
* comprehensive architecture documentation for v2.0.0 audit ([4b397b9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4b397b9740def27cd5e68d2de7ceff3f535a1de1))
* update README with logo and repository structure ([158f901](https://github.com/RaphaEnterprises-AI/argus-backend/commit/158f901b8469b72c58450b9a624bdf57991b5b94))


### Code Refactoring

* replace Upstash with Cloudflare KV and Vectorize ([71f8d56](https://github.com/RaphaEnterprises-AI/argus-backend/commit/71f8d5662f684d9f37065f1340271289fc7cc0d1))

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

# Changelog

## [2.12.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.11.0...v2.12.0) (2026-01-29)


### Features

* **a2a:** implement Agent-to-Agent communication architecture ([a6bdddc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a6bdddc2fa6fd45a1cb97781d6f57a6dbb35af73))
* add correlation engine and Tier 2 integration webhooks (RAP-82, RAP-84) ([d318059](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d318059903edf136cb5482e11a878f5033e50229))
* add GitHub webhooks, PR comments, and API testing (RAP-90, RAP-91, RAP-92) ([aeeefed](https://github.com/RaphaEnterprises-AI/argus-backend/commit/aeeefed4768ec1c1cc456551f4009742926c4c55))
* add missing API modules and integrations ([938917c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/938917c6d0648d0a0242efb0cdf70f1d1f85c1c1))
* add SAST integration, incident correlator, and Tier 1 integrations (RAP-83, RAP-93, RAP-94) ([54f60a1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/54f60a1449f4527f16583ac9089f3441e69818e7))
* **ai:** add multi-provider AI integration with 16 providers ([4c5991e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4c5991ee66bc447d52639bfa99f63728cf5264bc))
* **ai:** add OpenRouter and multi-provider support ([ce68998](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ce68998a4aee4d538d8bd3eaa7b90b8d38e14460))
* **api:** add authenticated monitoring proxy endpoints ([0212296](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0212296a4b0303f81896f565f6ab4cac996acd80))
* **api:** add browser pool REST endpoints ([369b61b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/369b61becf0f6af3765256118e7cbd34b26a6a7a))
* **api:** add extended model info fields for UI ([b031e05](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b031e0538c15709d6fdc21609566246fbe61d5f8))
* **api:** add real-time provider health checks ([e45e2f3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e45e2f339a5f30f95e4f415723ba71290bfef526))
* **api:** use Selenium Grid for video recording in browser tests ([e8c475c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e8c475c94d21170dd0efac54e06f30eb5754685a))
* **cache:** add Upstash Redis as preferred fallback (227x cheaper than KV) ([697f87a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/697f87a657c4379510bfd64dc857ad7cfc1c639a))
* **chat:** add visual regression, test export, and knowledge graph tools ([9e5b91b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9e5b91b17e5e91a09d74e0aa038e60234b33ede4))
* **chat:** add web search and semantic codebase search tools ([2f3fbaf](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2f3fbafb084d000e585394f9333aaa9f77b03a60))
* **chat:** BYOK-only model - remove platform API key fallback ([22820dc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22820dc0c58d9316d22ac4750ae30498447c9bb8))
* Cloudflare-only BYOK encryption (remove local fallback) ([b74289b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b74289be1954ec2e327cf657b899dba3a0d4ab8f))
* **cognee:** add direct Langfuse SDK instrumentation ([0457663](https://github.com/RaphaEnterprises-AI/argus-backend/commit/04576639e31a9ab4398cb00432119e297ec1afd2))
* **cognee:** add langfuse package for LLM tracing ([a27e7a4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a27e7a434494e09b4393cb539f72f35650d9b060))
* **cognee:** add LiteLLM Langfuse callback for LLM tracing ([278db4e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/278db4e60dd9ebc53d7ca297d9d8a059d329be8d))
* complete A2A architecture with agent capabilities and monitoring ([f49704a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f49704a71d7c9c91c00d6dc0d509983c68d52bab))
* complete AI integration layer - wire all agents, dashboards, events ([3fd212f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3fd212fb14ca69392ed42e76611f12b30152a8ae))
* **crawlee:** add video recording support to discovery crawler ([4f4db43](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4f4db434ce25a814e4fdcb1b112d7b4a6a7cd7de))
* **data-layer:** implement multi-tenant Cognee knowledge graph pipeline ([d575a42](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d575a4291b6b3c8307b09d64c0ddcba58bc5bf94))
* **discovery:** add cloud persistence, SSE fixes, and incremental checkpoints ([44f9047](https://github.com/RaphaEnterprises-AI/argus-backend/commit/44f9047db38869e4224348db5ce7e34846352a60))
* **discovery:** add multi-backend execution context and session recording ([552908e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/552908e348192bb0ac3486102662d4bb9a38a0a0))
* **discovery:** add session recording support like BrowserStack/LambdaTest ([2adf07c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2adf07ce235e16a3a1d9b13ae4e39b96e3996ed1))
* **events:** add Redpanda client and streaming infrastructure ([acb68c1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/acb68c106f279d103fa7fcb05bc67a3366bc41ca))
* **flink:** add Apache Flink streaming platform for real-time analytics ([0955b24](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0955b24fe731cd3d545c8e12cf666f86812cf134))
* **impact-graph:** add test impact graph schema & API (RAP-88) ([feef70a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/feef70a566e288b4f344e696592fb79324017431))
* implement RAP-52, RAP-70, RAP-71 - Visual AI, Hybrid Search, Knowledge Graph ([cba7116](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cba7116973cf71f25d2140988605b5ba5a246e34))
* **infra:** add data layer health API and Kubernetes monitoring stack ([5e00a96](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5e00a967f052e551d96b868b2f777da7173e5c52))
* **infra:** add Flink and Grafana health checks to infrastructure page ([6d7cdb1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6d7cdb1e24ffc0459ca388b82955d60667b975fa))
* **infra:** add kube-prometheus-stack Helm values for monitoring ([eca1185](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eca1185f64761b1f377d3eb69b4ba2d3e3104414))
* **intelligence:** implement Unified Instant Intelligence Layer (UIIL) ([ca45c81](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ca45c8197f1ad25758e25629fe3c416f5a9d368e))
* **k8s:** add Upstash Redis credentials to cognee-worker ([4668395](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4668395a86757c6a8baa203f1f16baae03a99417))
* **keda:** configure KEDA ScaledObject for Cognee workers ([23da698](https://github.com/RaphaEnterprises-AI/argus-backend/commit/23da6984047dcc9e3533eab90f581e03f44fa741))
* **knowledge:** add Valkey caching and complete Cognee migration (RAP-132) ([692dd83](https://github.com/RaphaEnterprises-AI/argus-backend/commit/692dd83b6c9dba3ce846fe26f336e48520a731ab))
* **knowledge:** consolidate memory systems to Cognee (RAP-132) ([e8d8500](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e8d8500aef99cb640eecab9ea161b88f427935f1))
* **monitoring:** add Cloudflare Tunnel for secure external access ([92f7659](https://github.com/RaphaEnterprises-AI/argus-backend/commit/92f765967cc2ca501495dd15ab8c284ffb5f8de2))
* **monitoring:** add Upstash and enhanced Cognee metrics to Grafana dashboards ([67294f9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/67294f9368b22a50dba1c81185da7c8c1ae76613))
* **monitoring:** update backend proxy for kube-prometheus-stack ([45b96cb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/45b96cb09c9324f621e52f0e9b1bb9f02c6d5aa7))
* **observability:** add centralized Langfuse LLM tracing across all AI calls ([845d69d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/845d69dd053e87e3d3a635e0fb03669b4ba6a4e2))
* **patterns:** add failure pattern learning system (RAP-89) ([38c95f6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/38c95f6830563d1ad6bcb880b52827ae9731e382))
* **profile:** add professional info, social links, and avatar upload ([a490e76](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a490e7637d1e0cc6597cad9d133523bad4006401))
* **scheduler:** add automatic cleanup of stale/stuck runs ([86c723d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/86c723dfa30edaa3059946856c713ef3dbe727bd))
* **scheduler:** add comprehensive AI features for intelligent test execution ([6506dc7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6506dc7b81b80d0a0ff8c010052f31862ca29880))
* **scheduler:** implement real test execution for scheduled runs ([770ed4a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/770ed4a594fab32448a4ca4567d2cca9521d6fbd))
* **security:** add Cloudflare Access support for monitoring proxy ([5818484](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5818484f2e508efe880e85f506cc773ce9140217))
* **storage:** unified R2 video storage with Selenium Grid integration ([18ab1b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/18ab1b4f89f3cea9b3d2396babda83c31df21257))
* **terraform:** add Confluent Cloud infrastructure as code ([ca74cf2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ca74cf2f8601980de66bfb125d04741c9e92f9d8))
* **video:** add video recording for discovery sessions via Selenium Grid ([3ad10ac](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3ad10ac9d3ad3a34807f17029409017eb6d66c70))
* **worker:** add Brain service proxy for /api/v1/users/* endpoints ([3bfb6b2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3bfb6b24205146f4196ec5c3e3d2e9d47e55305d))


### Bug Fixes

* add prometheus-client dependency for intelligence metrics ([080fc5b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/080fc5bdc6979164f94cd4776290cb5d246ea00c))
* **ai-analysis:** extract error from step results for assertion failures ([45f7952](https://github.com/RaphaEnterprises-AI/argus-backend/commit/45f795256fa78e6205bda1e4a66ebd0c69f1a189))
* **ai-analysis:** improve root cause analyzer parsing and error extraction ([cac5e77](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cac5e77b4b004e2e2ab806dd926d24fdf5535084))
* **ai-settings:** add fallback when Cloudflare Key Vault fails ([8f8aa20](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8f8aa2005101706053a00e20056d19f2083eb4db))
* **ai-settings:** comprehensive error handling for all endpoints ([28739b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/28739b44f6d0448395111235a9c70faed9672684))
* **ai-settings:** resolve undefined variable in add_provider_key ([e37843e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e37843ee3d83ca32b488c016c840a2e986c1e20d))
* **ai:** align Provider enums across all modules ([d70bbea](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d70bbea63e0fb7f450dee46611da3c3342b86875))
* **api:** add signed recording_url to browser test response ([0ba06ff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0ba06ffe9b44784ba1e9ab1ee2343e897d7da5e1))
* **api:** align ProviderResponse with frontend ProviderInfo interface ([71c48a3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/71c48a307acf95d079286f3d2c236d7de52bf1f2))
* **api:** apply screenshot conversion to act endpoint ([38fb100](https://github.com/RaphaEnterprises-AI/argus-backend/commit/38fb100535ecfcbba2fd9473fc467d420b757c1b))
* **api:** handle Buffer screenshot format from browser pool ([5bcc9a9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bcc9a931adaf6c5e55179d6579f7bf47fe60e20))
* **api:** persist API test runs to Supabase for UI visibility ([0b5933f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0b5933f9a9b907a4aca78265f343adb0b7c19d63))
* **api:** use get_settings() factory instead of settings global ([f363972](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f363972691bd3cc3b8a210ade96139f6d39a5399))
* **auth:** add Cloudflare Access headers to all Prometheus/Grafana requests ([222a6e5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/222a6e5588d83e2df490480fd7221699bcc28a68))
* **auth:** resolve Clerk user_id for API key authentication ([bba024d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bba024d1a0999915b3d138625d1035c16d326a5c))
* **auth:** support JWT token in query params for SSE endpoints ([6787eca](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6787eca517028c29dd5e068429aaaac3fd216e62))
* **chat:** always try BYOK key lookup for authenticated users ([5169fe8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5169fe86484a7dc854387637fdd811f502713ecf))
* **chat:** catch errors during AI config building ([1fe8090](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1fe8090fbd084ab308f8f97524861de664de70c6))
* **chat:** restore platform API key fallback for chat ([1f9fdb7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1f9fdb76a50e9841534d49d70744b64d65786792))
* **chat:** use authenticated user_id for BYOK key lookup ([6b5902a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6b5902a84247add3b19fdc412631d80712beda62))
* **cognee-worker:** add SSL context for Redpanda Cloud SASL_SSL ([c0ff0cf](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c0ff0cfcb392f2ab824c01518bf0ba0dc8e91a5f))
* **cognee-worker:** disable backend access control for Neo4j compatibility ([a18ed7d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a18ed7d26c57c69ba2df2d50e8bd58ddf7fe7b34))
* **cognee-worker:** handle ArgusEvent 'data' field in payload extraction ([3fb6d89](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3fb6d89eab33b2e7cb5f6289bcee7bceca6af43f))
* **cognee-worker:** update search API to Cognee 0.5+ positional args ([9c19794](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9c19794974c2a1d9da1cdbdd6581c4ffa170c20e))
* **cognee:** add proper LLM and database configuration ([35e1459](https://github.com/RaphaEnterprises-AI/argus-backend/commit/35e145947fe7e92ac06d62fe9184f88183d91b83))
* **cognee:** connect to Redpanda Cloud instead of self-hosted K8s instance ([b6aa9ee](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b6aa9eed73a8405f647629ab65c9114e4b269c39))
* **cognee:** enable Redpanda Cloud connectivity with proper SSL and NetworkPolicy ([f7ca948](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f7ca948d4b7e69ef9edd537a436461e585ae41a6))
* **cognee:** handle nested payload structure in event parsing ([edb18bc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/edb18bc4e855e6dbec2726807d1848d804b65f09))
* **cognee:** make Cognee mandatory with proper error handling ([f18166e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f18166e1f83ae34710004164d705718978da40b0))
* **cognee:** pin Langfuse SDK to 2.x for REST API ([a488ea3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a488ea3495a244ec60d0b9d292269b374a5ed8cb))
* **cognee:** simplify Langfuse LiteLLM callback registration ([6328d1d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6328d1d2d21c5cb8f7c0ae05824c4d4ec9c340df))
* **cognee:** update search API parameters to match Cognee v0.3+ ([b456e25](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b456e2556153839d3a2cf795d8ea87267ab91374))
* **cognee:** use environment variables for PostgreSQL configuration ([8c1d893](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8c1d893b67ed303a1fb887f5c72a1c5689ac5003))
* **cognee:** use PostgreSQL instead of SQLite on Railway ([7fbe991](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7fbe991eb3d339c4b2fcbd15be47a113c9b72bb6))
* **crawlee:** properly configure video recording for discovery crawl ([19429f2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/19429f25a2a89d7654b76a36d42b0bced196ac9a))
* **crawlee:** use launchOptions for video recording per Crawlee [#1849](https://github.com/RaphaEnterprises-AI/argus-backend/issues/1849) ([240e685](https://github.com/RaphaEnterprises-AI/argus-backend/commit/240e685fb651c24423bbc38a32fb5266b41a920c))
* **crawlee:** use page.video().saveAs() to properly save recordings ([bc372dd](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bc372dd6cf3908f28009ccd88e6ccffb78d94d7a))
* **deps:** add langchain for Langfuse CallbackHandler ([b89ef56](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b89ef567d3d778d5909c992d8fc3f07de3a45978))
* **deps:** add langfuse to Dockerfile fallback dependencies ([eb5e340](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eb5e340b7421130eda7f200516d8449fb7b463cb))
* **discovery:** add app_url to all persistence functions ([43bac0d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43bac0d63b82927cd454aff85738b5a01ab15756))
* **discovery:** add pages_found and flows_found columns ([5f34263](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5f342636b766b9054b34b6ce309179d0394d2244))
* **discovery:** add project_id to pages and flows persistence ([043a686](https://github.com/RaphaEnterprises-AI/argus-backend/commit/043a686b9f15306fa2f84246a450af2e63b74ebe))
* **discovery:** add updated_at field to persistence records ([871e1bc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/871e1bca8f8b528f034d848302219eecb0f6e8ab))
* **discovery:** fix database persistence for pages and flows ([7738274](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7738274b9f74a757ef1bfa77a7ef2e19862a20cd))
* **discovery:** handle both element_count and elements_count field names ([c94ac71](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c94ac712f9ac32bc2e0d877ebbe5ad34ca2e9eec))
* **discovery:** handle duplicate page/flow errors gracefully ([b31df9a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b31df9acb9fd6c2c3476589a1b74384eee047fa9))
* **discovery:** improve checkpoint error logging ([5fc0554](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5fc05543cc5b62f42a541e60cb58bad4c0a9f4d1))
* **discovery:** improve data persistence and session naming ([5598cd7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5598cd799bfa543e03509ad17e883605de5ceee3))
* **discovery:** include video_artifact_id when loading sessions from DB ([3404eb7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3404eb77abd7ae4e697867bde751f4ead29fc508))
* **discovery:** load completed sessions from database on refresh ([0a3be56](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0a3be565d6a2559de2d18b7d3b4baaa3ca881a93))
* **discovery:** make generate-test request body optional ([0e21792](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0e21792d2a105682185af13aeec5d81e54f41007))
* **discovery:** persist sessions to database immediately and add missing columns ([f501ea4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f501ea4cd3b98505b120662818c5f4bd85ed4781))
* **discovery:** prevent duplicate page/flow inserts ([fcc25f1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fcc25f1e1982716cf1e54391da5feb542a412fb7))
* **discovery:** regenerate signed video URLs on each API fetch ([df4018e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df4018ea8bc0608b8b027566c43a391105992da7))
* **discovery:** revert updated_at code and add DB migration ([398acff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/398acff13c0e25a7ab92d4108c183e837174f940))
* **discovery:** route video-enabled sessions to BrowserPool ([3ca37af](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3ca37af503b8612965503416cabdcba4b9353539))
* **docker:** add prometheus-client to Dockerfile fallback deps ([b99e907](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b99e907871991245ffba1eee4bec2fab274928c6))
* **event-gateway:** add SSL context for SASL_SSL connections ([1694485](https://github.com/RaphaEnterprises-AI/argus-backend/commit/169448555881a067fbddf8d4cb95d8ccc2ad8954))
* **event-gateway:** read SASL config from env vars, use gzip compression ([d4a963a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4a963af58b00a9ed41880fb2f79abbe47716e5f))
* **grafana:** enable anonymous access behind Cloudflare Access ([5fba7cb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5fba7cb364db4d5952378c3f5ca521270efcb149))
* **health:** correct aiokafka admin client import path ([4c7810d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4c7810d6257fcd3d699202d07f429ea9c25208fe))
* **health:** handle self-signed SSL certs and Supabase auth ([22673ee](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22673ee0d5bbee84762ec8a9800bebf7140eaa60))
* **health:** show internal K8s services as healthy with proxy info ([a5b1143](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a5b11437f2db1b04d023a695092ad5a1c3c7b35a))
* **incident_correlator:** handle duplicate timezone suffixes in timestamps ([dbf14fa](https://github.com/RaphaEnterprises-AI/argus-backend/commit/dbf14fa2222088965d5c8b1dd5322daf5055f46e))
* **infra:** remove BrowserStack comparison and add platform-specific costs ([5bbdc2e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bbdc2ef1b2ee3c998d96083c71850ae214911d2))
* **intelligence:** add Cloudflare KV fallback when Valkey unreachable ([384ec9f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/384ec9fdbfe1c0111e77d3a36dd5c1ab6c93f155))
* **langfuse:** update import path for Langfuse v2 SDK ([baca510](https://github.com/RaphaEnterprises-AI/argus-backend/commit/baca510ada7de43255110d4dd487efcc616dfb41))
* lint issues - import sorting, deprecated typing, duplicate key ([af03b38](https://github.com/RaphaEnterprises-AI/argus-backend/commit/af03b3826ff81cc791de7a87e820cad328e31a69))
* **lint:** resolve I001, F821, F541 lint errors in new modules ([5bcee9d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bcee9d5bdca6e6b3d6abc7df5b17aab670f47f8))
* **migrations:** fix Supabase deployment issues ([f97a4c2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f97a4c26857cb189105d728ab0f1aa2fbfa43902))
* **migrations:** resolve type mismatch and version conflicts ([56da4df](https://github.com/RaphaEnterprises-AI/argus-backend/commit/56da4dfde7e0d86eb0bc4c6b451d80950e44c39c))
* **monitoring:** add CF_ACCESS headers to all monitoring endpoints ([fb281ce](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fb281ce0ee7940be6747534b1156425c91e06194))
* **monitoring:** add UID to Prometheus datasource for Grafana dashboards ([7e9d7b7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7e9d7b7033c2240e411d1132a72b1af74ca6aa48))
* **monitoring:** configure Grafana datasource UID correctly ([702c123](https://github.com/RaphaEnterprises-AI/argus-backend/commit/702c123b84f894de8844227bb8baa8f2216a2541))
* **monitoring:** remove duplicate datasource definition in Helm values ([f1addab](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f1addabee14a069ef4759998146f7f3e0e9575f0))
* **network:** allow Cognee worker to reach Langfuse ([8b2b25d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8b2b25d00008580a99139d122fd55a98b58261de))
* resolve all lint errors for CI ([7d08d9b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7d08d9b716c82db2e63893939f1d82d6b300970b))
* resolve lint errors (F541, I001) in src and tests ([d4ab3d0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4ab3d02054cec12ae968e557fe79b76ddb7e5f3))
* resolve lint errors in test and source files ([196778f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/196778faf08a62ed85779c2b8cd66d19dd7ed3f5))
* **scheduler:** add error handling to AI analysis aggregation ([710c36d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/710c36d46393f41f2e638beda6db99bc0f0bd6a6))
* **scheduler:** add missing status and trigger_type values to ScheduleRunResponse ([16d4cd6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/16d4cd64d0239592f6b4e3e87988a124af45df6a))
* **scheduler:** add missing status values to last_run_status Literal type ([22a4ac6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22a4ac6f2000622406c8d9a36f4e0efa12866b91))
* **scheduler:** add proper assertion handling with assertion_type support ([eb15ea9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eb15ea916882d1035aee346eefaa4cc976288d36))
* **scheduler:** handle non-numeric wait values like networkidle ([c6da61f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c6da61f3af2155d57d30ad66cdbb1fba0c01a094))
* **scheduler:** save AI analysis fields to schedule_runs database ([6a4ccce](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6a4ccce09e9f5f037941c57ae501049d03bfc116))
* **scheduling:** fix Supabase query filters and update argument order ([242250d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/242250d5f80beca5837576323b254d6916f7cf59))
* **security:** address vulnerabilities from security audit ([efff8ec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/efff8ecb47e662167c7869e024eca3f3aa970027))
* **security:** fix RLS policies for reports and discovery_history tables ([ee571b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ee571b4570896a5307c50ab9590a5b0a8c957c60))
* **selenium-grid:** prevent zombie sessions and use KEDA selenium-grid scaler ([7ea8c5b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7ea8c5bd96914c3a0d9039fb6c36a8b8c4627cb7))
* **server:** add missing asyncio import for stale run checker ([0be9054](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0be9054bf121e19967a0165f055aaf68d07b171c))
* **tests:** add mock_user to all send_message calls in test_chat.py ([8af391b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8af391b5958ac4f53c90b515d636b601bd32c199))
* **tests:** add mock_user to TestStreamMessageEndpoint tests ([bff0986](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bff0986faf46a1b6d3ff9a917696fb02b3aea294))
* **tests:** implement actual step execution in Selenium Grid test runner ([295d5ec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/295d5ec13d10774c158399590241fa8da1b0acc9))
* **tests:** pass user param in test_send_message_generates_thread_id ([5bfdb31](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bfdb31aeb6c631541153c06387ba409de35528d))
* **tests:** remove extraneous f-string prefixes ([f13aedc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f13aedc6b3560599259869880a566dda8ce2e5a6))
* use 'body' instead of 'data' in audit logger Supabase request ([f24e1a0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f24e1a025e013a316c16f47d112ae2ff47d47502))
* use Pydantic alias for underscore-prefixed LaunchDarkly fields ([66ef881](https://github.com/RaphaEnterprises-AI/argus-backend/commit/66ef881a7a3d9d1e55e88c7e95cd0efab2e5920d))
* **webhooks:** export combined router from webhooks module ([3403ee1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3403ee17dd36862440ae9529ee2f5f52416ab193))


### Documentation

* add API versioning strategy and version control guide ([43c1d2f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43c1d2f1bd93cdac15bfc393b99cfe7fb089428e))
* add architecture and business documentation ([28edeec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/28edeec3fb1f4c4c53605cdbcdb4c26a7ee82083))
* **architecture:** add comprehensive AI intelligence implementation blueprint ([c1bdaed](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c1bdaedd931c28b3025febfd43b8db0ba517f72c))
* **CLAUDE.md:** add A2A architecture documentation ([da77db2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/da77db28b4efe6de3282ccfa98b54edf8da1ef08))
* **security:** add penetration testing scope document ([f9c51f1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f9c51f11822f1c071a8169b91700c8686f1c7af8))
* **security:** add secrets management and rotation guide ([c88d9de](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c88d9de9489b72c30f24f0f62442be38e18884c4))
* **security:** add SOC 2 Type II gap analysis and policy templates ([d45d95b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d45d95b6db942c951467e53aef7d929b559d018e))

## [2.11.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.10.0...v2.11.0) (2026-01-28)


### Features

* **a2a:** implement Agent-to-Agent communication architecture ([a6bdddc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a6bdddc2fa6fd45a1cb97781d6f57a6dbb35af73))
* add correlation engine and Tier 2 integration webhooks (RAP-82, RAP-84) ([d318059](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d318059903edf136cb5482e11a878f5033e50229))
* add enterprise multi-agent architecture and evaluation framework ([d1c6961](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d1c696179aca1a0ee6edf6200b5b18c0560efa96))
* add GitHub webhooks, PR comments, and API testing (RAP-90, RAP-91, RAP-92) ([aeeefed](https://github.com/RaphaEnterprises-AI/argus-backend/commit/aeeefed4768ec1c1cc456551f4009742926c4c55))
* add HMAC-signed URLs for authenticated media access ([999e44c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/999e44ca938d456e5fcc4b26260c8cd11d14e185))
* add missing API modules and integrations ([938917c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/938917c6d0648d0a0242efb0cdf70f1d1f85c1c1))
* add SAST integration, incident correlator, and Tier 1 integrations (RAP-83, RAP-93, RAP-94) ([54f60a1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/54f60a1449f4527f16583ac9089f3441e69818e7))
* **ai:** add multi-provider AI integration with 16 providers ([4c5991e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4c5991ee66bc447d52639bfa99f63728cf5264bc))
* **api:** add authenticated monitoring proxy endpoints ([0212296](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0212296a4b0303f81896f565f6ab4cac996acd80))
* **api:** add browser pool REST endpoints ([369b61b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/369b61becf0f6af3765256118e7cbd34b26a6a7a))
* **api:** use Selenium Grid for video recording in browser tests ([e8c475c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e8c475c94d21170dd0efac54e06f30eb5754685a))
* **audit:** migrate audit logs to cloud-based Supabase storage ([1fa2fa3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1fa2fa311b5c857a298fcbccea161ebd86a8f363))
* BYOK multi-provider AI system with Cloudflare Key Vault ([3d3bd8a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3d3bd8a5f4c7dc03b424f811681d5930413449a8))
* **chat:** add self-healing intelligence and R2 screenshot proxy ([d462c48](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d462c487306f8bb8436ede91942fffc791a3ca5f))
* **chat:** BYOK-only model - remove platform API key fallback ([22820dc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22820dc0c58d9316d22ac4750ae30498447c9bb8))
* **checkpointer:** upgrade to production-grade async PostgresSaver ([98b752e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/98b752e596d615889df7d4f41d8a13703d4ab475))
* Cloudflare-only BYOK encryption (remove local fallback) ([b74289b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b74289be1954ec2e327cf657b899dba3a0d4ab8f))
* complete A2A architecture with agent capabilities and monitoring ([f49704a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f49704a71d7c9c91c00d6dc0d509983c68d52bab))
* complete AI integration layer - wire all agents, dashboards, events ([3fd212f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3fd212fb14ca69392ed42e76611f12b30152a8ae))
* **crawlee:** add video recording support to discovery crawler ([4f4db43](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4f4db434ce25a814e4fdcb1b112d7b4a6a7cd7de))
* **data-layer:** implement multi-tenant Cognee knowledge graph pipeline ([d575a42](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d575a4291b6b3c8307b09d64c0ddcba58bc5bf94))
* **discovery:** add cloud persistence, SSE fixes, and incremental checkpoints ([44f9047](https://github.com/RaphaEnterprises-AI/argus-backend/commit/44f9047db38869e4224348db5ce7e34846352a60))
* **discovery:** add multi-backend execution context and session recording ([552908e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/552908e348192bb0ac3486102662d4bb9a38a0a0))
* **discovery:** add session recording support like BrowserStack/LambdaTest ([2adf07c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2adf07ce235e16a3a1d9b13ae4e39b96e3996ed1))
* **events:** add Redpanda client and streaming infrastructure ([acb68c1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/acb68c106f279d103fa7fcb05bc67a3366bc41ca))
* **flink:** add Apache Flink streaming platform for real-time analytics ([0955b24](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0955b24fe731cd3d545c8e12cf666f86812cf134))
* **impact-graph:** add test impact graph schema & API (RAP-88) ([feef70a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/feef70a566e288b4f344e696592fb79324017431))
* implement RAP-52, RAP-70, RAP-71 - Visual AI, Hybrid Search, Knowledge Graph ([cba7116](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cba7116973cf71f25d2140988605b5ba5a246e34))
* **infra:** add data layer health API and Kubernetes monitoring stack ([5e00a96](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5e00a967f052e551d96b868b2f777da7173e5c52))
* **infra:** add Flink and Grafana health checks to infrastructure page ([6d7cdb1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6d7cdb1e24ffc0459ca388b82955d60667b975fa))
* **infra:** add kube-prometheus-stack Helm values for monitoring ([eca1185](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eca1185f64761b1f377d3eb69b4ba2d3e3104414))
* **keda:** configure KEDA ScaledObject for Cognee workers ([23da698](https://github.com/RaphaEnterprises-AI/argus-backend/commit/23da6984047dcc9e3533eab90f581e03f44fa741))
* **knowledge:** add Valkey caching and complete Cognee migration (RAP-132) ([692dd83](https://github.com/RaphaEnterprises-AI/argus-backend/commit/692dd83b6c9dba3ce846fe26f336e48520a731ab))
* **knowledge:** consolidate memory systems to Cognee (RAP-132) ([e8d8500](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e8d8500aef99cb640eecab9ea161b88f427935f1))
* **monitoring:** add Cloudflare Tunnel for secure external access ([92f7659](https://github.com/RaphaEnterprises-AI/argus-backend/commit/92f765967cc2ca501495dd15ab8c284ffb5f8de2))
* **monitoring:** update backend proxy for kube-prometheus-stack ([45b96cb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/45b96cb09c9324f621e52f0e9b1bb9f02c6d5aa7))
* **patterns:** add failure pattern learning system (RAP-89) ([38c95f6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/38c95f6830563d1ad6bcb880b52827ae9731e382))
* **scheduler:** add automatic cleanup of stale/stuck runs ([86c723d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/86c723dfa30edaa3059946856c713ef3dbe727bd))
* **scheduler:** add comprehensive AI features for intelligent test execution ([6506dc7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6506dc7b81b80d0a0ff8c010052f31862ca29880))
* **scheduler:** implement real test execution for scheduled runs ([770ed4a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/770ed4a594fab32448a4ca4567d2cca9521d6fbd))
* **security:** add Cloudflare Access support for monitoring proxy ([5818484](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5818484f2e508efe880e85f506cc773ce9140217))
* **storage:** unified R2 video storage with Selenium Grid integration ([18ab1b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/18ab1b4f89f3cea9b3d2396babda83c31df21257))
* **terraform:** add Confluent Cloud infrastructure as code ([ca74cf2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ca74cf2f8601980de66bfb125d04741c9e92f9d8))
* **video:** add video recording for discovery sessions via Selenium Grid ([3ad10ac](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3ad10ac9d3ad3a34807f17029409017eb6d66c70))
* **worker:** add Brain service proxy for /api/v1/users/* endpoints ([3bfb6b2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3bfb6b24205146f4196ec5c3e3d2e9d47e55305d))


### Bug Fixes

* **ai-analysis:** extract error from step results for assertion failures ([45f7952](https://github.com/RaphaEnterprises-AI/argus-backend/commit/45f795256fa78e6205bda1e4a66ebd0c69f1a189))
* **ai-analysis:** improve root cause analyzer parsing and error extraction ([cac5e77](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cac5e77b4b004e2e2ab806dd926d24fdf5535084))
* **ai-settings:** add fallback when Cloudflare Key Vault fails ([8f8aa20](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8f8aa2005101706053a00e20056d19f2083eb4db))
* **ai-settings:** comprehensive error handling for all endpoints ([28739b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/28739b44f6d0448395111235a9c70faed9672684))
* **ai-settings:** resolve undefined variable in add_provider_key ([e37843e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e37843ee3d83ca32b488c016c840a2e986c1e20d))
* **api:** add signed recording_url to browser test response ([0ba06ff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0ba06ffe9b44784ba1e9ab1ee2343e897d7da5e1))
* **api:** apply screenshot conversion to act endpoint ([38fb100](https://github.com/RaphaEnterprises-AI/argus-backend/commit/38fb100535ecfcbba2fd9473fc467d420b757c1b))
* **api:** handle Buffer screenshot format from browser pool ([5bcc9a9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bcc9a931adaf6c5e55179d6579f7bf47fe60e20))
* **api:** persist API test runs to Supabase for UI visibility ([0b5933f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0b5933f9a9b907a4aca78265f343adb0b7c19d63))
* **auth:** add Cloudflare Access headers to all Prometheus/Grafana requests ([222a6e5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/222a6e5588d83e2df490480fd7221699bcc28a68))
* **auth:** resolve Clerk user_id for API key authentication ([bba024d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bba024d1a0999915b3d138625d1035c16d326a5c))
* **auth:** support JWT token in query params for SSE endpoints ([6787eca](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6787eca517028c29dd5e068429aaaac3fd216e62))
* **chat:** always try BYOK key lookup for authenticated users ([5169fe8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5169fe86484a7dc854387637fdd811f502713ecf))
* **chat:** catch errors during AI config building ([1fe8090](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1fe8090fbd084ab308f8f97524861de664de70c6))
* **chat:** restore platform API key fallback for chat ([1f9fdb7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1f9fdb76a50e9841534d49d70744b64d65786792))
* **chat:** use authenticated user_id for BYOK key lookup ([6b5902a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6b5902a84247add3b19fdc412631d80712beda62))
* **checkpointer:** force IPv4 connection for Supabase ([df52f66](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df52f66e7f4fb65f68d90cbaf0a523499ec3591f))
* **checkpointer:** properly enter async context manager ([b0cd869](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b0cd86942af06433bee9ffc6a35907ac15bb5f25))
* **checkpointer:** use from_conn_string API for AsyncPostgresSaver ([d57f3f7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d57f3f7d47e98d458f4b02ef24674de4d3fed205))
* **checkpointer:** use getaddrinfo with AF_INET for IPv4 resolution ([eaf9ffe](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eaf9ffef6c9b7a4b5c82394753ff4e3741fc8c9d))
* **cognee:** add proper LLM and database configuration ([35e1459](https://github.com/RaphaEnterprises-AI/argus-backend/commit/35e145947fe7e92ac06d62fe9184f88183d91b83))
* **cognee:** make Cognee mandatory with proper error handling ([f18166e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f18166e1f83ae34710004164d705718978da40b0))
* **cognee:** update search API parameters to match Cognee v0.3+ ([b456e25](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b456e2556153839d3a2cf795d8ea87267ab91374))
* **cognee:** use environment variables for PostgreSQL configuration ([8c1d893](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8c1d893b67ed303a1fb887f5c72a1c5689ac5003))
* **cognee:** use PostgreSQL instead of SQLite on Railway ([7fbe991](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7fbe991eb3d339c4b2fcbd15be47a113c9b72bb6))
* correct Worker URL default for screenshot proxy ([868e778](https://github.com/RaphaEnterprises-AI/argus-backend/commit/868e778f238cfa8ac69536fd3508851638118782))
* **crawlee:** properly configure video recording for discovery crawl ([19429f2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/19429f25a2a89d7654b76a36d42b0bced196ac9a))
* **crawlee:** use launchOptions for video recording per Crawlee [#1849](https://github.com/RaphaEnterprises-AI/argus-backend/issues/1849) ([240e685](https://github.com/RaphaEnterprises-AI/argus-backend/commit/240e685fb651c24423bbc38a32fb5266b41a920c))
* **crawlee:** use page.video().saveAs() to properly save recordings ([bc372dd](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bc372dd6cf3908f28009ccd88e6ccffb78d94d7a))
* **discovery:** add app_url to all persistence functions ([43bac0d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43bac0d63b82927cd454aff85738b5a01ab15756))
* **discovery:** add pages_found and flows_found columns ([5f34263](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5f342636b766b9054b34b6ce309179d0394d2244))
* **discovery:** add project_id to pages and flows persistence ([043a686](https://github.com/RaphaEnterprises-AI/argus-backend/commit/043a686b9f15306fa2f84246a450af2e63b74ebe))
* **discovery:** add updated_at field to persistence records ([871e1bc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/871e1bca8f8b528f034d848302219eecb0f6e8ab))
* **discovery:** fix database persistence for pages and flows ([7738274](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7738274b9f74a757ef1bfa77a7ef2e19862a20cd))
* **discovery:** handle both element_count and elements_count field names ([c94ac71](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c94ac712f9ac32bc2e0d877ebbe5ad34ca2e9eec))
* **discovery:** handle duplicate page/flow errors gracefully ([b31df9a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b31df9acb9fd6c2c3476589a1b74384eee047fa9))
* **discovery:** improve checkpoint error logging ([5fc0554](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5fc05543cc5b62f42a541e60cb58bad4c0a9f4d1))
* **discovery:** improve data persistence and session naming ([5598cd7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5598cd799bfa543e03509ad17e883605de5ceee3))
* **discovery:** include video_artifact_id when loading sessions from DB ([3404eb7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3404eb77abd7ae4e697867bde751f4ead29fc508))
* **discovery:** load completed sessions from database on refresh ([0a3be56](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0a3be565d6a2559de2d18b7d3b4baaa3ca881a93))
* **discovery:** make generate-test request body optional ([0e21792](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0e21792d2a105682185af13aeec5d81e54f41007))
* **discovery:** persist sessions to database immediately and add missing columns ([f501ea4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f501ea4cd3b98505b120662818c5f4bd85ed4781))
* **discovery:** prevent duplicate page/flow inserts ([fcc25f1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fcc25f1e1982716cf1e54391da5feb542a412fb7))
* **discovery:** regenerate signed video URLs on each API fetch ([df4018e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df4018ea8bc0608b8b027566c43a391105992da7))
* **discovery:** revert updated_at code and add DB migration ([398acff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/398acff13c0e25a7ab92d4108c183e837174f940))
* **discovery:** route video-enabled sessions to BrowserPool ([3ca37af](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3ca37af503b8612965503416cabdcba4b9353539))
* **event-gateway:** read SASL config from env vars, use gzip compression ([d4a963a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4a963af58b00a9ed41880fb2f79abbe47716e5f))
* **grafana:** enable anonymous access behind Cloudflare Access ([5fba7cb](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5fba7cb364db4d5952378c3f5ca521270efcb149))
* **health:** correct aiokafka admin client import path ([4c7810d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4c7810d6257fcd3d699202d07f429ea9c25208fe))
* **health:** handle self-signed SSL certs and Supabase auth ([22673ee](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22673ee0d5bbee84762ec8a9800bebf7140eaa60))
* **health:** show internal K8s services as healthy with proxy info ([a5b1143](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a5b11437f2db1b04d023a695092ad5a1c3c7b35a))
* **incident_correlator:** handle duplicate timezone suffixes in timestamps ([dbf14fa](https://github.com/RaphaEnterprises-AI/argus-backend/commit/dbf14fa2222088965d5c8b1dd5322daf5055f46e))
* **infra:** remove BrowserStack comparison and add platform-specific costs ([5bbdc2e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bbdc2ef1b2ee3c998d96083c71850ae214911d2))
* lint issues - import sorting, deprecated typing, duplicate key ([af03b38](https://github.com/RaphaEnterprises-AI/argus-backend/commit/af03b3826ff81cc791de7a87e820cad328e31a69))
* **lint:** resolve I001, F821, F541 lint errors in new modules ([5bcee9d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bcee9d5bdca6e6b3d6abc7df5b17aab670f47f8))
* **migrations:** fix Supabase deployment issues ([f97a4c2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f97a4c26857cb189105d728ab0f1aa2fbfa43902))
* **migrations:** resolve type mismatch and version conflicts ([56da4df](https://github.com/RaphaEnterprises-AI/argus-backend/commit/56da4dfde7e0d86eb0bc4c6b451d80950e44c39c))
* **monitoring:** add CF_ACCESS headers to all monitoring endpoints ([fb281ce](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fb281ce0ee7940be6747534b1156425c91e06194))
* resolve all lint errors for CI ([7d08d9b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7d08d9b716c82db2e63893939f1d82d6b300970b))
* resolve lint errors (F541, I001) in src and tests ([d4ab3d0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4ab3d02054cec12ae968e557fe79b76ddb7e5f3))
* resolve lint errors in test and source files ([196778f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/196778faf08a62ed85779c2b8cd66d19dd7ed3f5))
* **scheduler:** add error handling to AI analysis aggregation ([710c36d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/710c36d46393f41f2e638beda6db99bc0f0bd6a6))
* **scheduler:** add missing status and trigger_type values to ScheduleRunResponse ([16d4cd6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/16d4cd64d0239592f6b4e3e87988a124af45df6a))
* **scheduler:** add missing status values to last_run_status Literal type ([22a4ac6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/22a4ac6f2000622406c8d9a36f4e0efa12866b91))
* **scheduler:** add proper assertion handling with assertion_type support ([eb15ea9](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eb15ea916882d1035aee346eefaa4cc976288d36))
* **scheduler:** handle non-numeric wait values like networkidle ([c6da61f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c6da61f3af2155d57d30ad66cdbb1fba0c01a094))
* **scheduler:** save AI analysis fields to schedule_runs database ([6a4ccce](https://github.com/RaphaEnterprises-AI/argus-backend/commit/6a4ccce09e9f5f037941c57ae501049d03bfc116))
* **scheduling:** fix Supabase query filters and update argument order ([242250d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/242250d5f80beca5837576323b254d6916f7cf59))
* **security:** address vulnerabilities from security audit ([efff8ec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/efff8ecb47e662167c7869e024eca3f3aa970027))
* **security:** fix RLS policies for reports and discovery_history tables ([ee571b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ee571b4570896a5307c50ab9590a5b0a8c957c60))
* **selenium-grid:** prevent zombie sessions and use KEDA selenium-grid scaler ([7ea8c5b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/7ea8c5bd96914c3a0d9039fb6c36a8b8c4627cb7))
* **server:** add missing asyncio import for stale run checker ([0be9054](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0be9054bf121e19967a0165f055aaf68d07b171c))
* **tests:** add mock_user to all send_message calls in test_chat.py ([8af391b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8af391b5958ac4f53c90b515d636b601bd32c199))
* **tests:** add mock_user to TestStreamMessageEndpoint tests ([bff0986](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bff0986faf46a1b6d3ff9a917696fb02b3aea294))
* **tests:** implement actual step execution in Selenium Grid test runner ([295d5ec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/295d5ec13d10774c158399590241fa8da1b0acc9))
* **tests:** pass user param in test_send_message_generates_thread_id ([5bfdb31](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5bfdb31aeb6c631541153c06387ba409de35528d))
* **tests:** remove extraneous f-string prefixes ([f13aedc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f13aedc6b3560599259869880a566dda8ce2e5a6))
* use Pydantic alias for underscore-prefixed LaunchDarkly fields ([66ef881](https://github.com/RaphaEnterprises-AI/argus-backend/commit/66ef881a7a3d9d1e55e88c7e95cd0efab2e5920d))
* **webhooks:** export combined router from webhooks module ([3403ee1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3403ee17dd36862440ae9529ee2f5f52416ab193))


### Documentation

* add API versioning strategy and version control guide ([43c1d2f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43c1d2f1bd93cdac15bfc393b99cfe7fb089428e))
* add architecture and business documentation ([28edeec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/28edeec3fb1f4c4c53605cdbcdb4c26a7ee82083))
* **architecture:** add comprehensive AI intelligence implementation blueprint ([c1bdaed](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c1bdaedd931c28b3025febfd43b8db0ba517f72c))
* **CLAUDE.md:** add A2A architecture documentation ([da77db2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/da77db28b4efe6de3282ccfa98b54edf8da1ef08))
* **security:** add penetration testing scope document ([f9c51f1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f9c51f11822f1c071a8169b91700c8686f1c7af8))
* **security:** add secrets management and rotation guide ([c88d9de](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c88d9de9489b72c30f24f0f62442be38e18884c4))
* **security:** add SOC 2 Type II gap analysis and policy templates ([d45d95b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d45d95b6db942c951467e53aef7d929b559d018e))

## [2.10.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.9.1...v2.10.0) (2026-01-19)


### Features

* **artifacts:** add presigned URL support for R2 screenshots ([c3cf2b3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c3cf2b30971ea3e8214cd0781d656234cdf94e77))


### Bug Fixes

* **tests:** clear DATABASE_URL in mock_env_vars to prevent PostgresSaver issues ([ff6284a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ff6284a77989ef610238aa071993a7ed0cf546ae))
* **tests:** properly mock httpx.AsyncClient for BrowserPoolClient test ([15603be](https://github.com/RaphaEnterprises-AI/argus-backend/commit/15603be319c5f80ed941363eb335d47309672bb8))
* **tests:** update streaming test to patch EnhancedTestingOrchestrator ([e335929](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e3359293aead627465f3e4abcbff8f03ff40b4bb))
* **tests:** update streaming tests to patch create_enhanced_testing_graph ([e75d713](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e75d71393002c0e9bf1123a63db0ec7b19fa8c9d))
* **tests:** use MemorySaver directly in checkpointer integration tests ([d4b6f1e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d4b6f1ed171ddb065c3c2a1e617a23f25cd612ed))

## [2.9.1](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.9.0...v2.9.1) (2026-01-19)


### Bug Fixes

* **artifacts:** persist screenshots to R2 and Supabase ([e7cbcf7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e7cbcf7075b92521773d2ce8e16c28a44b59c45f))
* **storage:** handle Buffer format screenshots from browser pool ([d5e142f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d5e142fbc8382265eb857c228db42280b10096fb))

## [2.9.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.8.0...v2.9.0) (2026-01-19)


### Features

* **ci:** implement tiered testing with smoke tests for PRs ([a29ca9f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a29ca9fed1a1a2b3e8a3e9a2d6a223a49b6b9310))
* **dashboard:** add authenticated screenshot fetching ([b4c7956](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b4c7956392af7c607c01c5be8af9d336d4f7acd4))
* **logging:** add comprehensive audit logging across execution flow ([2dc4bb3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2dc4bb39c88971d88d462465d5e4ad8344577bea))
* **tests:** add comprehensive test coverage and CI/CD pipeline ([59d44ba](https://github.com/RaphaEnterprises-AI/argus-backend/commit/59d44ba29c633409a4f482e7883f95f939544573))


### Bug Fixes

* add missing desired_replicas variable ([dfcff8e](https://github.com/RaphaEnterprises-AI/argus-backend/commit/dfcff8e4b8f89d1e2ef2e48b2fe1f278fd200f25))
* **api:** resolve 5 E2E API test failures ([a7b9158](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a7b9158ed6091215881b5993b0ae17e0b5b08584))
* **ci:** remove dashboard jobs from backend test workflow ([da4496c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/da4496c1cab58fea93744b1c2544ca24919de1e0))
* **discovery:** add 'log out' keyword to authentication categorization ([bb1c6b1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bb1c6b1d214229b794dfbbb6d5a6cc5fc3bc9fef))
* **discovery:** move commerce keyword check before generic type checks ([fe91565](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fe9156557bd3e50d27d8be39a44844f8460404c9))
* **discovery:** swap UUID/number regex order in _normalize_title ([2935c70](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2935c705c36f24153369572beceb94c483264328))
* **infra:** update Prometheus collector for Vultr browser pool metrics ([700c8de](https://github.com/RaphaEnterprises-AI/argus-backend/commit/700c8ded7eaece48d973a9c69aa2e70510d257fe))
* **lint:** import Callable from collections.abc per Python 3.9+ convention ([b8213a7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b8213a78d4afb64dd885bfc434732ce681ef7688))
* **lint:** resolve all ruff lint errors ([79ffbb2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/79ffbb20c53b1fc473fd25de8cb653e726b063cc))
* **screenshots:** enable public access for artifact images ([4529efc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4529efc9cd59cb7bb88f2d5e7ed9d8b6e072368c))
* **supabase:** add offset parameter to select() for pagination ([3affa11](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3affa11965491c172476de7507c239ee0bc27701))
* **tests:** add AsyncMock for update_webhook_log in exception test ([df5ed31](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df5ed31bf3000fa2502e70e852028329bb4ca2a7))
* **tests:** add explicit project_id=None for webhook tests ([c81d569](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c81d569650b4be223658c2d6eeecff192446944f))
* **tests:** check lowercase 'uuid' in normalize_title test ([b582341](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b5823413770bc5f1335e04dcc023712b2513f2db))
* **tests:** correct argument order in scheduling endpoint tests ([e45fbc5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e45fbc50347e32ca96520a74816f3ab9a53a0b9c))
* **tests:** correct assertion for feature mesh error handling test ([af480ca](https://github.com/RaphaEnterprises-AI/argus-backend/commit/af480cadc2a87af08f9493dd0b0ecffb391257f9))
* **tests:** correct assertion for slack configuration message ([fdc7282](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fdc7282661857946987335a798d3d1876ff1ec30))
* **tests:** correct function names and AsyncMock usage in notification tests ([0df450b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/0df450b3f444eda09ef29ed354e52b96b47002ea))
* **tests:** correct mock patch path for get_supabase_client ([5c3a472](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5c3a472e2bfa299d17bcac09877dc13cea54ccaa))
* **tests:** correct scheduling endpoint tests for actual function signatures ([8f6dab2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8f6dab2eb148a1c517a24469a429ab000c90209d))
* **tests:** fix notification endpoint tests for correct function signatures ([33ae6fc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/33ae6fc2fabb599397af834abd8584b93bef0ec4))
* **tests:** mock BrowserPoolClient instead of httpx.AsyncClient ([df009e2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/df009e249d5b8fb7d1af1b647146994ceaab923d))
* **tests:** mock BrowserPoolClient.act() for executeAction test ([d46b546](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d46b5463036bc48d4ee918536c454294b9f2b45c))
* **tests:** mock pages and elements tables separately ([3e984a1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3e984a1292edf4d8149dd439b9c1ba43a2117580))
* **tests:** mock Path.exists for crawlee script not found test ([298ddec](https://github.com/RaphaEnterprises-AI/argus-backend/commit/298ddecb0fd8a5fdc065db1e1152791377e754cf))
* **tests:** skip broken playwright import test ([ba77273](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ba77273b24160fa57451ec03b78afeeefd807056))
* **tests:** skip flaky health_check_without_db test ([aacd33f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/aacd33f812d5908dba89b85d29df73b9de77196f))
* **tests:** use AsyncMock for check_connection in test_get_slack_status ([b549e42](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b549e429f6503301d378620d97a9c1cf4ffc4dd6))
* **tests:** use Callable type annotation instead of callable builtin ([e74a9c8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e74a9c8ab93307565335c4366bfc4c7a7c789e78))
* **tests:** use correct patch path for BrowserPoolClient ([cb448b1](https://github.com/RaphaEnterprises-AI/argus-backend/commit/cb448b1c881dcd56a431603a06c24801f53ca583))
* **webhooks:** correct regex order in generate_fingerprint for UUID normalization ([5754f18](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5754f185e9eac06aeb366d006a623092ae67a64f))


### Documentation

* **architecture:** update with v2.8.0 audit findings ([ea1f0b0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ea1f0b05d755462d46888c93b6c0b37b7b14bc69))

## [2.8.0](https://github.com/RaphaEnterprises-AI/argus-backend/compare/v2.7.0...v2.8.0) (2026-01-17)


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
* **api:** Add AI-driven infrastructure optimizer ([eb95295](https://github.com/RaphaEnterprises-AI/argus-backend/commit/eb95295255e8e4951d84d51917c102900e95111f))
* **api:** register MCP sessions router in FastAPI app ([18eee40](https://github.com/RaphaEnterprises-AI/argus-backend/commit/18eee40ebcf764ca0eb81eeadbe2f57967a60dfa))
* **browser-pool,worker:** add /extract endpoint and fix R2 storage serving ([3e980fa](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3e980fa5f46a8dda130a0cd6b25161c2942f23dc))
* **browser-pool:** add Kubernetes browser pool with production-grade JWT auth ([1137be6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1137be6ab129fd60bfce5bf7334d165a6fd9fe2f))
* **browser:** add Python client for browser pool with JWT auth ([60d109c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/60d109c9c879e9a41df458b5ef248ed1d4a2b8b3))
* **chat:** add structured test preview response for createTest tool ([e1ff7dc](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e1ff7dc4bd947361cfc55a2fa44db95278277989))
* **cloudflare-worker:** add Vultr browser pool as primary backend ([e726176](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e726176655c2eea847a4a4ba44f777c72980ee8a))
* **dashboard:** add frontend with performance and UUID validation fixes ([320caf5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/320caf518c23d47964f83641d6daa371f312d7f9))
* **dashboard:** add Vercel Speed Insights for performance monitoring ([f32d930](https://github.com/RaphaEnterprises-AI/argus-backend/commit/f32d930aa8c0d0b54ce28703866ffa707ce691a5))
* deploy Cloudflare Worker with queue handler and KV cache ([fd48087](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fd48087b63ef4004478039df44df023e67ff9e08))
* enterprise security layer for SOC2 compliance ([e00d970](https://github.com/RaphaEnterprises-AI/argus-backend/commit/e00d9701b715607542d605b43e5e22fc5a813d43))
* fetch user email from Clerk Backend API when not in JWT ([41e460b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/41e460b44c627f10c8e717db5c59f9bcfc92ac4b))
* implement complete user management system ([4437660](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4437660325672766410cec26baebe800cf4d5fd5))
* implement full LangGraph 1.0 feature suite for production-ready orchestration ([1c2c003](https://github.com/RaphaEnterprises-AI/argus-backend/commit/1c2c003f6122491c827b220cf6ceb2709466b104))
* **infra:** Add KEDA autoscaling for Selenium Grid browser pool ([ea09c3d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ea09c3d3cb90929a6d4411398e027e7d9af230d2))
* **llm:** Simplify to OpenRouter as single LLM provider ([5775534](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5775534b8bef2ffb8813faa45e49b012f6b820f6))
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
* **api:** comprehensive bug fixes from E2E testing - 19 bugs fixed ([2fa23c5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2fa23c5dab9eff782a084163ab02ea77ad42cebb))
* **api:** fix remaining bugs in mcp_sessions.py ([b314cc2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b314cc2b67dddb8d409ca01871c3733848888bd4))
* **api:** use dict access for user object in mcp_sessions ([9ee520d](https://github.com/RaphaEnterprises-AI/argus-backend/commit/9ee520d5fc402622c6f3404cbabd0e971eada233))
* **auth:** handle SecretStr in JWT token generation ([4d329c0](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4d329c0a8187966c22b5c5b54e98b2b399a54096))
* auto-detect Clerk JWKS URL from token issuer claim ([b85ceff](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b85ceff7997c398a124047d6e5a8b8857b1df700))
* backend chat improvements - artifacts API, cancel, tool serialization ([33ee271](https://github.com/RaphaEnterprises-AI/argus-backend/commit/33ee271ae39eede6086861b76f71fd372afcd66f))
* **browser-pool:** prevent browser leaks by releasing browsers on error ([3b4dfc8](https://github.com/RaphaEnterprises-AI/argus-backend/commit/3b4dfc892080c6d6f94e20e65e53e78917b5acba))
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
* critical security and data integrity fixes ([fd75fd2](https://github.com/RaphaEnterprises-AI/argus-backend/commit/fd75fd24d843819e131a7fea05fcc44fba6014a3))
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
* **migration:** fix FK reference to non-existent users table ([b4f5134](https://github.com/RaphaEnterprises-AI/argus-backend/commit/b4f51347819490dddcb20f0d7cd6ecc74742aabc))
* **migrations:** rename RLS migration to fix timestamp collision and add conditional checks ([c06862c](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c06862c7c04986876be2d30842d92dd2d3859988))
* **migrations:** split RPC functions into separate migrations for Supabase compatibility ([c577fca](https://github.com/RaphaEnterprises-AI/argus-backend/commit/c577fca3fa00053d2853b7f86990416973550ad3))
* null safety in vectorize and seed default project ([bd293a7](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bd293a78bb39242845c3cc131a3ca1e913109936))
* pass text as array to Workers AI embedding API ([77e5d7a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/77e5d7ac0b3ceba4f6a515b0cce7dc9a8598fa9f))
* remove unused CognitiveEngine import in predictive_quality ([5c893b3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5c893b3edad078ad72afebdc5da5fa14c346daf3))
* rename security audit migration to fix duplicate timestamp ([598b069](https://github.com/RaphaEnterprises-AI/argus-backend/commit/598b06916cbe71cdaf37148f17f6dc03008ef868))
* resolve 'default' org_id to user's actual organization ([8fb39b4](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8fb39b452e9ca623a509ee72d5f9eb64a9801314))
* resolve E2E API bugs and infrastructure issues ([2dc871f](https://github.com/RaphaEnterprises-AI/argus-backend/commit/2dc871f2c3ec845e9bb9135b2e849be26855d1f2))
* resolve production bugs from Railway testing ([bdfe8f5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/bdfe8f5e4ab61abe0e061a2e228f9f44a41ba1dc))
* resolve supabase migration schema drift issues ([4e6d6f3](https://github.com/RaphaEnterprises-AI/argus-backend/commit/4e6d6f384598eeddaf1acee640b34bf4dcb74b2c))
* **router:** use correct model key 'llama-small' and add OpenRouter support ([918c51a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/918c51ac017af0fa87feeb0db13775b03902051d))
* **security:** comprehensive security remediation - N+1 queries, input validation, size limits ([69007f5](https://github.com/RaphaEnterprises-AI/argus-backend/commit/69007f503201d9a23a041efcf342a93b9c9c1183))
* **security:** comprehensive SOC2 security audit fixes ([43a8311](https://github.com/RaphaEnterprises-AI/argus-backend/commit/43a8311def50edbb3d3506e2910f6ab166458ace))
* simplify Dockerfile for Railway deployment ([5a58c3b](https://github.com/RaphaEnterprises-AI/argus-backend/commit/5a58c3be961ec3e6ed75b26ed65da391cdbec40d))
* simplify Vectorize v2 query payload ([64e4d8a](https://github.com/RaphaEnterprises-AI/argus-backend/commit/64e4d8a383e3869c3b249ecbe85cdf92547c8570))
* skip org membership verification for API key auth ([a138fb6](https://github.com/RaphaEnterprises-AI/argus-backend/commit/a138fb67a0f4ce5ac82a0b6a90ea1a0455060c57))
* stream tool results in AI SDK format (a: prefix) ([ed3bd65](https://github.com/RaphaEnterprises-AI/argus-backend/commit/ed3bd657b27cfd1a88f1a229029563a6452b79c6))
* support API key authentication for organization access ([468c4ae](https://github.com/RaphaEnterprises-AI/argus-backend/commit/468c4aed04fb139da91823191ab6c795bb8427f1))
* support comma-separated CORS_ALLOWED_ORIGINS environment variable ([8cb55be](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8cb55bed5667c3d66199c8d6af579848e1ac38e8))
* sync Clerk organizations to Supabase on first access ([8074b69](https://github.com/RaphaEnterprises-AI/argus-backend/commit/8074b69c25fcabb4ebaae570749afc4e11e4fd29))
* **tests:** preserve real API keys for integration tests ([d651b14](https://github.com/RaphaEnterprises-AI/argus-backend/commit/d651b14fc377bd88701370e98412cda531076619))
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

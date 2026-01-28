-- Migration: Add Support for Additional AI Providers
-- This migration extends the user_provider_keys table to support all 16 providers
-- that were added in the multi-provider AI Hub implementation.
--
-- New providers added:
-- - openrouter (aggregator with 300+ models)
-- - azure_openai (enterprise Azure deployment)
-- - aws_bedrock (enterprise AWS deployment)
-- - vertex_ai (enterprise Google Cloud deployment)
-- - deepseek (DeepSeek V3 and R1 reasoning)
-- - mistral (Mistral AI models)
-- - fireworks (fast open-source inference)
-- - perplexity (search-augmented AI)
-- - cohere (enterprise RAG)
-- - xai (Grok models)
-- - cerebras (ultra-fast inference)

-- ============================================================================
-- UPDATE PROVIDER CHECK CONSTRAINT
-- ============================================================================

-- Drop the existing CHECK constraint
ALTER TABLE user_provider_keys
DROP CONSTRAINT IF EXISTS user_provider_keys_provider_check;

-- Add new CHECK constraint with all 16 supported providers
ALTER TABLE user_provider_keys
ADD CONSTRAINT user_provider_keys_provider_check
CHECK (provider IN (
    -- Original providers
    'anthropic',
    'openai',
    'google',
    'groq',
    'together',
    -- Aggregator
    'openrouter',
    -- Enterprise cloud providers
    'azure_openai',
    'aws_bedrock',
    'vertex_ai',
    -- Additional direct providers
    'deepseek',
    'mistral',
    'fireworks',
    'perplexity',
    'cohere',
    'xai',
    'cerebras'
));

-- ============================================================================
-- ADD PROVIDER METADATA TABLE (Optional: for UI display)
-- ============================================================================

-- This table stores metadata about providers for the dashboard
-- The backend providers have hardcoded info, but this allows for dynamic updates
CREATE TABLE IF NOT EXISTS ai_provider_metadata (
    provider TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    website TEXT,
    key_url TEXT,
    description TEXT,
    is_aggregator BOOLEAN DEFAULT FALSE,
    is_enterprise BOOLEAN DEFAULT FALSE,
    supports_streaming BOOLEAN DEFAULT TRUE,
    supports_tools BOOLEAN DEFAULT TRUE,
    supports_vision BOOLEAN DEFAULT FALSE,
    supports_computer_use BOOLEAN DEFAULT FALSE,
    models_count INTEGER DEFAULT 0,
    icon_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert provider metadata
INSERT INTO ai_provider_metadata (
    provider, display_name, website, key_url, description,
    is_aggregator, is_enterprise, supports_vision, supports_computer_use, models_count
) VALUES
    ('anthropic', 'Anthropic', 'https://anthropic.com', 'https://console.anthropic.com/account/keys',
     'Claude family - state-of-the-art AI from Anthropic', FALSE, FALSE, TRUE, TRUE, 4),
    ('openai', 'OpenAI', 'https://openai.com', 'https://platform.openai.com/api-keys',
     'GPT-4, o1, and DALL-E models', FALSE, FALSE, TRUE, FALSE, 10),
    ('google', 'Google AI', 'https://ai.google.dev', 'https://aistudio.google.com/app/apikey',
     'Gemini models from Google', FALSE, FALSE, TRUE, FALSE, 6),
    ('groq', 'Groq', 'https://groq.com', 'https://console.groq.com/keys',
     'Ultra-fast inference on LPU hardware', FALSE, FALSE, FALSE, FALSE, 5),
    ('together', 'Together AI', 'https://together.ai', 'https://api.together.xyz/settings/api-keys',
     'Open-source models with fast inference', FALSE, FALSE, TRUE, FALSE, 50),
    ('openrouter', 'OpenRouter', 'https://openrouter.ai', 'https://openrouter.ai/keys',
     'Unified API for 300+ models from all providers', TRUE, FALSE, TRUE, TRUE, 300),
    ('azure_openai', 'Azure OpenAI', 'https://azure.microsoft.com/products/ai-services/openai-service',
     'https://portal.azure.com', 'Enterprise OpenAI deployment on Azure', FALSE, TRUE, TRUE, FALSE, 8),
    ('aws_bedrock', 'AWS Bedrock', 'https://aws.amazon.com/bedrock',
     'https://console.aws.amazon.com/iam', 'Foundation models on AWS with IAM', FALSE, TRUE, TRUE, TRUE, 20),
    ('vertex_ai', 'Google Vertex AI', 'https://cloud.google.com/vertex-ai',
     'https://console.cloud.google.com/iam-admin', 'Gemini on GCP with enterprise features', FALSE, TRUE, TRUE, TRUE, 6),
    ('deepseek', 'DeepSeek', 'https://deepseek.com', 'https://platform.deepseek.com/api_keys',
     'DeepSeek V3 and R1 reasoning models', FALSE, FALSE, FALSE, FALSE, 2),
    ('mistral', 'Mistral AI', 'https://mistral.ai', 'https://console.mistral.ai/api-keys',
     'European AI lab with efficient models', FALSE, FALSE, TRUE, FALSE, 9),
    ('fireworks', 'Fireworks AI', 'https://fireworks.ai', 'https://fireworks.ai/account/api-keys',
     'Fast inference for open-source models', FALSE, FALSE, TRUE, FALSE, 15),
    ('perplexity', 'Perplexity AI', 'https://perplexity.ai', 'https://www.perplexity.ai/settings/api',
     'Search-augmented AI with citations', FALSE, FALSE, FALSE, FALSE, 4),
    ('cohere', 'Cohere', 'https://cohere.com', 'https://dashboard.cohere.com/api-keys',
     'Enterprise AI with strong RAG support', FALSE, FALSE, FALSE, FALSE, 4),
    ('xai', 'xAI', 'https://x.ai', 'https://console.x.ai',
     'Grok models with real-time data access', FALSE, FALSE, TRUE, FALSE, 3),
    ('cerebras', 'Cerebras', 'https://cerebras.ai', 'https://cloud.cerebras.ai/platform',
     'Ultra-fast inference on Wafer-Scale Engine', FALSE, FALSE, FALSE, FALSE, 4)
ON CONFLICT (provider) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    website = EXCLUDED.website,
    key_url = EXCLUDED.key_url,
    description = EXCLUDED.description,
    is_aggregator = EXCLUDED.is_aggregator,
    is_enterprise = EXCLUDED.is_enterprise,
    supports_vision = EXCLUDED.supports_vision,
    supports_computer_use = EXCLUDED.supports_computer_use,
    models_count = EXCLUDED.models_count,
    updated_at = NOW();

-- ============================================================================
-- ADD INDEXES FOR NEW QUERY PATTERNS
-- ============================================================================

-- Index for provider type queries (aggregators, enterprise)
CREATE INDEX IF NOT EXISTS idx_ai_provider_metadata_aggregator ON ai_provider_metadata(is_aggregator);
CREATE INDEX IF NOT EXISTS idx_ai_provider_metadata_enterprise ON ai_provider_metadata(is_enterprise);

-- ============================================================================
-- ADD RLS POLICIES
-- ============================================================================

ALTER TABLE ai_provider_metadata ENABLE ROW LEVEL SECURITY;

-- Everyone can read provider metadata
DROP POLICY IF EXISTS "Anyone can read provider metadata" ON ai_provider_metadata;
CREATE POLICY "Anyone can read provider metadata" ON ai_provider_metadata
    FOR SELECT USING (TRUE);

-- Only admins can update metadata
DROP POLICY IF EXISTS "Admins can update provider metadata" ON ai_provider_metadata;
CREATE POLICY "Admins can update provider metadata" ON ai_provider_metadata
    FOR UPDATE USING (current_setting('role', true) = 'service_role');

DROP POLICY IF EXISTS "Service role has full access to provider metadata" ON ai_provider_metadata;
CREATE POLICY "Service role has full access to provider metadata" ON ai_provider_metadata
    FOR ALL USING (current_setting('role', true) = 'service_role');

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

GRANT SELECT ON ai_provider_metadata TO authenticated;
GRANT SELECT ON ai_provider_metadata TO anon;

-- ============================================================================
-- UPDATE AI PREFERENCES DEFAULTS
-- ============================================================================

-- Update the default ai_preferences to include new routing settings
COMMENT ON COLUMN user_profiles.ai_preferences IS 'User AI preferences including:
- default_provider: Primary provider for requests
- default_model: Default model to use
- cost_limit_per_day: Daily spending limit in USD
- cost_limit_per_message: Per-message spending limit
- use_platform_key_fallback: Whether to fallback to platform keys
- show_token_costs: Display token costs in UI
- show_model_in_chat: Show model used in chat
- preferred_models_by_task: Task-specific model preferences (TaskType -> model ID)
- routing_mode: "auto" | "manual" - how models are selected
- auto_routing_preference: "cost" | "speed" | "quality" - priority for auto routing
';

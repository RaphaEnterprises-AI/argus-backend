-- Add DEK Reference for Cloudflare Key Vault Integration
-- This migration adds support for envelope encryption where:
-- - DEK (Data Encryption Key) is stored in Cloudflare KV
-- - Encrypted API key is stored in Supabase
-- - Backend never sees plaintext API keys

-- Add dek_reference column to track which Cloudflare KV key holds the DEK
ALTER TABLE user_provider_keys
ADD COLUMN IF NOT EXISTS dek_reference TEXT;

-- Add dek_version for key rotation tracking
ALTER TABLE user_provider_keys
ADD COLUMN IF NOT EXISTS dek_version INTEGER DEFAULT 1;

-- Add encrypted_at timestamp
ALTER TABLE user_provider_keys
ADD COLUMN IF NOT EXISTS encrypted_at TIMESTAMPTZ;

-- Add encryption_method to distinguish between local and Cloudflare encryption
-- 'local' = AES-256-GCM encrypted locally (legacy)
-- 'cloudflare' = Envelope encryption via Cloudflare Key Vault
ALTER TABLE user_provider_keys
ADD COLUMN IF NOT EXISTS encryption_method TEXT DEFAULT 'local'
CHECK (encryption_method IN ('local', 'cloudflare'));

-- Create index for dek_reference lookups (for key rotation)
CREATE INDEX IF NOT EXISTS idx_user_provider_keys_dek_ref
ON user_provider_keys(dek_reference)
WHERE dek_reference IS NOT NULL;

-- Add comment explaining the columns
COMMENT ON COLUMN user_provider_keys.dek_reference IS
'Reference to DEK in Cloudflare KV (e.g., "dek:user_123:anthropic:v1"). Null for legacy local encryption.';

COMMENT ON COLUMN user_provider_keys.dek_version IS
'Version of the DEK for rotation tracking. Increments on each key rotation.';

COMMENT ON COLUMN user_provider_keys.encrypted_at IS
'Timestamp when the key was last encrypted/rotated.';

COMMENT ON COLUMN user_provider_keys.encryption_method IS
'Encryption method: "local" for AES-256-GCM in backend, "cloudflare" for envelope encryption via Key Vault.';

-- Update existing rows to have encryption_method = 'local'
UPDATE user_provider_keys
SET encryption_method = 'local'
WHERE encryption_method IS NULL;

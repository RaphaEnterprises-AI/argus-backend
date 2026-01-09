-- Organization enhancements
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS logo_url TEXT;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS domain TEXT;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS sso_enabled BOOLEAN DEFAULT false;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS sso_config JSONB;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS allowed_email_domains TEXT[];
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS require_2fa BOOLEAN DEFAULT false;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS data_retention_days INTEGER DEFAULT 90;
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS created_by TEXT;

CREATE INDEX IF NOT EXISTS idx_organizations_domain ON organizations(domain) WHERE domain IS NOT NULL;

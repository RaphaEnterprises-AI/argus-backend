-- Migration: Integration Platforms Catalog
-- Description: Creates a table for storing integration platform metadata
--              enabling dynamic platform registration without code changes.
-- Created: 2026-01-29

-- ============================================================================
-- Integration Platforms Catalog Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS integration_platforms (
    -- Primary key
    slug VARCHAR(50) PRIMARY KEY,

    -- Display information
    name VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    icon_url TEXT,
    website_url TEXT,
    docs_url TEXT,

    -- Categorization
    category VARCHAR(50) NOT NULL CHECK (category IN (
        'source_code',
        'issue_tracking',
        'chat',
        'cicd',
        'deployment',
        'observability',
        'session_replay',
        'analytics',
        'incident_management',
        'feature_flags',
        'testing',
        'webhooks',
        'ai_agents',
        'database'
    )),

    -- Authentication
    auth_type VARCHAR(30) NOT NULL CHECK (auth_type IN (
        'api_key',
        'oauth2',
        'bearer_token',
        'basic_auth',
        'webhook',
        'service_account',
        'app_password',
        'mcp'
    )),

    -- Field definitions (JSON arrays of FieldDefinition objects)
    required_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
    optional_fields JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- OAuth configuration (for oauth2 auth_type)
    oauth_config JSONB,

    -- Features and capabilities
    features TEXT[] NOT NULL DEFAULT '{}',
    sync_resources TEXT[] NOT NULL DEFAULT '{}',
    supports_webhooks BOOLEAN NOT NULL DEFAULT false,
    supports_sync BOOLEAN NOT NULL DEFAULT true,

    -- Status
    enabled BOOLEAN NOT NULL DEFAULT true,
    coming_soon BOOLEAN NOT NULL DEFAULT false,
    beta BOOLEAN NOT NULL DEFAULT false,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for category filtering
CREATE INDEX IF NOT EXISTS idx_integration_platforms_category ON integration_platforms(category);

-- Index for enabled platforms
CREATE INDEX IF NOT EXISTS idx_integration_platforms_enabled ON integration_platforms(enabled) WHERE enabled = true;

-- ============================================================================
-- Seed Data: All 57 Integration Platforms
-- ============================================================================

INSERT INTO integration_platforms (slug, name, description, category, auth_type, features, required_fields, optional_fields, supports_webhooks, docs_url)
VALUES
-- Source Code (4)
('github', 'GitHub', 'Source code hosting, pull requests, and CI/CD', 'source_code', 'oauth2',
 ARRAY['repos', 'pull_requests', 'webhooks', 'actions'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "org_slug", "label": "Organization", "type": "text", "required": false}]'::jsonb,
 true, 'https://docs.github.com/en/rest'),

('gitlab', 'GitLab', 'DevOps platform with source control and CI/CD', 'source_code', 'oauth2',
 ARRAY['repos', 'merge_requests', 'pipelines', 'webhooks'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "instance_url", "label": "Instance URL", "type": "url", "required": false, "placeholder": "https://gitlab.com"}]'::jsonb,
 true, 'https://docs.gitlab.com/ee/api/'),

('bitbucket', 'Bitbucket', 'Git repository management by Atlassian', 'source_code', 'app_password',
 ARRAY['repos', 'pull_requests', 'pipelines'],
 '[{"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "password", "label": "App Password", "type": "password", "required": true}]'::jsonb,
 '[{"name": "workspace_id", "label": "Workspace", "type": "text", "required": false}]'::jsonb,
 true, 'https://developer.atlassian.com/cloud/bitbucket/rest/'),

('azure_devops', 'Azure DevOps', 'Microsoft DevOps platform', 'source_code', 'bearer_token',
 ARRAY['repos', 'pull_requests', 'pipelines', 'boards'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Organization URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://learn.microsoft.com/en-us/rest/api/azure/devops/'),

-- Issue Tracking (9)
('jira', 'Jira', 'Issue tracking and project management', 'issue_tracking', 'api_key',
 ARRAY['issues', 'projects', 'sprints', 'webhooks'],
 '[{"name": "email", "label": "Email", "type": "email", "required": true}, {"name": "api_key", "label": "API Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Instance URL", "type": "url", "required": true}]'::jsonb,
 '[{"name": "project_key", "label": "Default Project", "type": "text", "required": false}]'::jsonb,
 true, 'https://developer.atlassian.com/cloud/jira/platform/rest/v3/'),

('linear', 'Linear', 'Modern issue tracking for software teams', 'issue_tracking', 'oauth2',
 ARRAY['issues', 'projects', 'cycles', 'webhooks'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "team_id", "label": "Default Team", "type": "text", "required": false}]'::jsonb,
 true, 'https://developers.linear.app/docs/graphql/working-with-the-graphql-api'),

('asana', 'Asana', 'Work management and collaboration', 'issue_tracking', 'oauth2',
 ARRAY['tasks', 'projects', 'portfolios'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "workspace_id", "label": "Workspace ID", "type": "text", "required": false}]'::jsonb,
 true, 'https://developers.asana.com/docs'),

('trello', 'Trello', 'Visual project management boards', 'issue_tracking', 'api_key',
 ARRAY['cards', 'boards', 'lists'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}, {"name": "api_secret", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developer.atlassian.com/cloud/trello/rest/'),

('clickup', 'ClickUp', 'All-in-one productivity platform', 'issue_tracking', 'oauth2',
 ARRAY['tasks', 'spaces', 'folders', 'lists'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "workspace_id", "label": "Workspace ID", "type": "text", "required": false}]'::jsonb,
 true, 'https://clickup.com/api'),

('shortcut', 'Shortcut', 'Project management for software teams', 'issue_tracking', 'api_key',
 ARRAY['stories', 'epics', 'iterations'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developer.shortcut.com/api/rest/v3'),

('monday', 'Monday.com', 'Work OS for teams', 'issue_tracking', 'api_key',
 ARRAY['boards', 'items', 'groups'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developer.monday.com/api-reference/docs'),

('notion', 'Notion', 'All-in-one workspace', 'issue_tracking', 'oauth2',
 ARRAY['databases', 'pages', 'blocks'],
 '[{"name": "access_token", "label": "Integration Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 false, 'https://developers.notion.com/'),

('height', 'Height', 'Autonomous project collaboration', 'issue_tracking', 'api_key',
 ARRAY['tasks', 'lists', 'projects'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://height.notion.site/API-documentation'),

-- Chat & Notifications (4)
('slack', 'Slack', 'Team communication and notifications', 'chat', 'oauth2',
 ARRAY['messages', 'channels', 'webhooks', 'notifications'],
 '[{"name": "access_token", "label": "Bot Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "channel", "label": "Default Channel", "type": "text", "required": false}]'::jsonb,
 true, 'https://api.slack.com/'),

('discord', 'Discord', 'Community platform with webhooks', 'chat', 'webhook',
 ARRAY['webhooks', 'notifications'],
 '[{"name": "webhook_url", "label": "Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[{"name": "bot_name", "label": "Bot Name", "type": "text", "required": false}]'::jsonb,
 true, 'https://discord.com/developers/docs/resources/webhook'),

('teams', 'Microsoft Teams', 'Microsoft team collaboration', 'chat', 'oauth2',
 ARRAY['messages', 'channels', 'webhooks'],
 '[{"name": "webhook_url", "label": "Incoming Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/'),

('google_chat', 'Google Chat', 'Google Workspace messaging', 'chat', 'service_account',
 ARRAY['messages', 'spaces'],
 '[{"name": "webhook_url", "label": "Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developers.google.com/chat/api'),

-- CI/CD (7)
('github_actions', 'GitHub Actions', 'GitHub native CI/CD', 'cicd', 'bearer_token',
 ARRAY['workflows', 'runs', 'artifacts'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "org_slug", "label": "Organization", "type": "text", "required": false}]'::jsonb,
 true, 'https://docs.github.com/en/rest/actions'),

('gitlab_ci', 'GitLab CI/CD', 'GitLab native CI/CD', 'cicd', 'bearer_token',
 ARRAY['pipelines', 'jobs', 'artifacts'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "instance_url", "label": "Instance URL", "type": "url", "required": false}]'::jsonb,
 true, 'https://docs.gitlab.com/ee/api/pipelines.html'),

('jenkins', 'Jenkins', 'Open source automation server', 'cicd', 'basic_auth',
 ARRAY['jobs', 'builds', 'artifacts'],
 '[{"name": "instance_url", "label": "Jenkins URL", "type": "url", "required": true}, {"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.jenkins.io/doc/book/using/remote-access-api/'),

('circleci', 'CircleCI', 'Continuous integration and delivery', 'cicd', 'api_key',
 ARRAY['pipelines', 'workflows', 'jobs'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "org_slug", "label": "Organization", "type": "text", "required": false}]'::jsonb,
 true, 'https://circleci.com/docs/api/v2/'),

('azure_pipelines', 'Azure Pipelines', 'Microsoft CI/CD service', 'cicd', 'bearer_token',
 ARRAY['pipelines', 'builds', 'releases'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Organization URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://learn.microsoft.com/en-us/rest/api/azure/devops/pipelines/'),

('bitbucket_pipelines', 'Bitbucket Pipelines', 'Bitbucket native CI/CD', 'cicd', 'app_password',
 ARRAY['pipelines', 'steps', 'artifacts'],
 '[{"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "password", "label": "App Password", "type": "password", "required": true}]'::jsonb,
 '[{"name": "workspace_id", "label": "Workspace", "type": "text", "required": false}]'::jsonb,
 true, 'https://developer.atlassian.com/cloud/bitbucket/rest/api-group-pipelines/'),

('teamcity', 'TeamCity', 'JetBrains CI/CD server', 'cicd', 'bearer_token',
 ARRAY['builds', 'projects', 'agents'],
 '[{"name": "instance_url", "label": "TeamCity URL", "type": "url", "required": true}, {"name": "access_token", "label": "Access Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.jetbrains.com/help/teamcity/rest-api.html'),

-- Deployment (7)
('vercel', 'Vercel', 'Frontend deployment platform', 'deployment', 'bearer_token',
 ARRAY['deployments', 'projects', 'domains'],
 '[{"name": "access_token", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "team_id", "label": "Team ID", "type": "text", "required": false}]'::jsonb,
 true, 'https://vercel.com/docs/rest-api'),

('netlify', 'Netlify', 'Web deployment and hosting', 'deployment', 'bearer_token',
 ARRAY['deploys', 'sites', 'builds'],
 '[{"name": "access_token", "label": "Personal Access Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.netlify.com/api/get-started/'),

('heroku', 'Heroku', 'Cloud application platform', 'deployment', 'bearer_token',
 ARRAY['apps', 'dynos', 'releases'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://devcenter.heroku.com/articles/platform-api-reference'),

('railway', 'Railway', 'Infrastructure platform', 'deployment', 'bearer_token',
 ARRAY['deployments', 'services', 'environments'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.railway.app/reference/public-api'),

('render', 'Render', 'Cloud hosting platform', 'deployment', 'bearer_token',
 ARRAY['services', 'deploys', 'logs'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://api-docs.render.com/reference/introduction'),

('flyio', 'Fly.io', 'Global application platform', 'deployment', 'bearer_token',
 ARRAY['apps', 'machines', 'volumes'],
 '[{"name": "access_token", "label": "Access Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://fly.io/docs/machines/api/'),

('aws', 'AWS', 'Amazon Web Services', 'deployment', 'api_key',
 ARRAY['cloudwatch', 'ecs', 'lambda', 's3'],
 '[{"name": "api_key", "label": "Access Key ID", "type": "password", "required": true}, {"name": "api_secret", "label": "Secret Access Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "site", "label": "Region", "type": "text", "required": false, "placeholder": "us-east-1"}]'::jsonb,
 true, 'https://docs.aws.amazon.com/'),

-- Observability (6)
('sentry', 'Sentry', 'Error tracking and performance monitoring', 'observability', 'bearer_token',
 ARRAY['issues', 'events', 'releases', 'webhooks'],
 '[{"name": "access_token", "label": "Auth Token", "type": "password", "required": true}, {"name": "org_slug", "label": "Organization Slug", "type": "text", "required": true}, {"name": "project_slug", "label": "Project Slug", "type": "text", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.sentry.io/api/'),

('datadog', 'Datadog', 'Infrastructure and application monitoring', 'observability', 'api_key',
 ARRAY['metrics', 'events', 'logs', 'monitors'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}, {"name": "api_secret", "label": "Application Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "site", "label": "Site", "type": "select", "required": false, "options": [{"value": "datadoghq.com", "label": "US1"}, {"value": "us3.datadoghq.com", "label": "US3"}, {"value": "datadoghq.eu", "label": "EU"}]}]'::jsonb,
 true, 'https://docs.datadoghq.com/api/'),

('newrelic', 'New Relic', 'Observability platform', 'observability', 'api_key',
 ARRAY['apm', 'browser', 'logs', 'alerts'],
 '[{"name": "api_key", "label": "User API Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "site", "label": "Region", "type": "select", "required": false, "options": [{"value": "US", "label": "US"}, {"value": "EU", "label": "EU"}]}]'::jsonb,
 true, 'https://docs.newrelic.com/docs/apis/'),

('grafana', 'Grafana', 'Observability and visualization', 'observability', 'api_key',
 ARRAY['dashboards', 'alerts', 'datasources'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}, {"name": "instance_url", "label": "Grafana URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://grafana.com/docs/grafana/latest/developers/http_api/'),

('splunk', 'Splunk', 'Data platform for security and observability', 'observability', 'bearer_token',
 ARRAY['search', 'alerts', 'dashboards'],
 '[{"name": "access_token", "label": "Authentication Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Splunk URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.splunk.com/Documentation/Splunk/latest/RESTREF/RESTprolog'),

('dynatrace', 'Dynatrace', 'Software intelligence platform', 'observability', 'api_key',
 ARRAY['problems', 'metrics', 'events'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Environment URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.dynatrace.com/support/help/dynatrace-api'),

-- Session Replay (5)
('logrocket', 'LogRocket', 'Session replay and monitoring', 'session_replay', 'api_key',
 ARRAY['sessions', 'issues', 'users'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.logrocket.com/reference/introduction'),

('fullstory', 'FullStory', 'Digital experience analytics', 'session_replay', 'api_key',
 ARRAY['sessions', 'events', 'users'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developer.fullstory.com/'),

('posthog', 'PostHog', 'Product analytics with session replay', 'session_replay', 'api_key',
 ARRAY['events', 'persons', 'recordings', 'feature_flags'],
 '[{"name": "api_key", "label": "Project API Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "instance_url", "label": "Instance URL", "type": "url", "required": false, "placeholder": "https://app.posthog.com"}]'::jsonb,
 true, 'https://posthog.com/docs/api'),

('hotjar', 'Hotjar', 'Heatmaps and session recordings', 'session_replay', 'api_key',
 ARRAY['recordings', 'heatmaps', 'surveys'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 false, 'https://help.hotjar.com/hc/en-us'),

('heap', 'Heap', 'Digital insights platform', 'session_replay', 'api_key',
 ARRAY['events', 'users', 'sessions'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 false, 'https://developers.heap.io/reference/getting-started'),

-- Analytics (3)
('amplitude', 'Amplitude', 'Product analytics platform', 'analytics', 'api_key',
 ARRAY['events', 'cohorts', 'charts'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}, {"name": "api_secret", "label": "Secret Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 false, 'https://www.docs.developers.amplitude.com/'),

('mixpanel', 'Mixpanel', 'Product analytics', 'analytics', 'api_key',
 ARRAY['events', 'profiles', 'reports'],
 '[{"name": "api_key", "label": "Project Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "api_secret", "label": "API Secret", "type": "password", "required": false}]'::jsonb,
 false, 'https://developer.mixpanel.com/reference/overview'),

('segment', 'Segment', 'Customer data platform', 'analytics', 'api_key',
 ARRAY['sources', 'destinations', 'tracking'],
 '[{"name": "api_key", "label": "Write Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://segment.com/docs/connections/sources/catalog/'),

-- Incident Management (5)
('pagerduty', 'PagerDuty', 'Incident management platform', 'incident_management', 'api_key',
 ARRAY['incidents', 'services', 'escalations', 'webhooks'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://developer.pagerduty.com/docs/'),

('opsgenie', 'Opsgenie', 'Incident management by Atlassian', 'incident_management', 'api_key',
 ARRAY['alerts', 'incidents', 'schedules'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.opsgenie.com/docs/api-overview'),

('incident_io', 'incident.io', 'Modern incident management', 'incident_management', 'api_key',
 ARRAY['incidents', 'actions', 'announcements'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://api-docs.incident.io/'),

('spike', 'Spike.sh', 'Incident management and alerting', 'incident_management', 'api_key',
 ARRAY['incidents', 'alerts', 'schedules'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.spike.sh/api-reference/'),

('victorops', 'Splunk On-Call', 'Incident management (formerly VictorOps)', 'incident_management', 'api_key',
 ARRAY['incidents', 'teams', 'rotations'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}, {"name": "api_secret", "label": "API ID", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://help.victorops.com/knowledge-base/victorops-restendpoint-integration/'),

-- Feature Flags (4)
('launchdarkly', 'LaunchDarkly', 'Feature management platform', 'feature_flags', 'api_key',
 ARRAY['flags', 'environments', 'segments'],
 '[{"name": "api_key", "label": "Access Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://apidocs.launchdarkly.com/'),

('split', 'Split', 'Feature delivery platform', 'feature_flags', 'api_key',
 ARRAY['splits', 'environments', 'segments'],
 '[{"name": "api_key", "label": "Admin API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.split.io/reference/introduction-to-split-api'),

('flagsmith', 'Flagsmith', 'Open source feature flags', 'feature_flags', 'api_key',
 ARRAY['flags', 'environments', 'identities'],
 '[{"name": "api_key", "label": "Server-side Environment Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "instance_url", "label": "API URL", "type": "url", "required": false, "placeholder": "https://edge.api.flagsmith.com/api/v1"}]'::jsonb,
 true, 'https://docs.flagsmith.com/clients/rest/'),

('unleash', 'Unleash', 'Open source feature management', 'feature_flags', 'api_key',
 ARRAY['features', 'strategies', 'environments'],
 '[{"name": "api_key", "label": "API Token", "type": "password", "required": true}, {"name": "instance_url", "label": "Unleash URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.getunleash.io/reference/api/unleash'),

-- Testing (7)
('browserstack', 'BrowserStack', 'Cross-browser testing platform', 'testing', 'basic_auth',
 ARRAY['automate', 'sessions', 'builds'],
 '[{"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "api_key", "label": "Access Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.browserstack.com/docs/automate/api-reference/selenium/introduction'),

('saucelabs', 'Sauce Labs', 'Cross-browser and mobile testing', 'testing', 'basic_auth',
 ARRAY['jobs', 'builds', 'tunnels'],
 '[{"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "api_key", "label": "Access Key", "type": "password", "required": true}]'::jsonb,
 '[{"name": "site", "label": "Data Center", "type": "select", "required": false, "options": [{"value": "us-west-1", "label": "US West"}, {"value": "eu-central-1", "label": "EU Central"}]}]'::jsonb,
 true, 'https://docs.saucelabs.com/dev/api/'),

('lambdatest', 'LambdaTest', 'Cross-browser testing cloud', 'testing', 'basic_auth',
 ARRAY['sessions', 'builds', 'tunnels'],
 '[{"name": "username", "label": "Username", "type": "text", "required": true}, {"name": "api_key", "label": "Access Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.lambdatest.com/support/api-doc/'),

('percy', 'Percy', 'Visual testing and review', 'testing', 'bearer_token',
 ARRAY['builds', 'snapshots', 'comparisons'],
 '[{"name": "access_token", "label": "Project Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.browserstack.com/docs/percy/integrate/api'),

('chromatic', 'Chromatic', 'Visual testing for Storybook', 'testing', 'bearer_token',
 ARRAY['builds', 'snapshots', 'reviews'],
 '[{"name": "access_token", "label": "Project Token", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://www.chromatic.com/docs/'),

('playwright', 'Playwright', 'Browser automation framework', 'testing', 'api_key',
 ARRAY['test_results', 'traces', 'artifacts'],
 '[{"name": "api_key", "label": "API Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 false, 'https://playwright.dev/docs/test-reporters'),

('cypress', 'Cypress', 'JavaScript E2E testing', 'testing', 'bearer_token',
 ARRAY['runs', 'instances', 'projects'],
 '[{"name": "access_token", "label": "Record Key", "type": "password", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.cypress.io/guides/cloud/introduction'),

-- Webhooks & Automation (3)
('webhook', 'Generic Webhook', 'Send events to any URL', 'webhooks', 'webhook',
 ARRAY['events', 'notifications'],
 '[{"name": "webhook_url", "label": "Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[{"name": "webhook_secret", "label": "Secret (for signature)", "type": "password", "required": false}]'::jsonb,
 true, NULL),

('zapier', 'Zapier', 'Workflow automation', 'webhooks', 'webhook',
 ARRAY['triggers', 'zaps'],
 '[{"name": "webhook_url", "label": "Zapier Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://zapier.com/apps/webhook/integrations'),

('n8n', 'n8n', 'Workflow automation platform', 'webhooks', 'webhook',
 ARRAY['triggers', 'workflows'],
 '[{"name": "webhook_url", "label": "n8n Webhook URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://docs.n8n.io/'),

-- AI Agents (2)
('cursor', 'Cursor', 'AI-powered code editor', 'ai_agents', 'mcp',
 ARRAY['code_context', 'suggestions'],
 '[]'::jsonb,
 '[]'::jsonb,
 false, 'https://www.cursor.com/'),

('claude_code', 'Claude Code', 'Anthropic CLI for Claude', 'ai_agents', 'mcp',
 ARRAY['code_context', 'tool_use'],
 '[]'::jsonb,
 '[]'::jsonb,
 false, 'https://claude.ai/claude-code'),

-- Database (2)
('supabase', 'Supabase', 'Open source Firebase alternative', 'database', 'api_key',
 ARRAY['database', 'auth', 'storage', 'realtime'],
 '[{"name": "api_key", "label": "Service Role Key", "type": "password", "required": true}, {"name": "instance_url", "label": "Project URL", "type": "url", "required": true}]'::jsonb,
 '[]'::jsonb,
 true, 'https://supabase.com/docs/reference/api/introduction'),

('turso', 'Turso', 'Edge database platform', 'database', 'bearer_token',
 ARRAY['databases', 'groups', 'tokens'],
 '[{"name": "access_token", "label": "API Token", "type": "password", "required": true}]'::jsonb,
 '[{"name": "org_slug", "label": "Organization", "type": "text", "required": false}]'::jsonb,
 false, 'https://docs.turso.tech/api-reference/')

ON CONFLICT (slug) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    auth_type = EXCLUDED.auth_type,
    features = EXCLUDED.features,
    required_fields = EXCLUDED.required_fields,
    optional_fields = EXCLUDED.optional_fields,
    supports_webhooks = EXCLUDED.supports_webhooks,
    docs_url = EXCLUDED.docs_url,
    updated_at = NOW();

-- ============================================================================
-- Trigger for updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_integration_platforms_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_integration_platforms_updated_at ON integration_platforms;
CREATE TRIGGER trigger_integration_platforms_updated_at
    BEFORE UPDATE ON integration_platforms
    FOR EACH ROW
    EXECUTE FUNCTION update_integration_platforms_updated_at();

-- ============================================================================
-- RLS Policies (public read, admin write)
-- ============================================================================

ALTER TABLE integration_platforms ENABLE ROW LEVEL SECURITY;

-- Anyone can read enabled platforms
CREATE POLICY "Anyone can read enabled integration platforms"
    ON integration_platforms
    FOR SELECT
    USING (enabled = true);

-- Only service role can modify
CREATE POLICY "Service role can manage integration platforms"
    ON integration_platforms
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Grant public read access
GRANT SELECT ON integration_platforms TO anon, authenticated;

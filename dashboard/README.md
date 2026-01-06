# Argus Dashboard

<div align="center">
  <img src="public/argus-logo.png" alt="Argus Logo" width="120" />
  <h3>AI-Powered E2E Testing Platform</h3>
  <p>Autonomous end-to-end testing with Claude AI integration</p>

  [![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/RaphaEnterprises-AI/argus/releases)
  [![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
  [![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
</div>

---

## Overview

Argus is a comprehensive AI-powered end-to-end testing platform that leverages Claude's intelligence to autonomously test, analyze, and validate web applications. This repository contains the frontend dashboard built with Next.js 15, React 19, and Supabase.

### Key Features

- **AI Chat Interface** - Natural language test creation and debugging with Claude AI
- **Test Runner** - Execute browser-based E2E tests with real-time feedback
- **Discovery Tool** - Automatically find testable surfaces in your application
- **Visual AI** - Visual regression testing with AI-powered diff analysis
- **AI Insights** - Pattern analysis and test optimization recommendations
- **Global Testing** - Cross-browser and cross-region validation
- **Quality Audit** - Code health checks and best practice verification
- **Intelligence Dashboard** - Quality metrics and trends visualization
- **Reports & Analytics** - Comprehensive test result reporting
- **Integrations** - Connect with Datadog, Sentry, PagerDuty, and more

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARGUS DASHBOARD                              │
│                    (Next.js 15 + React 19)                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│    Clerk      │           │   Supabase    │           │  Argus Core   │
│ Authentication│           │   Database    │           │   (Backend)   │
└───────────────┘           └───────────────┘           └───────────────┘
                                                                │
                                                    ┌───────────┴───────────┐
                                                    ▼                       ▼
                                            ┌───────────────┐       ┌───────────────┐
                                            │  Claude AI    │       │   Browser     │
                                            │  (Anthropic)  │       │   Automation  │
                                            └───────────────┘       └───────────────┘
```

## Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | Next.js 15 (App Router) |
| **UI Library** | React 19 |
| **Language** | TypeScript 5.6 |
| **Styling** | Tailwind CSS 3.4 |
| **Components** | Radix UI Primitives |
| **State Management** | Zustand, TanStack Query |
| **Database** | Supabase (PostgreSQL) |
| **Authentication** | Clerk |
| **AI Integration** | Anthropic Claude (via Vercel AI SDK) |
| **Charts** | Recharts |
| **Animations** | Framer Motion |

## Getting Started

### Prerequisites

- Node.js 18.17 or later
- npm, yarn, or pnpm
- Supabase account
- Clerk account
- Anthropic API key

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/samuelvinay91/argus.git
cd argus
```

2. **Install dependencies**

```bash
npm install
# or
yarn install
# or
pnpm install
```

3. **Set up environment variables**

Create a `.env.local` file in the root directory:

```env
# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Anthropic AI
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Analytics & Monitoring
NEXT_PUBLIC_POSTHOG_KEY=phc_...
SENTRY_DSN=https://...
```

4. **Set up the database**

Run the Supabase migrations:

```bash
npx supabase db push
```

Or apply migrations manually from `supabase/migrations/`.

5. **Start the development server**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Project Structure

```
argus/
├── app/                    # Next.js App Router pages
│   ├── (auth)/            # Authentication routes
│   ├── api/               # API routes
│   ├── discovery/         # Discovery tool
│   ├── global/            # Global testing
│   ├── insights/          # AI insights
│   ├── integrations/      # Third-party integrations
│   ├── intelligence/      # Quality metrics
│   ├── legal/             # Legal pages
│   ├── quality/           # Quality audit
│   ├── reports/           # Reports & analytics
│   ├── settings/          # User settings
│   ├── tests/             # Test runner
│   ├── visual/            # Visual AI testing
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home (Chat interface)
│
├── components/
│   ├── chat/              # Chat components
│   ├── layout/            # Layout components (Sidebar, Header)
│   ├── providers/         # Context providers
│   └── ui/                # Reusable UI components
│
├── lib/
│   ├── hooks/             # Custom React hooks
│   ├── supabase/          # Supabase client & types
│   ├── utils.ts           # Utility functions
│   └── version.ts         # Version management
│
├── public/                # Static assets
├── supabase/
│   └── migrations/        # Database migrations
│
├── middleware.ts          # Next.js middleware (auth)
├── tailwind.config.ts     # Tailwind configuration
└── tsconfig.json          # TypeScript configuration
```

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run version:patch` | Bump patch version (1.0.x) |
| `npm run version:minor` | Bump minor version (1.x.0) |
| `npm run version:major` | Bump major version (x.0.0) |

## Database Schema

The application uses Supabase with the following main tables:

| Table | Description |
|-------|-------------|
| `projects` | User projects/applications to test |
| `test_runs` | Test execution records |
| `test_results` | Individual test results |
| `test_specs` | Test specifications |
| `chat_conversations` | AI chat history |
| `chat_messages` | Individual chat messages |
| `discoveries` | Discovered testable surfaces |
| `visual_snapshots` | Visual regression snapshots |
| `integrations` | Third-party integration configs |

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Configure environment variables in Vercel dashboard
3. Deploy automatically on push to `main`

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/samuelvinay91/argus)

### Manual Deployment

```bash
# Build the application
npm run build

# Start the production server
npm run start
```

## Related Repositories

| Repository | Description |
|------------|-------------|
| [argus-core](https://github.com/samuelvinay91/argus-core) | Python backend with LangGraph orchestrator and AI agents |
| [e2e-testing-agent](https://github.com/samuelvinay91/e2e-testing-agent) | Cloudflare Worker for browser automation |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.heyargus.ai](https://docs.heyargus.ai)
- **Issues**: [GitHub Issues](https://github.com/samuelvinay91/argus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/samuelvinay91/argus/discussions)

---

<div align="center">
  <p>Built with AI by the Argus Team</p>
  <p>
    <a href="https://heyargus.ai">Website</a> •
    <a href="https://docs.heyargus.ai">Docs</a> •
    <a href="https://twitter.com/heyargus">Twitter</a>
  </p>
</div>

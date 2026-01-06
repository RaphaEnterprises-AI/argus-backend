# Contributing to Argus Dashboard

Thank you for your interest in contributing to Argus! This document provides guidelines and information about contributing to the frontend dashboard.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@heyargus.ai](mailto:conduct@heyargus.ai).

## Getting Started

### Prerequisites

- Node.js 18.17 or later
- npm, yarn, or pnpm
- Git
- A code editor (VS Code recommended)

### Setting Up Your Development Environment

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/argus.git
   cd argus
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/samuelvinay91/argus.git
   ```

4. **Install dependencies**

   ```bash
   npm install
   ```

5. **Set up environment variables**

   ```bash
   cp .env.example .env.local
   # Edit .env.local with your credentials
   ```

6. **Start the development server**

   ```bash
   npm run dev
   ```

## Development Workflow

### Branching Strategy

We use a simplified Git flow:

- `main` - Production-ready code, auto-deployed to Vercel
- `feature/*` - New features (e.g., `feature/visual-testing`)
- `fix/*` - Bug fixes (e.g., `fix/auth-redirect`)
- `docs/*` - Documentation updates
- `refactor/*` - Code refactoring

### Creating a Feature Branch

```bash
# Ensure you're on main and up to date
git checkout main
git pull upstream main

# Create your feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. Make your changes in small, logical commits
2. Write or update tests as needed
3. Ensure the build passes locally:
   ```bash
   npm run build
   npm run lint
   ```

### Keeping Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

## Pull Request Process

### Before Submitting

1. **Ensure your code builds**: `npm run build`
2. **Run linting**: `npm run lint`
3. **Update documentation** if you're changing APIs or adding features
4. **Write meaningful commit messages** (see below)

### Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to the [Argus repository](https://github.com/samuelvinay91/argus) and click "New Pull Request"

3. Select your branch and fill out the PR template

4. Request review from maintainers

### PR Title Format

Use conventional commit format for PR titles:

```
type(scope): description

Examples:
feat(chat): add message threading support
fix(auth): resolve redirect loop on logout
docs(readme): update installation instructions
refactor(ui): simplify button component
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, etc.) |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |

### Review Process

1. At least one maintainer must approve the PR
2. All CI checks must pass
3. No merge conflicts with `main`
4. Code coverage should not decrease

## Coding Standards

### TypeScript

- Use TypeScript for all new code
- Enable strict mode
- Prefer `interface` over `type` for object shapes
- Use explicit return types for functions

```typescript
// Good
interface UserProps {
  name: string;
  email: string;
}

function getUser(id: string): Promise<UserProps> {
  // ...
}

// Avoid
type UserProps = {
  name: string;
  email: string;
}

function getUser(id: string) {
  // ...
}
```

### React Components

- Use functional components with hooks
- Use named exports
- Keep components focused and small
- Extract logic into custom hooks

```typescript
// Good
export function TestCard({ test }: TestCardProps) {
  const { status, runTest } = useTest(test.id);

  return (
    <Card>
      <CardHeader>{test.name}</CardHeader>
      <CardContent>
        <StatusBadge status={status} />
      </CardContent>
    </Card>
  );
}
```

### File Organization

```
components/
  feature-name/
    index.ts           # Exports
    FeatureComponent.tsx
    FeatureComponent.test.tsx
    useFeature.ts      # Custom hook
    types.ts           # Types specific to this feature
```

### Styling

- Use Tailwind CSS utility classes
- Use the `cn()` utility for conditional classes
- Follow the existing design system tokens

```typescript
import { cn } from '@/lib/utils';

function Button({ variant, className, ...props }) {
  return (
    <button
      className={cn(
        'px-4 py-2 rounded-lg font-medium',
        variant === 'primary' && 'bg-primary text-primary-foreground',
        variant === 'outline' && 'border border-input bg-background',
        className
      )}
      {...props}
    />
  );
}
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body (optional)

footer (optional)
```

### Examples

```bash
# Feature
feat(discovery): add URL pattern filtering

# Bug fix
fix(tests): resolve timeout on long-running tests

Fixes #123

# Breaking change
feat(api)!: change response format for test results

BREAKING CHANGE: Test results now return an array instead of an object
```

### Guidelines

- Use present tense: "add feature" not "added feature"
- Use imperative mood: "move cursor to..." not "moves cursor to..."
- Keep the subject line under 72 characters
- Reference issues in the footer

## Issue Guidelines

### Bug Reports

When filing a bug report, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Numbered steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Browser, OS, Node version
6. **Screenshots**: If applicable

### Feature Requests

When requesting a feature, please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How would you like it to work?
3. **Alternatives**: Other approaches you've considered
4. **Additional Context**: Mockups, examples, etc.

## Questions?

- **Discord**: [Join our community](https://discord.gg/argus)
- **Discussions**: [GitHub Discussions](https://github.com/samuelvinay91/argus/discussions)
- **Email**: [support@heyargus.ai](mailto:support@heyargus.ai)

---

Thank you for contributing to Argus! Your efforts help make E2E testing better for everyone.

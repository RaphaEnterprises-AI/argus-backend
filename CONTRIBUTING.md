# Contributing to Argus E2E Testing Agent

Thank you for your interest in contributing to Argus! This document provides guidelines and instructions for contributing to the backend/core testing engine.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to uphold this standard.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (for Computer Use sandbox)
- Anthropic API key

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/argus-backend.git
cd argus-backend
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/samuelvinay91/argus-backend.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 4. Run Tests

```bash
pytest
```

## Project Structure

```
e2e-testing-agent/
├── src/
│   ├── agents/           # AI agents (code analyzer, test planner, etc.)
│   ├── api/              # FastAPI server
│   ├── browser/          # Browser automation (Playwright, Selenium)
│   ├── collaboration/    # Team features
│   ├── computer_use/     # Claude Computer Use integration
│   ├── config.py         # Configuration management
│   ├── core/             # Core testing logic
│   ├── infrastructure/   # Infrastructure code
│   ├── integrations/     # Third-party integrations
│   ├── main.py          # CLI entry point
│   ├── mcp/             # Model Context Protocol
│   ├── orchestrator/    # LangGraph workflow
│   ├── security/        # Security and sanitization
│   ├── tools/           # Testing tools
│   └── utils/           # Utilities
├── tests/               # Test suite
├── docs/                # Documentation
├── pyproject.toml       # Python project config
└── README.md
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Changes

- Write clear, concise code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/
```

### 4. Commit Changes

Use conventional commit messages:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Test changes
- `chore`: Maintenance

Examples:
```
feat(agents): add visual regression testing agent
fix(browser): resolve stale element reference issue
docs(api): update API endpoint documentation
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory mirroring `src/` structure
- Use descriptive test names: `test_should_do_something_when_condition()`
- Use pytest fixtures for common setup
- Mock external dependencies (API calls, file I/O)

Example:

```python
import pytest
from unittest.mock import patch
from src.agents.code_analyzer import CodeAnalyzer

@pytest.fixture
def analyzer():
    return CodeAnalyzer()

def test_should_analyze_python_file_successfully(analyzer):
    """Test that CodeAnalyzer can parse a Python file."""
    result = analyzer.analyze_file("test.py")
    assert result.language == "python"
    assert len(result.functions) > 0
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for all functions
- Docstrings for all public functions/classes
- Use `ruff` for formatting and linting

### Example

```python
from typing import List, Optional
from pydantic import BaseModel

class TestResult(BaseModel):
    """Represents the result of a single test execution.

    Attributes:
        name: Test name
        status: Pass/fail status
        duration: Execution time in seconds
        error: Error message if failed
    """
    name: str
    status: str
    duration: float
    error: Optional[str] = None

def run_test(test_name: str, timeout: int = 30) -> TestResult:
    """Execute a single test with timeout.

    Args:
        test_name: Name of the test to run
        timeout: Maximum execution time in seconds

    Returns:
        TestResult object with execution details

    Raises:
        TimeoutError: If test exceeds timeout
    """
    # Implementation
    pass
```

### Running Formatters

```bash
# Format code
ruff format src/ tests/

# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

## Submitting Changes

### Pull Request Process

1. **Update your branch with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues
   - Screenshots/videos for UI changes

4. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   How has this been tested?

   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code follows style guidelines
   - [ ] All tests passing
   ```

### Review Process

- All PRs require at least one review
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized
- CI must pass before merging

## Additional Guidelines

### Security

- Never commit API keys or secrets
- Use environment variables for configuration
- Follow security best practices
- Report security issues privately

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update docs/ for architectural changes
- Include code examples where helpful

### Performance

- Profile code for performance bottlenecks
- Optimize expensive operations
- Use async/await for I/O operations
- Monitor token usage for AI calls

### Error Handling

- Use descriptive error messages
- Log errors with appropriate context
- Handle edge cases gracefully
- Provide helpful error recovery suggestions

## Getting Help

- 💬 **Discussions**: Ask questions in GitHub Discussions
- 🐛 **Issues**: Report bugs in GitHub Issues
- 📧 **Email**: Contact maintainers at support@heyargus.ai
- 📚 **Docs**: Read the full documentation at docs/

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Argus! 🤖✨

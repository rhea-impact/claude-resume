# Contributing to resume-resume

Thanks for your interest in contributing.

## Quick start

```bash
git clone https://github.com/eidos-agi/resume-resume.git
cd resume-resume
pip install -e ".[dev]"
```

## Development

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

Run tests:

```bash
pytest
```

## For agent developers

If you're building tools that AI agents will use, pay special attention to:

1. **Tool descriptions** — Every `@tool` decorator must have a description that explains *when* to use it, not just *what* it does. An agent choosing between 20 tools needs clear differentiation.
2. **Parameter descriptions** — Every parameter needs a `description` field. Agents don't have UI tooltips — the description is all they get.
3. **Error messages** — When something fails, the error message must tell the agent what to do next. "Invalid input" is useless. "Expected ISO 8601 date string (e.g., 2026-03-22), got: 'yesterday'" is actionable.
4. **Typed everything** — Type hints on all public functions. Agents parse types to understand contracts.

## Releasing to PyPI

This repo uses **GitHub Actions trusted publishing** — no PyPI tokens, no `.pypirc`, no `twine`. The workflow is in `.github/workflows/publish.yml`.

**To release:**

```bash
# 1. Bump version in pyproject.toml
# 2. Commit and push
git add pyproject.toml
git commit -m "release: vX.Y.Z"
git push origin master

# 3. Tag and push — this triggers the publish
git tag vX.Y.Z
git push origin vX.Y.Z
```

GitHub Actions builds the package and publishes to PyPI automatically. That's it.

**Do NOT:**
- Run `twine upload` locally
- Store PyPI tokens anywhere
- Create a `.pypirc` file
- Try to publish any other way

If the publish workflow fails, check `gh run view --repo eidos-agi/resume-resume` for logs.

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update CHANGELOG.md with your changes
- Ensure `ruff check .` and `pytest` pass

## Reporting issues

Open an issue with:

1. What you were trying to do
2. What happened instead
3. Steps to reproduce

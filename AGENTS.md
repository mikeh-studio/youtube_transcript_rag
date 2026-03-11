# Repository Guidelines

## Project Structure & Module Organization
- `multilingual/`: core retrieval pipeline (`video_library.py`, `rag_engine.py`, `text_processing.py`) and language-specific tests in `multilingual/tests/`.
- `local_preview/`: local HTTP API (`local_api.py`) plus static UI in `local_preview/web/`.
- `tests/`: top-level regression tests for retrieval, feedback tuning, and library behavior.
- `data/`: local indexed artifacts, runtime data, and cache files (do not commit generated files).
- `docs/media/`: screenshots used in documentation.

## Build, Test, and Development Commands
- `cd youtube_rag_v2_portfolio && pip install -r requirements.txt`: install Python dependencies.
- `python local_preview/local_api.py`: run local UI + API at `http://127.0.0.1:8000/`.
- `pytest tests/`: run repository-level tests.
- `pytest multilingual/tests/`: run multilingual module tests.
- `pytest tests/ multilingual/tests/ -q`: quick full test pass before opening a PR.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and PEP 8-style naming.
- Modules/files: `snake_case.py`; classes: `PascalCase`; functions/variables: `snake_case`; constants: `UPPER_SNAKE_CASE`.
- Keep functions focused; prefer explicit return shapes (`dict` keys stable across code paths).
- Add short docstrings for non-trivial public functions and API handlers.

## Testing Guidelines
- Framework: `pytest` with test files named `test_*.py`.
- Co-locate specialized tests (`multilingual/tests/`) and keep cross-module behavior in top-level `tests/`.
- Mock external APIs (`openai`, `anthropic`, network fetches) for deterministic runs.
- No strict coverage gate is configured; add tests for all bug fixes and behavior changes.

## Critical Validation Rules (Local Preview)
- Ingestion is a core feature. For any UI/API ingestion verification, run the real backend: `python local_preview/local_api.py`.
- Do not treat static hosting (`python -m http.server ...`) or mock `/v1/*` responses as a valid ingest verification path.
- When asked to confirm ingest works, validate end-to-end with a real YouTube URL and report:
  - `POST /v1/ingest/videos` response
  - job status from `GET /v1/ingest/jobs` (must reach `completed`)
  - video presence/chunk count in `GET /v1/videos`
- If network/model download blocks startup (e.g., Hugging Face model fetch), explicitly report that as a blocker instead of claiming ingestion works.
- When sharing a manual test URL for core features, always provide the real app URL: `http://127.0.0.1:8000/index.html`.

## Commit & Pull Request Guidelines
- Follow existing commit style: conventional prefixes like `feat:` and `docs:` with concise summaries.
- Keep commits scoped (one logical change per commit) and include tests/docs with code changes.
- PRs should include: purpose, key changes, test commands run, and screenshots for UI updates (`local_preview/web/*`).
- Link related issues/tasks and list any required environment variables (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Security & Configuration Tips
- Never commit secrets or local runtime artifacts.
- Use environment variables for provider keys and keep `.gitignore` entries for generated data/logs up to date.

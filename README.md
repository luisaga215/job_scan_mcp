# Job Screening MCP Server

An MCP server that scrapes job postings, screens them against your CV/profile, evaluates the best matches, and renders an interactive decision dashboard — plus a **Dynamic CV Tailoring** engine that rewrites your CV for a specific job and exports it to PDF.

Built with a decoupled, enterprise-grade architecture: database → services → LangGraph evaluation → MCP presentation layer.

---

## Features

- **Job pipeline:** `scrape → filter (remote/salary/visa) → fast screening → deep evaluation → report`.
- **Visa & relocation aware:** detects sponsorship / H-1B / TN / relocation signals, and can filter strictly (`require_visa_friendly`). Screening prioritizes visa-friendly roles and rejects postings that exclude sponsorship.
- **Resilient batching:** long LLM batches run in waves with a time budget — they return partial results instead of timing out.
- **Interactive dashboard:** Tailwind + Alpine.js report with multi-level sorting, rich filters (work mode, date range, salary, visa, friction, red flags…), skill-gap chips, a kanban application tracker, and **report history** with AI summaries.
- **Dynamic CV Tailoring:** given a job description, rewrites your CV in XYZ format, flags every changed bullet with a `match_reason`, previews it with a highlight engine, and exports an ATS-friendly PDF via Playwright.

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An LLM provider key (DeepSeek, Gemini, or OpenAI-compatible) — see [Configuration](#configuration)

---

## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:luisaga215/job_scan_mcp.git
cd job_scan_mcp
uv sync

# 2. Configure environment
cp .env.example .env
#   - fill in your LLM API key (e.g. DEEPSEEK_API_KEY or GEMINI_API_KEY)

# 3. Run the MCP server (stdio)
uv run job-scan-mcp
```

> The server auto-initializes its SQLite database and folders under `~/.job_evaluator/` on first run.

### Configuration (`.env`)

| Variable | Description |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek key (OpenAI-compatible endpoint) |
| `GEMINI_API_KEY` | Google Gemini key |
| `OPENAI_API_KEY` | OpenAI-compatible key (vLLM, LM Studio, Groq…) |
| `OPENAI_BASE_URL` | Optional custom OpenAI-compatible base URL |
| `OLLAMA_BASE_URL` | Ollama endpoint (default `http://localhost:11434`) |
| `DEFAULT_SCREENING_MODEL` | Model for fast screening, e.g. `deepseek/deepseek-chat` |
| `DEFAULT_EVALUATION_MODEL` | Model for deep evaluation, e.g. `deepseek/deepseek-chat` |

The default models can be overridden at runtime with the `configure_llm` tool.

---

## MCP Tools

| Tool | Purpose |
| --- | --- |
| `sync_cv(file_path)` | Parse a CV (PDF/MD/TXT) into a structured candidate profile |
| `get_user_profile()` | Return the synced candidate profile |
| `fetch_and_filter_jobs(queries, locations, ...)` | Scrape jobs and apply remote/salary/visa filters |
| `run_fast_screening(batch_size, ...)` | Quick relevance screening (visa-friendly first, partial batching) |
| `run_deep_evaluation(batch_size, ...)` | LangGraph deep evaluation (fit, seniority, red flags) |
| `get_pipeline_status()` | Pipeline counts + active LLM configuration |
| `set_job_application_status(job_id, status)` | Persist kanban state: `apply` / `applied` / `interview` / `rejected` |
| `generate_html_report()` | Render the interactive dashboard + snapshot history |
| `archive_report(file)` / `restore_report(file)` | Soft-delete a report snapshot (moves to `reports/archive/`) |
| `generate_tailored_cv(job_description_text, base_cv_json)` | Rewrite a CV for a JD, flagging changes with `modified` + `match_reason` |
| `export_cv_to_pdf(tailored_cv_data, file_name)` | Render the tailored CV to an ATS-friendly PDF (Playwright) |
| `configure_llm(stage, provider, model, base_url)` | Override the model per pipeline stage |

### Typical workflow

```
sync_cv → fetch_and_filter_jobs → run_fast_screening → run_deep_evaluation
       → generate_html_report → set_job_application_status → generate_tailored_cv → export_cv_to_pdf
```

---

## Registering the server with a client

### opencode (`opencode.json`)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "job-scan-mcp": {
      "type": "local",
      "command": ["uv", "run", "job-scan-mcp"],
      "enabled": true
    }
  }
}
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "job-scan-mcp": {
      "command": "uvx",
      "args": ["--from", "c:/path/to/job_scan_mcp", "job-scan-mcp"]
    }
  }
}
```

> The `.env` file is loaded automatically from the project root, so no keys need to be passed in the client config.

---

## Development & Testing

```bash
uv run pytest --cov=src
```

The suite covers services, the CV tailor (LLM flags + mocked Playwright), report history/archive, and the dashboard rendering. Frontend logic (filters, sorting, highlights) is exercised with headless Node assertions.

## Project layout

```
src/job_scan_mcp/
├── mcp_server.py        # MCP tools (presentation layer)
├── models.py            # SQLModel + Pydantic schemas
├── database.py          # async SQLite + migrations
├── repository.py        # data access layer
├── services/
│   ├── job_service.py   # scraping + deterministic filters (remote/salary/visa)
│   ├── screening.py     # fast screening (batching + concurrency guards)
│   ├── evaluation.py    # LangGraph deep evaluation
│   ├── cv_service.py    # CV parsing → candidate profile
│   ├── cv_tailor.py     # CV tailoring (LLM) + PDF export (Playwright)
│   ├── llm_factory.py   # provider abstraction (openai/gemini/ollama/deepseek)
│   └── report.py        # dashboard + report history/manifest + AI summaries
└── templates/           # Jinja2 dashboard + CV preview
```

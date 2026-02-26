# Agent Skills Demo

## Short Demo of Capabilities

https://github.com/user-attachments/assets/10a82328-3e38-4a91-9511-c8aee043324c

(**Note:** inference times are sped up for the sake of demonstration)

A **Databricks App** serving a multi-turn LangGraph agent with [Claude Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview). The agent uses filesystem-based skills for progressive disclosure and saves all output to Unity Catalog Volumes.

## Architecture

```
Databricks App (FastAPI / MLflow Agent Server)
       │
       ▼
DocumentResponsesAgent          ← per-request: derives thread_id, session_id
       │
       ▼
DocumentAgent (LangGraph)       ← MemorySaver checkpointer, keyed by thread_id
  ┌────────────┐   ┌────────────┐
  │ agent_node │◀──│  tool_node │
  │  (LLM)     │──▶│  (tools)   │
  └────────────┘   └────────────┘
       │
       ▼
  UC Volume output  /Volumes/…/{session_id}/
```

**Multi-turn isolation:** each conversation gets a `thread_id = user_email:conversation_id`. LangGraph's `MemorySaver` checkpoints the full message history per thread, so follow-up questions have full context. Different users/conversations never share state.

## Project Structure

```
agent-skills-demo/
├── agent/
│   ├── config.py            # AgentConfig — env vars, UC Volume path, skills dir
│   ├── graph.py             # LangGraph workflow (DocumentAgent)
│   ├── tools.py             # Tool implementations + ToolContext
│   ├── responses_agent.py   # MLflow ResponsesAgent wrapper + thread_id logic
│   └── __init__.py
├── app.py                   # @invoke / @stream handlers (MLflow Agent Server)
├── start_server.py          # Entrypoint: starts uvicorn via AgentServer
├── app.yaml                 # Databricks App config (env vars, resources)
├── pyproject.toml
└── bin/
    ├── test_app_endpoint.py # Test client (interactive REPL + single-shot)
    └── get_traces.py        # Fetch a trace by ID from MLflow
.claude/
└── skills/
    ├── docx/
    │   └── SKILL.md         # Word document skill (create, edit, analyze)
    └── pdf/
        └── SKILL.md         # PDF skill (read, fill blanks, annotate, merge, redact)
```

## Prerequisites

- **Python 3.10–3.11**
- **Databricks CLI** used to deploy the bundle and app
- Access to the Unity Catalog Volume defined in `AGENT_UC_VOLUME_PATH`

## Local Development

```bash
# Install dependencies
uv sync

# Start the agent server locally (port 8000 by default)
uv run start_server.py

# With multiple workers
uv run start_server.py --workers 4

# On a custom port
uv run start_server.py --port 9000
```

The server exposes:
- `POST /invocations` — standard MLflow Responses API endpoint
- `GET /health` — health check
- `GET /agent/info` — agent metadata

### Environment Variables

Copy from `app.yaml` or set directly:

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL_ENDPOINT` | `databricks-gpt-5-2` | Databricks model serving endpoint |
| `AGENT_UC_VOLUME_PATH` | _(required)_ | UC Volume root; agent can read anywhere under this path |
| `AGENT_OUTPUT_MODE` | `auto` | `auto`, `uc_volume`, or `local` |
| `AGENT_SKILLS_DIR` | `./.claude/skills` | Path to skills directory |
| `AGENT_MAX_ITERATIONS` | `10` | Max LangGraph iterations per request |
| `AGENT_LLM_TIMEOUT` | `120` | Per-LLM-call timeout in seconds |
| `MLFLOW_EXPERIMENT_ID` | — | MLflow experiment for tracing |
| `DATABRICKS_CONFIG_PROFILE` | — | Databricks CLI profile (local dev) |

## Testing the Deployed App

### Interactive REPL (multi-turn)

```bash
uv run bin/test_app_endpoint.py
```

```
Multi-turn session  conversation_id=3f2a1b4c-…
Type your message and press Enter. Use 'quit' or Ctrl-D to exit.

you> Create a quarterly status report as a Word doc
agent> I've created the report and saved it to …
  [session=a1b2c3d4  path=/Volumes/…/a1b2c3d4/quarterly_report.docx]

you> Add an executive summary section at the top
agent> Done — the executive summary has been added …
```

The same `conversation_id` is sent with every turn so the server routes all messages to the same LangGraph checkpoint.

### Single-shot

```bash
uv run bin/test_app_endpoint.py "List the skills available"
```

### Fetching Traces

```bash
uv run bin/get_traces.py <trace_id>
```

`trace_id` is returned in the MLflow response metadata when `x-mlflow-return-trace-id: true` is set on the request.

## Available Tools

| Tool | Description |
|---|---|
| `list_skills` | List all skills discovered in the skills directory |
| `load_skill` | Load full SKILL.md instructions for a skill |
| `execute_python` | Execute Python code; result available to `save_to_volume` |
| `execute_bash` | Run shell commands; working directory persists within a turn |
| `save_to_volume` | Save a file to the session's UC Volume directory |
| `read_from_volume` | Read a file by filename (session folder) or full absolute path |
| `copy_to_session` | Copy a file from another session into the current one |
| `list_volume_files` | List files in the session folder or any path in the volume |

## How Skills Work

1. **Discovery** — at startup, the agent scans `.claude/skills/` for directories containing `SKILL.md`
2. **Progressive disclosure** — only skill names and descriptions go into the system prompt; full instructions are loaded on demand via `load_skill`
3. **Execution** — skills run via `execute_python` or `execute_bash` and write output via `save_to_volume`

## Available Skills

### `docx` — Word Documents

Create, edit, and analyze `.docx` files using `python-docx` (new documents) or an unpack/edit XML/repack workflow (editing existing documents). Supports tracked changes, comments, images, tables, headers/footers, and page layout.

**Trigger:** any mention of `.docx`, Word doc, report, memo, or letter.

### `pdf` — PDF Files

Read, fill, annotate, and manipulate PDF files using PyMuPDF (`fitz`). Primary use case is filling blanks in PDF templates — the skill detects whether blanks are AcroForm fields, underscore placeholders, or drawn lines and applies the appropriate fill strategy. Also supports annotations (highlight, underline, sticky notes), redaction, watermarks, page split/merge/rotate, and rendering pages to images.

**Trigger:** any mention of `.pdf`, form filling, PDF annotation, or PDF page manipulation.

**Note on test data:** the agent will decline to edit completed financial documents (receipts, invoices with real figures) as a fraud-prevention measure. Use synthetic test fixtures or real template files with blank fields.

### Adding a Skill

Create a directory under `.claude/skills/` with a `SKILL.md` file:

```
.claude/skills/my-skill/
└── SKILL.md        # YAML frontmatter: name, description; body: instructions
```

```markdown
---
name: My Skill
description: One-line description shown in the system prompt
---

## Instructions

Step-by-step instructions for the agent…
```

## Deploying to Databricks Apps

The app is configured by `app.yaml` and deployed via the Databricks CLI:

```bash
databricks apps deploy {your_app_name_here} --profile {your_profile_here}
```

The app runs `python start_server.py` on startup. All output is written to the UC Volume configured via `AGENT_UC_VOLUME_PATH`.

## Differences from Upstream Skills

The skills in this repo are adapted from the [Anthropic skills library](https://github.com/anthropics/skills/tree/main/skills/docx) for a Databricks agent environment. Key changes:

### Runtime environment

The upstream skills are designed for general CLI/MCP use. This project runs them inside a LangGraph agent on Databricks Apps, so all file I/O goes through the Databricks Files API (`read_from_volume` / `save_to_volume`) rather than the local filesystem.

### Python instead of Node.js

| Capability | Upstream | This project |
|---|---|---|
| Create `.docx` | `docx-js` (Node.js) | `python-docx` (Python) |
| Extract text | `pandoc` (system binary) | `mammoth` (Python package) |
| PDF → image | `poppler-utils` (system binary) | `PyMuPDF` / `fitz` (Python package) |

The Node.js and system-binary dependencies (`pandoc`, `poppler-utils`, LibreOffice) are not available in the Databricks Apps runtime, so they are replaced with pure-Python equivalents that produce equivalent results.

### LibreOffice operations not available

The upstream skill includes `accept_changes.py` (bulk-accept tracked changes) and `.doc` → `.docx` conversion via LibreOffice. Both are unavailable here. The workaround for tracked changes is the manual XML unpack/edit/repack workflow documented in the `docx` skill.

### PDF skill

The upstream library has a separate `pdf` skill; this project includes a version adapted for the same Databricks/UC Volume runtime as the `docx` skill, with a focus on filling blanks in PDF templates.

## References

- [Claude Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [MLflow Agent Server](https://mlflow.org/docs/latest/genai/serving/agent-server/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)

## Disclaimer

- This is just for the sake of demonstrating skills integration with databricks apps and is not intended to be immediately adapted for large-scale use

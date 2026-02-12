# Agent Skills Demo

Minimal example of a **Databricks agent** integrating **Claude Agent Skills** using the **LangGraph** framework. This version uses **Python-only** skills and saves output to **Unity Catalog Volumes**.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                          │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────┐  │
│  │   Agent     │───▶│    Tools    │───▶│   Agent            │  │
│  │   Node      │◀───│    Node     │    │   (repeat)         │  │
│  └──────┬──────┘    └──────┬──────┘    └────────────────────┘  │
│         │                  │                                   │
│         │           ┌──────▼──────┐                            │
│         │           │  execute_python                          │
│         │           │  save_to_volume                          │
│         │           │  load_skill                              │
│         │           └──────┬──────┘                            │
│         │                  │                                   │
│         └──────────────────┼───────────────────────────────────┘
│                            │
│                   ┌────────▼────────┐
│                   │  ChatDatabricks │
│                   │  (GPT-5-2)      │
│                   └─────────────────┘
└────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  .claude/   │    │  python-    │    │  UC Volume  │
   │   skills/   │    │   docx      │    │  /Volumes/  │
   │   └─docx/   │    │  (library)  │    │  btbeal/... │
   └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

- **LangGraph** for workflow orchestration with a ReAct-style agent loop
- **ChatDatabricks** integration using the `databricks-gpt-5-2` endpoint
- **Python-only skills** - no Node.js, npm, or shell commands required
- **Unity Catalog Volume output** - generated files saved to `/Volumes/btbeal/docx_agent_skills_demo/created_docs`
- **Databricks SDK profile** support (`FE-EAST`)
- **Deployable to Model Serving** - Python-only design works in Databricks serving environment

## Project Structure

```
agent-skills-demo/
├── main.py                      # Entry point
├── pyproject.toml               # UV package management
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── config.py            # Databricks, UC Volume & Skills configuration
│       ├── graph.py             # LangGraph workflow definition
│       └── tools.py             # Tools: execute_python, save_to_volume, etc.
└── .claude/
    └── skills/
        └── docx/                # Python-only docx skill
            └── SKILL.md         # Skill instructions using python-docx
```

## Prerequisites

1. **Databricks CLI configured** with the `FE-EAST` profile in `~/.databrickscfg`:
   ```ini
   [FE-EAST]
   host = https://your-workspace.cloud.databricks.com
   token = your-token
   ```

2. **Unity Catalog Volume** exists at:
   ```
   /Volumes/btbeal/docx_agent_skills_demo/created_docs
   ```

3. **Python 3.10-3.11** (required for Databricks compatibility)

## Installation

```bash
# Enter the project
cd agent-skills-demo

# Install dependencies with uv
uv sync

# Verify the setup
uv run python -c "from src.agent import AgentConfig; c = AgentConfig(); print(f'Skills: {c.available_skills}')"
```

## Usage

### Interactive Mode

```bash
uv run python main.py
```

### Programmatic Usage

```python
from src.agent import AgentConfig, create_agent_graph
from langchain_core.messages import HumanMessage

# Create config
config = AgentConfig(
    databricks_profile="FE-EAST",
    model_endpoint="databricks-gpt-5-2",
    uc_volume_path="/Volumes/btbeal/docx_agent_skills_demo/created_docs",
)

# Create the LangGraph workflow
graph = create_agent_graph(config)

# Run a query
result = graph.invoke({
    "messages": [HumanMessage(content="Create a Word document with a project status report")],
    "iteration_count": 0,
})

# Get the response
for msg in reversed(result["messages"]):
    if hasattr(msg, "content"):
        print(msg.content)
        break
```

### Custom Configuration

```python
from src.agent import AgentConfig

config = AgentConfig(
    databricks_profile="CUSTOM-PROFILE",
    model_endpoint="your-model-endpoint",
    uc_volume_path="/Volumes/your_catalog/your_schema/your_volume",
    max_iterations=5,
    session_id="custom-session-123",  # Optional: auto-generated if not set
)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `load_skill` | Load full instructions from a skill's SKILL.md |
| `execute_python` | Execute Python code (python-docx, etc.) |
| `save_to_volume` | Save files to Unity Catalog Volume |
| `read_from_volume` | Read files from Unity Catalog Volume |
| `list_volume_files` | List files in the session's output directory |

## How Skills Work

This demo implements the [Claude Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) pattern, adapted for Databricks:

1. **Skill Discovery**: Skills are filesystem-based resources in `.claude/skills/`
2. **Metadata Loading**: YAML frontmatter in `SKILL.md` describes when to use the skill
3. **Progressive Disclosure**: Full instructions are loaded only when the skill is triggered
4. **Python Execution**: Code runs via `exec()` with python-docx available
5. **UC Volume Output**: Generated files saved to Unity Catalog for user access

## Example Prompts

Try these with the docx skill:

```
"Create a Word document with the title 'Project Report' and three bullet points"

"Create a quarterly status report with a table showing milestone progress"

"List the files I've created in this session"
```

## Deploying to Model Serving

This agent is designed to work in Databricks Model Serving:

1. **No system dependencies** - Uses only pip-installable Python packages
2. **No shell execution** - All operations via Python code
3. **UC Volume storage** - Files accessible across the Databricks workspace
4. **Stateless** - Each session gets a unique ID for output organization

### Logging with MLflow

```python
import mlflow
from src.agent import create_agent_graph, AgentConfig

# Log the agent
with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=create_agent_graph(AgentConfig()),
        artifact_path="agent",
        registered_model_name="docx-agent",
    )
```

## References

- [Claude Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [python-docx Documentation](https://python-docx.readthedocs.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Databricks LangChain Integration](https://docs.databricks.com/aws/en/generative-ai/agent-framework/langchain-uc-integration)
- [Unity Catalog Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html)

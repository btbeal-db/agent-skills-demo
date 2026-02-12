"""Databricks App entry point using MLflow AgentServer.

This file serves the agent using MLflow's AgentServer, which provides:
- /invocations endpoint for querying the agent
- Built-in tracing and observability
- Automatic request routing and error handling
"""

import os
import sys
import logging

# Add src to path so we can import the agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlflow.genai.agent_server import AgentServer, invoke, stream
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent import AgentConfig, DocumentResponsesAgent

logger = logging.getLogger(__name__)

_config = AgentConfig.from_env()
_responses_agent = DocumentResponsesAgent(config=_config)
logger.info("Agent initialized. Output path: %s", _config.session_output_path)
logger.info("Available skills: %s", _config.available_skills)


@invoke()
def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming agent invocation."""
    return _responses_agent.predict(request)


@stream()
def handle_stream(request: ResponsesAgentRequest):
    """Handle streaming agent invocation."""
    yield from _responses_agent.predict_stream(request)


# Create the server with ResponsesAgent type for proper validation
server = AgentServer(agent_type="ResponsesAgent")
app = server.app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

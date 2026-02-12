"""Agent handlers for MLflow Agent Server."""

import os
import sys
import logging
from collections.abc import Generator

# Add src to path so we can import the agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlflow.genai.agent_server import (
    invoke,
    stream,
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent import AgentConfig, DocumentResponsesAgent

logger = logging.getLogger(__name__)

_config = AgentConfig.from_env()
_responses_agent = DocumentResponsesAgent(config=_config)
logger.info("Agent initialized. Output path: %s", _config.session_output_path)
logger.info("Available skills: %s", _config.available_skills)


@invoke()
def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle synchronous Responses API requests."""
    return _responses_agent.predict(request)


@stream()
def handle_stream(
    request: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Handle streaming Responses API requests."""
    yield from _responses_agent.predict_stream(request)

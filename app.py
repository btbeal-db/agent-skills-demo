"""Agent handlers for MLflow Agent Server."""

import logging
from collections.abc import AsyncGenerator

import mlflow
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

# Enable LangChain autologging so model/tool planning traces are captured in MLflow.
mlflow.langchain.autolog(log_traces=True, run_tracer_inline=False)

_config = AgentConfig.from_env()
# Databricks App serving should always persist outputs to UC Volume.
_config.output_mode = "uc_volume"
_responses_agent = DocumentResponsesAgent(config=_config)
logger.info(
    "Agent initialized. output_mode=%s output_path=%s",
    _config.output_mode,
    _config.session_output_path,
)
logger.info("Available skills: %s", _config.available_skills)


@invoke()
def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle synchronous Responses API requests."""
    return _responses_agent.predict(request)


@stream()
async def handle_stream(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming Responses API requests via async generator."""
    async for event in _responses_agent.predict_stream(request):
        yield event

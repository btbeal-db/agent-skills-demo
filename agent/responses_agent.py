"""ResponsesAgent wrapper for serving the document LangGraph agent."""

from __future__ import annotations

import dataclasses
import hashlib
import uuid
from collections.abc import Generator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from .config import AgentConfig
from .graph import DocumentAgent


class DocumentResponsesAgent(ResponsesAgent):
    """MLflow ResponsesAgent wrapper for DocumentAgent."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.document_agent = DocumentAgent(self.config)

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text from Responses API content payloads."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if hasattr(item, "text"):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
            return " ".join(text_parts)
        return str(content or "")

    @classmethod
    def _to_langchain_messages(cls, messages: list[Any]) -> list:
        """Convert Responses API messages to LangChain message objects."""
        lc_messages = []
        for msg in messages:
            if hasattr(msg, "role"):
                role = msg.role
                content = msg.content if hasattr(msg, "content") else ""
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            text_content = cls._extract_text_content(content)
            if role == "user":
                lc_messages.append(HumanMessage(content=text_content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=text_content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=text_content))
        return lc_messages

    @staticmethod
    def _extract_final_response_content(result: dict[str, Any]) -> str:
        """Extract the final assistant content from a LangGraph result."""
        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return ""

    @staticmethod
    def _build_response(text: str) -> ResponsesAgentResponse:
        """Build a standard ResponsesAgentResponse payload."""
        return ResponsesAgentResponse(
            output=[{
                "id": str(uuid.uuid4()),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }]
        )

    def _get_thread_id(self, request: ResponsesAgentRequest) -> str:
        """Derive a stable thread_id for this user+conversation.

        Uses the authenticated user's email (from Databricks App headers) combined
        with a caller-supplied conversation_id from custom_inputs.  If no
        conversation_id is provided a new UUID is generated, making the request
        behave like a fresh single-turn conversation.
        """
        try:
            from mlflow.genai.agent_server.server import get_request_headers
            headers = get_request_headers()
        except Exception:
            headers = {}

        user = headers.get("x-forwarded-email") or "anonymous"
        custom_inputs = getattr(request, "custom_inputs", None) or {}
        if not isinstance(custom_inputs, dict):
            custom_inputs = {}
        conversation_id = custom_inputs.get("conversation_id") or str(uuid.uuid4())
        return f"{user}:{conversation_id}"

    @staticmethod
    def _session_id_from_thread(thread_id: str) -> str:
        """Derive a stable 8-char session_id from a thread_id."""
        return hashlib.md5(thread_id.encode()).hexdigest()[:8]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle non-streaming invocation."""
        if not request.input:
            return self._build_response("No input provided")

        try:
            thread_id = self._get_thread_id(request)
            session_id = self._session_id_from_thread(thread_id)

            lc_messages = self._to_langchain_messages(request.input)
            result = self.document_agent.invoke(lc_messages, session_id=session_id, iteration_count=0)
            response_content = self._extract_final_response_content(result)

            response = self._build_response(response_content)
            session_config = dataclasses.replace(self.config, session_id=session_id)
            response.custom_outputs = {
                "session_id": session_id,
                "thread_id": thread_id,
                "output_path": session_config.session_output_path,
            }
            return response
        except Exception as e:
            return self._build_response(f"Error: {str(e)}")

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle streaming invocation using LangGraph's real streaming."""
        if not request.input:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": str(uuid.uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No input provided"}],
                },
            )
            return

        try:
            thread_id = self._get_thread_id(request)
            session_id = self._session_id_from_thread(thread_id)

            lc_messages = self._to_langchain_messages(request.input)
            item_id = "msg_" + session_id
            final_content = ""

            for msg_chunk, metadata in self.document_agent.stream(
                lc_messages, session_id=session_id, iteration_count=0
            ):
                # Only forward text tokens from the agent node's final responses.
                # Skip tool-calling chunks (they have tool_call_chunks, not content)
                # and skip ToolMessages entirely.
                if (
                    isinstance(msg_chunk, AIMessageChunk)
                    and metadata.get("langgraph_node") == "agent"
                    and msg_chunk.content
                    and not getattr(msg_chunk, "tool_call_chunks", None)
                ):
                    delta = str(msg_chunk.content)
                    final_content += delta
                    yield ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        item_id=item_id,
                        delta=delta,
                    )

            session_config = dataclasses.replace(self.config, session_id=session_id)
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_content}],
                    "custom_outputs": {
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "output_path": session_config.session_output_path,
                    },
                },
            )
        except Exception as e:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": str(uuid.uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Error: {str(e)}"}],
                },
            )

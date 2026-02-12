"""ResponsesAgent wrapper for serving the document LangGraph agent."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

    def __init__(self, config: AgentConfig | None = None, chunk_size: int = 50):
        self.config = config or AgentConfig.from_env()
        self.document_agent = DocumentAgent(self.config)
        self.chunk_size = chunk_size

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
            output=[
                {
                    "id": str(uuid.uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle non-streaming invocation."""
        if not request.input:
            return self._build_response("No input provided")

        try:
            lc_messages = self._to_langchain_messages(request.input)
            result = self.document_agent.invoke(lc_messages, iteration_count=0)
            response_content = self._extract_final_response_content(result)

            response = self._build_response(response_content)
            response.custom_outputs = {
                "session_id": self.config.session_id,
                "output_path": self.config.session_output_path,
            }
            return response
        except Exception as e:
            return self._build_response(f"Error: {str(e)}")

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Handle streaming invocation by chunking final text output."""
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
            lc_messages = self._to_langchain_messages(request.input)
            result = self.document_agent.invoke(lc_messages, iteration_count=0)
            response_content = self._extract_final_response_content(result)

            item_id = "msg_" + self.config.session_id
            for i in range(0, len(response_content), self.chunk_size):
                chunk_text = response_content[i : i + self.chunk_size]
                yield ResponsesAgentStreamEvent(
                    type="response.output_text.delta",
                    item_id=item_id,
                    delta=chunk_text,
                )

            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response_content}],
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


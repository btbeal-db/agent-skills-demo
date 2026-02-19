"""LangGraph workflow for the document agent with Claude Skills integration."""

from __future__ import annotations

import dataclasses
import logging
import operator
from collections.abc import Generator
from typing import Annotated, Any, Literal, TypedDict

import mlflow
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks

from .config import AgentConfig
from .tools import AGENT_TOOLS, ToolContext, build_skill_context, handle_tool_call  # ToolContext used via from_dict/to_dict

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    tool_context: dict  # serialized ToolContext — kept as plain dict for checkpoint compatibility
    session_id: str
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


class DocumentAgent:
    """LangGraph-based document agent with tool-calling workflow."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.llm = ChatDatabricks(
            endpoint=self.config.model_endpoint,
            workspace_client=self._create_workspace_client(),
            temperature=0.1,
        )
        self.skill_context = build_skill_context(self.config)
        self._checkpointer = MemorySaver()
        self._compiled_graph = None

    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client using runtime identity or local profile."""
        if self.config.is_running_in_databricks:
            return WorkspaceClient()
        if self.config.databricks_profile:
            return WorkspaceClient(profile=self.config.databricks_profile)
        return WorkspaceClient()

    def _build_system_prompt(self, session_output_path: str) -> str:
        """Build the system prompt with skill and storage context."""
        return f"""You are a helpful AI assistant with access to specialized skills and Unity Catalog storage.

{self.skill_context}

## How to Use Skills

1. Review the available skill descriptions in the system prompt.
2. When a request matches a skill, call `load_skill` to load the full instructions on demand.

## Storage

You have full read access to the Unity Catalog Volume at: {self.config.uc_volume_path}

- **To find files**: use `list_volume_files` starting from the volume root or any subdirectory. Browse freely — you are not limited to the session folder.
- **To read files**: use `read_from_volume` with the full absolute path (e.g. `{self.config.uc_volume_path}/some/folder/file.pdf`).
- **To save files**: always write to the current session folder using `save_to_volume`. Session path: {session_output_path}

If the user references a file and you cannot locate it immediately, search the volume before giving up. If you still cannot find it after searching, ask the user to confirm the path or folder.

## Guidelines

- Always load a skill's instructions before attempting to use it
- Use progressive disclosure: keep only skill summaries in context until `load_skill` is needed
- Follow the skill's documented workflows exactly
- After creating a document, always save it with `save_to_volume`
- Report the file path to the user after saving
- If a skill isn't appropriate for the task, explain what you can and cannot do
"""

    def _get_request_config(self, session_id: str) -> AgentConfig:
        """Return a config copy scoped to the given session_id."""
        return dataclasses.replace(self.config, session_id=session_id)

    def _ensure_system_prompt(
        self, messages: list[BaseMessage], session_output_path: str
    ) -> list[BaseMessage]:
        """Ensure a system prompt is the first message.

        On the first turn of a conversation there is no system prompt yet, so we
        prepend one.  On subsequent turns the checkpoint already contains the
        system prompt from the first turn, so we leave it in place.
        """
        if messages and isinstance(messages[0], SystemMessage):
            return messages
        return [SystemMessage(content=self._build_system_prompt(session_output_path)), *messages]

    def agent_node(self, state: AgentState) -> AgentState:
        """Main agent node - calls the LLM with tools."""
        session_id = state.get("session_id") or self.config.session_id
        request_config = self._get_request_config(session_id)

        messages = self._ensure_system_prompt(state["messages"], request_config.session_output_path)
        iteration = state.get("iteration_count", 0)
        tool_context = ToolContext.from_dict(state.get("tool_context") or {})

        logger.info("[agent_node] Iteration %s with %s message(s)", iteration + 1, len(messages))
        response = self.llm.invoke(messages, tools=AGENT_TOOLS)
        has_tools = bool(response.tool_calls) if hasattr(response, "tool_calls") else False
        usage = response.usage_metadata or {}
        logger.info(
            "[agent_node] LLM responded. tool_calls=%s | tokens: input=%s, output=%s",
            has_tools,
            usage.get("input_tokens", "n/a"),
            usage.get("output_tokens", "n/a"),
        )

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "tool_context": tool_context.to_dict(),
            "session_id": session_id,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    def tool_node(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM response."""
        messages = state["messages"]
        session_id = state.get("session_id") or self.config.session_id
        request_config = self._get_request_config(session_id)
        tool_context = ToolContext.from_dict(state.get("tool_context") or {})

        if not messages:
            return {
                "messages": [],
                "iteration_count": state.get("iteration_count", 0),
                "tool_context": tool_context.to_dict(),
                "session_id": session_id,
            }

        last_message = messages[-1]
        tool_messages: list[ToolMessage] = []

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info("[tool_node] Executing %s tool call(s)", len(last_message.tool_calls))
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                logger.info("[tool_node] Running tool '%s'", tool_name)

                with mlflow.start_span(name=f"tool:{tool_name}") as span:
                    span.set_attribute("tool.name", tool_name)
                    span.set_attribute("tool.call_id", tool_id)
                    span.set_inputs({"tool_name": tool_name, "tool_args": tool_args})
                    result = handle_tool_call(request_config, tool_name, tool_args, tool_context)
                    span.set_outputs({"result": result})

                logger.info("[tool_node] Tool '%s' completed", tool_name)
                tool_messages.append(ToolMessage(content=result, tool_call_id=tool_id))

        return {
            "messages": tool_messages,
            "iteration_count": state.get("iteration_count", 0),
            "tool_context": tool_context.to_dict(),
            "session_id": session_id,
        }

    def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)

        if iteration >= self.config.max_iterations:
            logger.info("[router] Ending run (max iterations=%s)", self.config.max_iterations)
            return "end"

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "end"

    def build(self):
        """Build and compile the LangGraph workflow."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        logger.info("Compiling DocumentAgent graph")
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")

        self._compiled_graph = workflow.compile(checkpointer=self._checkpointer)
        return self._compiled_graph

    def invoke(self, messages: list[BaseMessage], session_id: str, iteration_count: int = 0):
        """Invoke the agent graph with the provided messages."""
        logger.info(
            "Invoking DocumentAgent with %s message(s) [session=%s]", len(messages), session_id
        )
        thread_config = {"configurable": {"thread_id": session_id}}
        final_state = self.build().invoke(
            {
                "messages": messages,
                "session_id": session_id,
                "iteration_count": iteration_count,
                "tool_context": {},
                "input_tokens": 0,
                "output_tokens": 0,
            },
            config=thread_config,
        )
        logger.info(
            "[agent] Run complete. Total tokens — input=%s, output=%s",
            final_state.get("input_tokens", "n/a"),
            final_state.get("output_tokens", "n/a"),
        )
        return final_state

    def stream(
        self, messages: list[BaseMessage], session_id: str, iteration_count: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the agent graph execution, yielding state updates as they occur."""
        logger.info(
            "Streaming DocumentAgent with %s message(s) [session=%s]", len(messages), session_id
        )
        thread_config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "messages": messages,
            "session_id": session_id,
            "iteration_count": iteration_count,
            "tool_context": {},
            "input_tokens": 0,
            "output_tokens": 0,
        }
        total_input = total_output = 0
        for update in self.build().stream(initial_state, config=thread_config, stream_mode="updates"):
            if "agent" in update:
                total_input += update["agent"].get("input_tokens", 0)
                total_output += update["agent"].get("output_tokens", 0)
            yield update
        logger.info(
            "[agent] Run complete. Total tokens — input=%s, output=%s",
            total_input,
            total_output,
        )

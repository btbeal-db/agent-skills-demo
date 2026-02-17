"""LangGraph workflow for the document agent with Claude Skills integration."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Annotated, Any, Literal, TypedDict

import mlflow
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks

from .config import AgentConfig
from .tools import (
    AGENT_TOOLS,
    ToolContext,
    build_skill_context,
    handle_tool_call,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    tool_context: ToolContext


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
        self.system_prompt = self._build_system_prompt()
        self._compiled_graph = None

    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client using runtime identity or local profile."""
        if self.config.is_running_in_databricks:
            return WorkspaceClient()
        if self.config.databricks_profile:
            return WorkspaceClient(profile=self.config.databricks_profile)
        return WorkspaceClient()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with skill and storage context."""
        return f"""You are a helpful AI assistant with access to specialized skills and Unity Catalog storage.

{self.skill_context}

## How to Use Skills

1. Review the available skill descriptions in the system prompt.
2. When a request matches a skill, call `load_skill` to load the full instructions on demand.
3. If a source file is in another session, first use `copy_to_session` to duplicate it into the current session folder.
4. For editing an existing file, use `read_from_volume` first, then use `execute_python` with `source_doc_bytes` (provided automatically by the tool flow).
5. Follow the skill instructions and use `execute_python` for document operations.
6. Use `save_to_volume` to save generated files to Unity Catalog for the user to access.
7. Use `list_volume_files` to show the user what files have been created.

## File Output

All generated files are saved to Unity Catalog Volume:
- Session path: {self.config.session_output_path}
- Use `save_to_volume` with base64-encoded content

## Guidelines

- Always load a skill's instructions before attempting to use it
- Use progressive disclosure: keep only skill summaries in context until `load_skill` is needed
- Follow the skill's documented Python patterns exactly
- Never edit files directly in other sessions; copy them into the current session first
- After creating a document, always save it with `save_to_volume`
- Report the file path to the user after saving
- If a skill isn't appropriate for the task, explain what you can and cannot do
"""

    def _ensure_system_prompt(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Ensure the agent's canonical system prompt is present as the first message."""
        if messages and isinstance(messages[0], SystemMessage) and messages[0].content == self.system_prompt:
            return messages
        return [SystemMessage(content=self.system_prompt), *messages]

    def agent_node(self, state: AgentState) -> AgentState:
        """Main agent node - calls the LLM with tools."""
        messages = self._ensure_system_prompt(state["messages"])
        iteration = state.get("iteration_count", 0)
        tool_context = state.get("tool_context") or ToolContext()

        logger.info("[agent_node] Iteration %s with %s message(s)", iteration + 1, len(messages))
        try:
            response = self.llm.invoke(
                messages,
                tools=AGENT_TOOLS,
            )
            has_tools = bool(response.tool_calls) if hasattr(response, "tool_calls") else False
            logger.info("[agent_node] LLM responded. tool_calls=%s", has_tools)
        except Exception as e:
            logger.exception("[agent_node] LLM invocation failed")
            raise

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "tool_context": tool_context,
        }

    def tool_node(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM response."""
        messages = state["messages"]
        tool_context = state.get("tool_context") or ToolContext()

        if not messages:
            return {
                "messages": [],
                "iteration_count": state.get("iteration_count", 0),
                "tool_context": tool_context,
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
                    span.set_inputs(
                        {
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        }
                    )
                    result = handle_tool_call(self.config, tool_name, tool_args, tool_context)
                    span.set_outputs({"result": result})

                logger.info("[tool_node] Tool '%s' completed", tool_name)

                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_id,
                    )
                )
        return {
            "messages": tool_messages,
            "iteration_count": state.get("iteration_count", 0),
            "tool_context": tool_context,
        }

    def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        if not messages:
            logger.info("[router] Ending run (no messages in state)")
            return "end"

        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)

        if iteration >= self.config.max_iterations:
            logger.info("[router] Ending run (max iterations=%s)", self.config.max_iterations)
            return "end"

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.info("[router] Continuing to tools")
            return "tools"

        logger.info("[router] Ending run (final assistant response)")
        return "end"

    def build(self):
        """Build and compile the LangGraph workflow."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        logger.info("Compiling DocumentAgent graph")
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # Tools always go back to agent
        workflow.add_edge("tools", "agent")

        self._compiled_graph = workflow.compile()
        return self._compiled_graph

    def invoke(self, messages: list[BaseMessage], iteration_count: int = 0):
        """Invoke the agent graph with the provided messages."""
        logger.info(
            "Invoking DocumentAgent with %s input message(s), iteration_count=%s",
            len(messages),
            iteration_count,
        )
        return self.build().invoke(
            {
                "messages": messages,
                "iteration_count": iteration_count,
                "tool_context": ToolContext(),
            }
        )

    def stream(
        self, messages: list[BaseMessage], iteration_count: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the agent graph execution, yielding state updates as they occur.

        Yields:
            State update dicts containing node name and updated state values.
            Each update includes the node that produced it and the state changes.
        """
        logger.info(
            "Streaming DocumentAgent with %s input message(s), iteration_count=%s",
            len(messages),
            iteration_count,
        )
        initial_state = {
            "messages": messages,
            "iteration_count": iteration_count,
            "tool_context": ToolContext(),
        }

        # stream_mode="updates" yields {node_name: state_update} after each node
        for update in self.build().stream(initial_state, stream_mode="updates"):
            yield update


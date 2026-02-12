"""LangGraph workflow for the document agent with Claude Skills integration."""

from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .config import AgentConfig
from .nodes import (
    AGENT_TOOLS,
    build_skill_context,
    create_llm,
    handle_tool_call,
)


class AgentState(TypedDict):
    """State for the agent workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int


class DocumentAgent:
    """LangGraph-based document agent with tool-calling workflow."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.llm = create_llm(self.config)
        self.skill_context = build_skill_context(self.config)
        self.system_prompt = self._build_system_prompt()
        # Toggle verbose execution logs with AGENT_DEBUG=true
        self.debug = os.environ.get("AGENT_DEBUG", "false").lower() == "true"
        self._compiled_graph = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt with skill and storage context."""
        return f"""You are a helpful AI assistant with access to specialized skills and Unity Catalog storage.

{self.skill_context}

## How to Use Skills

1. When a user request matches a skill's capabilities, first use the `load_skill` tool to get detailed instructions.
2. Follow the skill's instructions carefully - they contain specific Python code patterns.
3. Use `execute_python` to run Python code for document operations (python-docx is available).
4. Use `save_to_volume` to save generated files to Unity Catalog for the user to access.
5. Use `list_volume_files` to show the user what files have been created.

## File Output

All generated files are saved to Unity Catalog Volume:
- Session path: {self.config.session_output_path}
- Use `save_to_volume` with base64-encoded content

## Guidelines

- Always load a skill's instructions before attempting to use it
- Follow the skill's documented Python patterns exactly
- After creating a document, always save it with `save_to_volume`
- Report the file path to the user after saving
- If a skill isn't appropriate for the task, explain what you can and cannot do
"""

    def _log(self, message: str) -> None:
        if self.debug:
            print(message, flush=True)

    def _ensure_system_prompt(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Ensure the agent's canonical system prompt is present as the first message."""
        if messages and isinstance(messages[0], SystemMessage) and messages[0].content == self.system_prompt:
            return messages
        return [SystemMessage(content=self.system_prompt), *messages]

    def agent_node(self, state: AgentState) -> AgentState:
        """Main agent node - calls the LLM with tools."""
        messages = self._ensure_system_prompt(state["messages"])
        iteration = state.get("iteration_count", 0)

        self._log(f"  [agent_node] Iteration {iteration + 1}, {len(messages)} messages")

        self._log("  [agent_node] Calling LLM...")
        try:
            response = self.llm.invoke(
                messages,
                tools=AGENT_TOOLS,
            )
            has_tools = bool(response.tool_calls) if hasattr(response, "tool_calls") else False
            self._log(f"  [agent_node] LLM responded. Has tool_calls: {has_tools}")
        except Exception as e:
            print(f"  [agent_node] LLM ERROR: {e}", flush=True)
            raise

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    def tool_node(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM response."""
        messages = state["messages"]
        if not messages:
            return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

        last_message = messages[-1]
        tool_messages: list[ToolMessage] = []

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            self._log(f"  [tool_node] Executing {len(last_message.tool_calls)} tool(s)")
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                self._log(f"    -> {tool_name}({list(tool_args.keys())})")

                result = handle_tool_call(self.config, tool_name, tool_args)

                if self.debug:
                    result_preview = result[:100] + "..." if len(result) > 100 else result
                    self._log(f"    <- {result_preview}")

                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_id,
                    )
                )
        else:
            self._log("  [tool_node] No tool calls to execute")

        return {"messages": tool_messages, "iteration_count": state.get("iteration_count", 0)}

    def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        if not messages:
            self._log("  [router] -> end (no messages)")
            return "end"

        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)

        if iteration >= self.config.max_iterations:
            self._log("  [router] -> end (max iterations)")
            return "end"

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            self._log("  [router] -> tools")
            return "tools"

        self._log("  [router] -> end (no tool calls)")
        return "end"

    def build(self):
        """Build and compile the LangGraph workflow."""
        if self._compiled_graph is not None:
            return self._compiled_graph

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
        prepared_messages = self._ensure_system_prompt(messages)
        return self.build().invoke(
            {
                "messages": prepared_messages,
                "iteration_count": iteration_count,
            }
        )


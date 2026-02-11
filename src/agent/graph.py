"""LangGraph workflow for the agent with Claude Skills integration."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
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


def create_agent_graph(config: AgentConfig | None = None):
    """Create the LangGraph agent workflow.
    
    This graph implements a ReAct-style agent loop:
    1. Process user input
    2. Call LLM with tool definitions
    3. If LLM wants to use tools, execute them and loop back
    4. If LLM has final answer, return it
    """
    if config is None:
        config = AgentConfig()
    
    llm = create_llm(config)
    skill_context = build_skill_context(config)
    
    # System prompt with skill awareness
    system_prompt = f"""You are a helpful AI assistant with access to specialized skills and Unity Catalog storage.

{skill_context}

## How to Use Skills

1. When a user request matches a skill's capabilities, first use the `load_skill` tool to get detailed instructions.
2. Follow the skill's instructions carefully - they contain specific Python code patterns.
3. Use `execute_python` to run Python code for document operations (python-docx is available).
4. Use `save_to_volume` to save generated files to Unity Catalog for the user to access.
5. Use `list_volume_files` to show the user what files have been created.

## File Output

All generated files are saved to Unity Catalog Volume:
- Session path: {config.session_output_path}
- Use `save_to_volume` with base64-encoded content

## Guidelines

- Always load a skill's instructions before attempting to use it
- Follow the skill's documented Python patterns exactly
- After creating a document, always save it with `save_to_volume`
- Report the file path to the user after saving
- If a skill isn't appropriate for the task, explain what you can and cannot do
"""

    def agent_node(state: AgentState) -> AgentState:
        """Main agent node - calls the LLM with tools."""
        messages = state["messages"]
        iteration = state.get("iteration_count", 0)
        
        print(f"  [agent_node] Iteration {iteration + 1}, {len(messages)} messages", flush=True)
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Call LLM with tools
        print(f"  [agent_node] Calling LLM...", flush=True)
        try:
            response = llm.invoke(
                messages,
                tools=AGENT_TOOLS,
            )
            print(f"  [agent_node] LLM responded. Has tool_calls: {bool(response.tool_calls) if hasattr(response, 'tool_calls') else 'N/A'}", flush=True)
        except Exception as e:
            print(f"  [agent_node] LLM ERROR: {e}", flush=True)
            raise
        
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
        }

    def tool_node(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM response."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_messages = []
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"  [tool_node] Executing {len(last_message.tool_calls)} tool(s)", flush=True)
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"    -> {tool_name}({list(tool_args.keys())})", flush=True)
                
                # Execute the tool
                result = handle_tool_call(config, tool_name, tool_args)
                
                # Truncate result for display
                result_preview = result[:100] + "..." if len(result) > 100 else result
                print(f"    <- {result_preview}", flush=True)
                
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_id,
                    )
                )
        else:
            print(f"  [tool_node] No tool calls to execute", flush=True)
        
        return {"messages": tool_messages, "iteration_count": state["iteration_count"]}

    def should_continue(state: AgentState) -> str:
        """Determine if the agent should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)
        
        # Stop if max iterations reached
        if iteration >= config.max_iterations:
            print(f"  [router] -> end (max iterations)", flush=True)
            return "end"
        
        # If the last message has tool calls, continue to tool execution
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            print(f"  [router] -> tools", flush=True)
            return "tools"
        
        # Otherwise, end the conversation
        print(f"  [router] -> end (no tool calls)", flush=True)
        return "end"

    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile and return
    return workflow.compile()


def run_agent(user_message: str, config: AgentConfig | None = None) -> str:
    """Convenience function to run the agent with a single message."""
    graph = create_agent_graph(config)
    
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "iteration_count": 0,
    }
    
    result = graph.invoke(initial_state)
    
    # Get the last AI message
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    
    return "No response generated"


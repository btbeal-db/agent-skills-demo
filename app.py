"""Databricks App entry point using MLflow AgentServer.

This file serves the agent using MLflow's AgentServer, which provides:
- /invocations endpoint for querying the agent
- Built-in tracing and observability
- Automatic request routing and error handling
"""

import os
import sys
import uuid

# Add src to path so we can import the agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlflow.genai.agent_server import AgentServer, invoke, stream
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent.config import AgentConfig
from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize the agent lazily
_agent_graph = None
_agent_config = None


def get_agent():
    """Get or create the agent graph (singleton pattern)."""
    global _agent_graph, _agent_config
    
    if _agent_graph is None:
        # Get configuration from environment
        uc_volume_path = os.getenv(
            "UC_VOLUME_PATH", 
            "/Volumes/btbeal/docx_agent_skills_demo/created_docs"
        )
        model_endpoint = os.getenv("SERVING_ENDPOINT_NAME", "databricks-gpt-5-2")
        skills_dir = os.getenv("SKILLS_DIR", ".claude/skills")
        
        # Create the agent configuration
        # Use FE-EAST profile for local dev, empty for Databricks runtime
        databricks_profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "FE-EAST")
        
        _agent_config = AgentConfig(
            databricks_profile=databricks_profile,
            model_endpoint=model_endpoint,
            uc_volume_path=uc_volume_path,
            skills_directory=skills_dir,
            max_iterations=10,
        )
        
        # Create the agent graph
        _agent_graph = create_agent_graph(_agent_config)
        
        print(f"Agent initialized. Output path: {_agent_config.session_output_path}")
        print(f"Available skills: {_agent_config.available_skills}")
    
    return _agent_graph, _agent_config


def convert_to_langchain_messages(messages) -> list:
    """Convert OpenAI Responses API messages to LangChain messages.
    
    Messages can be dicts or pydantic Message objects.
    """
    lc_messages = []
    for msg in messages:
        # Handle both dict and pydantic object formats
        if hasattr(msg, "role"):
            role = msg.role
            content = msg.content if hasattr(msg, "content") else ""
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
        
        # Handle content that might be a list (Responses API format)
        if isinstance(content, list):
            # Extract text from content items
            text_parts = []
            for item in content:
                if hasattr(item, "text"):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
            content = " ".join(text_parts)
        
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
    
    return lc_messages


@invoke()
def handle_invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming agent invocation.
    
    Request is a ResponsesAgentRequest pydantic object with:
    - input: list of Message objects with role and content
    - custom_inputs: optional dict
    - context: optional ChatContext
    """
    graph, config = get_agent()
    
    # Get messages from request
    messages = request.input
    
    if not messages:
        return ResponsesAgentResponse(
            output=[{"id": str(uuid.uuid4()), "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "No input provided"}]}]
        )
    
    # Convert to LangChain format
    lc_messages = convert_to_langchain_messages(messages)
    
    # Run the agent
    try:
        result = graph.invoke({
            "messages": lc_messages,
            "iteration_count": 0,
        })
        
        # Extract the final response
        response_content = ""
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                response_content = message.content
                break
        
        # Return ResponsesAgentResponse
        msg_id = str(uuid.uuid4())
        return ResponsesAgentResponse(
            output=[
                {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant", 
                    "content": [{"type": "output_text", "text": response_content}]
                }
            ],
            custom_outputs={
                "session_id": config.session_id,
                "output_path": config.session_output_path,
            }
        )
        
    except Exception as e:
        return ResponsesAgentResponse(
            output=[
                {
                    "id": str(uuid.uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Error: {str(e)}"}]
                }
            ]
        )


@stream()
def handle_stream(request: ResponsesAgentRequest):
    """Handle streaming agent invocation.
    
    Yields chunks in OpenAI Responses API streaming format.
    """
    from mlflow.types.responses import ResponsesAgentStreamEvent
    
    graph, config = get_agent()
    
    # Get messages from request
    messages = request.input
    
    if not messages:
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "id": str(uuid.uuid4()),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "No input provided"}]
            }
        )
        return
    
    # Convert to LangChain format
    lc_messages = convert_to_langchain_messages(messages)
    
    # Run the agent (non-streaming, then simulate streaming output)
    try:
        result = graph.invoke({
            "messages": lc_messages,
            "iteration_count": 0,
        })
        
        # Extract the final response
        response_content = ""
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                response_content = message.content
                break
        
        # Stream the content in chunks
        item_id = "msg_" + config.session_id
        chunk_size = 50
        
        for i in range(0, len(response_content), chunk_size):
            chunk_text = response_content[i:i + chunk_size]
            yield ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                item_id=item_id,
                delta=chunk_text,
            )
        
        # Send the done event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_content}]
            }
        )
        
    except Exception as e:
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done", 
            item={
                "id": str(uuid.uuid4()),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"Error: {str(e)}"}]
            }
        )


# Create the server with ResponsesAgent type for proper validation
server = AgentServer(agent_type="ResponsesAgent")
app = server.app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

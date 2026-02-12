"""Main entry point for the Agent Skills Demo."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import AgentConfig, DocumentAgent


def main():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("Agent Skills Demo")
    print("LangGraph + Databricks + Claude Skills (Python-Only)")
    print("=" * 60)
    
    # Initialize config
    config = AgentConfig.from_env()
    
    print(f"\nDatabricks Profile: {config.databricks_profile}")
    print(f"Model Endpoint: {config.model_endpoint}")
    env_type = "Databricks" if config.is_running_in_databricks else "Local"
    print(f"Environment: {env_type}")
    print(f"Output Path: {config.session_output_path}")
    print(f"Available Skills: {', '.join(config.available_skills) or 'None'}")
    print("\nType 'quit' or 'exit' to stop.\n")
    print("-" * 60)
    
    # Create the graph once for reuse
    agent = DocumentAgent(config)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            print("\n--- Agent processing ---", flush=True)
            
            # Run the agent
            from langchain_core.messages import HumanMessage, AIMessage
            
            try:
                result = agent.invoke([HumanMessage(content=user_input)], iteration_count=0)
                
                print(f"--- Done ({len(result['messages'])} messages) ---\n")
                
                # Get the last AI message with content
                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage) and message.content:
                        print(f"Agent: {message.content}")
                        break
                else:
                    print("Agent: (No response with content)")
                    
            except Exception as invoke_error:
                import traceback
                print(f"\n[Error]: {invoke_error}")
                traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()

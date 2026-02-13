"""Tool implementations for the document agent workflow."""

from __future__ import annotations

import base64
import binascii
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from databricks_langchain import ChatDatabricks

from .config import AgentConfig

logger = logging.getLogger(__name__)


# Store the last execute_python result for use by save_to_volume
_last_execute_result: dict[str, str] = {}


def _get_workspace_client(config: AgentConfig) -> WorkspaceClient:
    """Create a workspace client using runtime identity or local profile."""
    if config.is_running_in_databricks:
        return WorkspaceClient()
    if config.databricks_profile:
        return WorkspaceClient(profile=config.databricks_profile)
    return WorkspaceClient()


def create_llm(config: AgentConfig) -> ChatDatabricks:
    """Create a ChatDatabricks LLM instance."""
    if config.is_running_in_databricks:
        workspace_client = WorkspaceClient()
        logger.info("Initializing ChatDatabricks with runtime identity")
    elif config.databricks_profile:
        workspace_client = WorkspaceClient(profile=config.databricks_profile)
        logger.info(
            "Initializing ChatDatabricks with Databricks CLI profile '%s'",
            config.databricks_profile,
        )
    else:
        workspace_client = WorkspaceClient()
        logger.info("Initializing ChatDatabricks with default local Databricks auth")

    return ChatDatabricks(
        endpoint=config.model_endpoint,
        workspace_client=workspace_client,
        temperature=0.1,
        # Note: responses_api=True enables Databricks' built-in code execution
        # which conflicts with our custom tools. Keep it False for custom tools.
    )


def build_skill_context(config: AgentConfig) -> str:
    """Build system context from available skills."""
    skills_info = []
    for skill_name in config.available_skills:
        metadata = config.load_skill_metadata(skill_name)
        if metadata:
            name = metadata.get("name", skill_name)
            description = metadata.get("description", "No description available")
            skills_info.append(f"- **{name}**: {description}")
    
    if not skills_info:
        return ""
    
    return f"""
## Available Skills

You have access to the following skills. When a user request matches a skill's description, 
you should use that skill's capabilities.

{chr(10).join(skills_info)}

To use a skill, first load its instructions, then use the provided Python functions.
Skills are located at: {config.skills_directory}

## File Output

When you create documents, save them using the `save_to_volume` tool.
Files will be saved to: {config.session_output_path}
"""


def load_skill_instructions(config: AgentConfig, skill_name: str) -> str:
    """Load the full SKILL.md instructions for a skill."""
    skill_path = config.get_skill_path(skill_name) / "SKILL.md"
    if not skill_path.exists():
        return ""
    
    content = skill_path.read_text()
    
    # Remove YAML frontmatter
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            content = content[end_idx + 3:].strip()
    
    return content


# =============================================================================
# Unity Catalog Volume Operations
# =============================================================================

def save_to_uc_volume(
    config: AgentConfig,
    filename: str,
    content: bytes | str,
    content_type: str = "application/octet-stream"
) -> dict[str, Any]:
    """Save a file to the Unity Catalog Volume.
    
    Args:
        config: Agent configuration with UC Volume path
        filename: Name for the file (e.g., "report.docx")
        content: File content as bytes or base64-encoded string
        content_type: MIME type of the content
        
    Returns:
        dict with success status and file path or error
    """
    try:
        # Handle bytes and base64-encoded content safely.
        if isinstance(content, str):
            try:
                content = base64.b64decode(content, validate=True)
            except (binascii.Error, ValueError) as exc:
                return {
                    "success": False,
                    "error": f"Invalid content_base64 payload: {exc}",
                    "path": None,
                }
        
        # Construct full path
        output_dir = config.session_output_path
        full_path = f"{output_dir}/{filename}"

        if full_path.startswith("/Volumes/"):
            # In Databricks Apps, use Files API for UC volume I/O.
            workspace_client = _get_workspace_client(config)
            try:
                workspace_client.files.create_directory(output_dir)
            except Exception:
                # Directory may already exist.
                pass
            workspace_client.files.upload(full_path, BytesIO(content), overwrite=True)
        else:
            # Local development fallback.
            os.makedirs(output_dir, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(content)
        
        return {
            "success": True,
            "path": full_path,
            "message": f"File saved successfully to {full_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": None
        }


def read_from_uc_volume(
    config: AgentConfig,
    filename: str,
    return_base64: bool = False
) -> dict[str, Any]:
    """Read a file from the Unity Catalog Volume.
    
    Args:
        config: Agent configuration with UC Volume path
        filename: Name of the file to read
        return_base64: If True, return content as base64 string
        
    Returns:
        dict with success status and content or error
    """
    try:
        full_path = f"{config.session_output_path}/{filename}"

        if full_path.startswith("/Volumes/"):
            workspace_client = _get_workspace_client(config)
            response = workspace_client.files.download(full_path)
            content = response.contents.read() if response.contents is not None else b""
        else:
            with open(full_path, "rb") as f:
                content = f.read()
        
        if return_base64:
            content = base64.b64encode(content).decode("utf-8")
        
        return {
            "success": True,
            "path": full_path,
            "content": content,
            "size_bytes": len(content) if isinstance(content, bytes) else len(base64.b64decode(content))
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"File not found: {filename}",
            "path": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "path": None
        }


def list_uc_volume_files(config: AgentConfig) -> dict[str, Any]:
    """List files in the session's UC Volume directory.
    
    Returns:
        dict with success status and list of files or error
    """
    try:
        output_dir = config.session_output_path

        if output_dir.startswith("/Volumes/"):
            workspace_client = _get_workspace_client(config)
            files = []
            for entry in workspace_client.files.list_directory_contents(output_dir):
                if not entry.is_directory:
                    files.append(
                        {
                            "name": entry.name,
                            "path": entry.path,
                            "size_bytes": entry.file_size,
                            "modified_time": entry.last_modified,
                        }
                    )
            return {
                "success": True,
                "files": files,
                "path": output_dir,
                "count": len(files),
            }

        if not os.path.exists(output_dir):
            return {
                "success": True,
                "files": [],
                "path": output_dir,
                "message": "Directory is empty or does not exist yet"
            }

        files = []
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                files.append({
                    "name": f,
                    "size_bytes": os.path.getsize(file_path),
                    "path": file_path
                })
        
        return {
            "success": True,
            "files": files,
            "path": output_dir
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files": []
        }


# =============================================================================
# Python-based Document Operations (replaces shell/npm execution)
# =============================================================================

def execute_python_code(code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute Python code in a controlled environment.
    
    This is used for skill-generated Python code (e.g., python-docx operations).
    
    Args:
        code: Python code to execute
        context: Optional dict of variables to inject into the execution context
        
    Returns:
        dict with success status, output, and any returned values
    """
    try:
        # Create execution context with common imports
        exec_globals = {
            "__builtins__": __builtins__,
            "BytesIO": BytesIO,
            "Path": Path,
        }
        
        # Add python-docx imports
        try:
            from docx import Document
            from docx.shared import Inches, Pt, Cm
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.enum.style import WD_STYLE_TYPE
            exec_globals.update({
                "Document": Document,
                "Inches": Inches,
                "Pt": Pt,
                "Cm": Cm,
                "WD_ALIGN_PARAGRAPH": WD_ALIGN_PARAGRAPH,
                "WD_STYLE_TYPE": WD_STYLE_TYPE,
            })
        except ImportError:
            pass  # python-docx not installed
        
        # Add user context
        if context:
            exec_globals.update(context)
        
        exec_locals: dict[str, Any] = {}
        
        # Execute the code
        exec(code, exec_globals, exec_locals)
        
        # Look for a result variable
        result = exec_locals.get("result", exec_locals.get("output", None))
        
        return {
            "success": True,
            "result": result,
            "locals": {k: str(v)[:200] for k, v in exec_locals.items() if not k.startswith("_")}
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }


# =============================================================================
# Tool Definitions
# =============================================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full instructions for a skill to understand how to use it. Always call this before using a skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill to load (e.g., 'docx')"
                    }
                },
                "required": ["skill_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code for document operations. Use python-docx for Word documents. The code should set a 'result' variable with any output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Has access to: Document, Inches, Pt, Cm, WD_ALIGN_PARAGRAPH, BytesIO, Path"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_volume",
            "description": "Save a file to the Unity Catalog Volume. If content_base64 is omitted, automatically uses the result from the last execute_python call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name for the file (e.g., 'report.docx', 'chart.png')"
                    },
                    "content_base64": {
                        "type": "string",
                        "description": "File content as base64 string. Optional - if omitted, uses the 'result' from the last execute_python call."
                    },
                    "content_type": {
                        "type": "string",
                        "description": "MIME type (optional)"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_from_volume",
            "description": "Read a file from the Unity Catalog Volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_volume_files",
            "description": "List all files in the current session's output directory.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def handle_tool_call(
    config: AgentConfig,
    tool_name: str,
    tool_args: dict[str, Any]
) -> str:
    """Handle a tool call from the LLM."""
    
    if tool_name == "load_skill":
        skill_name = tool_args.get("skill_name", "")
        instructions = load_skill_instructions(config, skill_name)
        if instructions:
            return f"## {skill_name} Skill Instructions\n\n{instructions}"
        return f"Skill '{skill_name}' not found. Available skills: {config.available_skills}"
    
    elif tool_name == "execute_python":
        code = tool_args.get("code", "")
        result = execute_python_code(code)
        if result["success"]:
            output = "Code executed successfully."
            
            # Handle the result - if it's base64 content, don't return the full thing
            if result["result"] is not None:
                result_value = result["result"]
                if isinstance(result_value, bytes):
                    # Preserve binary payloads (e.g., .docx) by storing base64 for save_to_volume.
                    encoded = base64.b64encode(result_value).decode("utf-8")
                    _last_execute_result["content"] = encoded
                    output += f"\nResult: <{len(result_value)} bytes of binary data>"
                    output += "\n\nBinary result stored as base64. Use save_to_volume to save it."
                else:
                    result_str = str(result_value)
                    if len(result_str) > 500:
                        output += f"\nResult: <{len(result_str)} characters of data>"
                        output += "\n\nThe 'result' variable contains large data. Use save_to_volume to save it."
                        _last_execute_result["content"] = result_str
                    else:
                        output += f"\nResult: {result_str}"
            
            if result["locals"]:
                # Filter out large values from locals display
                filtered_locals = {
                    k: (v[:100] + "...") if len(str(v)) > 100 else v 
                    for k, v in result["locals"].items()
                }
                output += f"\nVariables: {filtered_locals}"
            return output
        return f"Code execution failed: {result['error']}"
    
    elif tool_name == "save_to_volume":
        # Get content - use provided content_base64, or fall back to last execute_python result
        content = tool_args.get("content_base64", "")
        if not content and _last_execute_result.get("content"):
            content = _last_execute_result["content"]
            _last_execute_result.clear()  # Clear after use
        
        result = save_to_uc_volume(
            config,
            tool_args.get("filename", "output.bin"),
            content,
            tool_args.get("content_type", "application/octet-stream")
        )
        if result["success"]:
            return f"✓ File saved: {result['path']}"
        return f"✗ Failed to save file: {result['error']}"
    
    elif tool_name == "read_from_volume":
        result = read_from_uc_volume(
            config,
            tool_args.get("filename", ""),
            return_base64=True
        )
        if result["success"]:
            return f"File read successfully ({result['size_bytes']} bytes). Content available as base64."
        return f"Failed to read file: {result['error']}"
    
    elif tool_name == "list_volume_files":
        result = list_uc_volume_files(config)
        if result["success"]:
            if not result["files"]:
                return f"No files in {result['path']}"
            file_list = "\n".join([f"- {f['name']} ({f['size_bytes']} bytes)" for f in result["files"]])
            return f"Files in {result['path']}:\n{file_list}"
        return f"Failed to list files: {result['error']}"
    
    return f"Unknown tool: {tool_name}"

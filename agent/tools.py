"""Tool implementations for the document agent workflow."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient

from .config import AgentConfig

logger = logging.getLogger(__name__)


class ToolContext:
    """Per-request context for tool execution state.

    This holds mutable state that needs to persist across tool calls within
    a single agent invocation, but must be isolated between concurrent requests.
    """

    def __init__(self):
        self.last_execute_result: dict[str, str] = {}
        self.last_read_from_volume: dict[str, Any] = {}
        self.bash_working_directory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_execute_result": self.last_execute_result,
            "last_read_from_volume": self.last_read_from_volume,
            "bash_working_directory": self.bash_working_directory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolContext":
        ctx = cls()
        ctx.last_execute_result = data.get("last_execute_result") or {}
        ctx.last_read_from_volume = data.get("last_read_from_volume") or {}
        ctx.bash_working_directory = data.get("bash_working_directory")
        return ctx


def _get_workspace_client(config: AgentConfig) -> WorkspaceClient:
    """Create a workspace client using runtime identity or local profile."""
    if config.is_running_in_databricks:
        return WorkspaceClient()
    if config.databricks_profile:
        return WorkspaceClient(profile=config.databricks_profile)
    return WorkspaceClient()


def build_skill_context(config: AgentConfig) -> str:
    """Build system context from skill metadata (without loading full content)."""
    skill_metadata_list = get_skill_metadata_list(config)
    if not skill_metadata_list:
        return ""

    skills_info = [
        f"- **{skill['name']}**: {skill['description']}" for skill in skill_metadata_list
    ]
    return f"""
## Available Skills

You have access to the following skills:

{chr(10).join(skills_info)}

Use the `load_skill` tool when you need the full instructions for a specific skill.
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


def get_skill_metadata_list(config: AgentConfig) -> list[dict[str, str]]:
    """Get skill metadata without loading full content (efficient for listing)."""
    skills: list[dict[str, str]] = []
    for skill_id in config.available_skills:
        metadata = config.load_skill_metadata(skill_id)
        skills.append(
            {
                "id": skill_id,
                "name": metadata.get("name", skill_id),
                "description": metadata.get("description", "No description available"),
                "path": str(config.get_skill_path(skill_id)),
            }
        )
    return skills


def list_skills(config: AgentConfig) -> dict[str, Any]:
    """Enumerate available skills (metadata only, no content loading)."""
    skills = get_skill_metadata_list(config)
    return {
        "skills": skills,
        "count": len(skills),
        "skill_directories": [str(path) for path in config.skill_directories],
    }


# =============================================================================
# Unity Catalog Volume Operations
# =============================================================================

def save_to_uc_volume(
    config: AgentConfig,
    filename: str,
    content: bytes | str,
    content_type: str = "application/octet-stream"
) -> dict[str, Any]:
    """Save a file to the Unity Catalog Volume."""
    try:
        if isinstance(content, str):
            try:
                content = base64.b64decode(content, validate=True)
            except (binascii.Error, ValueError) as exc:
                return {
                    "success": False,
                    "error": f"Invalid content_base64 payload: {exc}",
                    "path": None,
                }

        output_dir = config.session_output_path
        full_path = f"{output_dir}/{filename}"

        if full_path.startswith("/Volumes/"):
            workspace_client = _get_workspace_client(config)
            try:
                workspace_client.files.create_directory(output_dir)
            except Exception:
                pass
            workspace_client.files.upload(full_path, BytesIO(content), overwrite=True)
        else:
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

    Accepts either a bare filename (resolved under the current session folder)
    or an absolute path (used as-is, letting the SDK or OS enforce access).
    """
    try:
        # Absolute paths are used directly; relative names are scoped to the session folder.
        if filename.startswith("/"):
            full_path = filename
        else:
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


def list_uc_volume_files(config: AgentConfig, path: str | None = None) -> dict[str, Any]:
    """List files in a UC Volume directory, recursively.

    Defaults to the current session output folder. Pass an absolute path to
    browse any other directory in the volume.
    """
    try:
        output_dir = path.rstrip("/") if path and path.startswith("/") else config.session_output_path

        if output_dir.startswith("/Volumes/"):
            workspace_client = _get_workspace_client(config)
            files: list[dict[str, Any]] = []

            def _collect_uc(dir_path: str) -> None:
                for entry in workspace_client.files.list_directory_contents(dir_path):
                    if entry.is_directory:
                        _collect_uc(entry.path)
                    else:
                        files.append({
                            "name": entry.name,
                            "path": entry.path,
                            "size_bytes": entry.file_size,
                            "modified_time": entry.last_modified,
                        })

            _collect_uc(output_dir)
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
        for root, _dirs, filenames in os.walk(output_dir):
            for f in filenames:
                file_path = os.path.join(root, f)
                files.append({
                    "name": f,
                    "size_bytes": os.path.getsize(file_path),
                    "path": file_path,
                })

        return {
            "success": True,
            "files": files,
            "path": output_dir,
            "count": len(files),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "files": []
        }


def _safe_relative_path(path_value: str) -> str | None:
    """Validate and normalize a relative file path."""
    candidate = Path(path_value.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate.as_posix()


def copy_file_to_current_session(
    config: AgentConfig,
    source_path: str | None = None,
    source_session_id: str | None = None,
    filename: str | None = None,
    target_filename: str | None = None,
) -> dict[str, Any]:
    """Copy a file from another session into the current session directory."""
    try:
        resolved_source_path = ""
        if source_path and source_path.strip():
            resolved_source_path = source_path.strip()
        elif source_session_id and filename:
            safe_filename = _safe_relative_path(filename)
            if not safe_filename:
                return {
                    "success": False,
                    "error": "filename must be a safe relative path",
                    "source_path": None,
                    "target_path": None,
                }
            if config.session_output_path.startswith("/Volumes/"):
                resolved_source_path = f"{config.uc_volume_path}/{source_session_id.strip()}/{safe_filename}"
            else:
                resolved_source_path = f"{config.local_output_dir}/{source_session_id.strip()}/{safe_filename}"
        else:
            return {
                "success": False,
                "error": "Provide either source_path OR (source_session_id and filename)",
                "source_path": None,
                "target_path": None,
            }

        if config.session_output_path.startswith("/Volumes/"):
            if not resolved_source_path.startswith(config.uc_volume_path + "/"):
                return {
                    "success": False,
                    "error": f"source_path must be under {config.uc_volume_path}",
                    "source_path": resolved_source_path,
                    "target_path": None,
                }

            default_name = Path(resolved_source_path).name
            desired_target = target_filename.strip() if target_filename else default_name
            safe_target = _safe_relative_path(desired_target)
            if not safe_target:
                return {
                    "success": False,
                    "error": "target_filename must be a safe relative path",
                    "source_path": resolved_source_path,
                    "target_path": None,
                }
            target_path = f"{config.session_output_path}/{safe_target}"
            target_dir = str(Path(target_path).parent)

            workspace_client = _get_workspace_client(config)
            response = workspace_client.files.download(resolved_source_path)
            content = response.contents.read() if response.contents is not None else b""
            workspace_client.files.create_directory(target_dir)
            workspace_client.files.upload(target_path, BytesIO(content), overwrite=True)

            return {
                "success": True,
                "source_path": resolved_source_path,
                "target_path": target_path,
                "message": f"Copied file into current session: {target_path}",
            }

        # Local development fallback
        source_local = Path(resolved_source_path)
        if not source_local.is_absolute():
            source_local = Path.cwd() / source_local
        if not source_local.exists() or not source_local.is_file():
            return {
                "success": False,
                "error": f"Source file not found: {source_local}",
                "source_path": str(source_local),
                "target_path": None,
            }

        default_name = source_local.name
        desired_target = target_filename.strip() if target_filename else default_name
        safe_target = _safe_relative_path(desired_target)
        if not safe_target:
            return {
                "success": False,
                "error": "target_filename must be a safe relative path",
                "source_path": str(source_local),
                "target_path": None,
            }
        target_local = Path(config.session_output_path) / safe_target
        target_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_local, target_local)

        return {
            "success": True,
            "source_path": str(source_local),
            "target_path": str(target_local),
            "message": f"Copied file into current session: {target_local}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "source_path": source_path,
            "target_path": None,
        }


def execute_python_code(code: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute Python code in a controlled environment."""
    try:
        # Use a single namespace dict so imports, assignments, and function
        # definitions all share the same scope (avoids the exec() split-scope
        # bug where functions can't see names imported into exec_locals).
        exec_namespace: dict[str, Any] = {"__builtins__": __builtins__}
        if context:
            exec_namespace.update(context)

        exec(code, exec_namespace)

        result = exec_namespace.get("result", exec_namespace.get("output", None))

        return {
            "success": True,
            "result": result,
            "locals": {k: str(v)[:200] for k, v in exec_namespace.items() if not k.startswith("_")}
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }


def execute_bash_command(
    command: str,
    working_directory: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Execute a bash command in a subprocess."""
    try:
        if working_directory is None:
            working_directory = tempfile.mkdtemp(prefix="agent_bash_")

        os.makedirs(working_directory, exist_ok=True)

        # Ensure 'python' resolves even if only 'python3' is on the PATH.
        # Use a wrapper script (not a symlink) so the real executable's venv
        # detection via pyvenv.cfg continues to work.
        python_bin_dir = os.path.join(working_directory, ".bin")
        os.makedirs(python_bin_dir, exist_ok=True)
        python_wrapper = os.path.join(python_bin_dir, "python")
        if not os.path.lexists(python_wrapper):
            real_python = os.path.abspath(sys.executable)
            with open(python_wrapper, "w") as f:
                f.write(f"#!/bin/sh\nexec {real_python} \"$@\"\n")
            os.chmod(python_wrapper, 0o755)

        env = os.environ.copy()
        env["PATH"] = f"{python_bin_dir}:{env.get('PATH', '')}"

        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            "returncode": result.returncode,
            "working_directory": working_directory,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "working_directory": working_directory or "",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "working_directory": working_directory or "",
        }


# =============================================================================
# Tool Definitions
# =============================================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Scan the skills directories and return all available skills.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full instructions for a skill. Always call this before using a skill.",
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
            "description": "Execute Python code for document operations. Set a 'result' variable with any output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. If read_from_volume was used, source_doc_bytes/source_doc_base64/source_doc_filename/source_doc_path are available."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash shell command. Use for running Python helper scripts (unpack, pack, validate, comment, accept_changes) and other shell operations needed by skills. The working directory persists across calls within a session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120)"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_volume",
            "description": "Save a file to the Unity Catalog Volume. If content_base64 is omitted, uses result from last execute_python.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name for the file"},
                    "content_base64": {"type": "string", "description": "File content as base64 (optional)"},
                    "content_type": {"type": "string", "description": "MIME type (optional)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_from_volume",
            "description": "Read a file from the Unity Catalog Volume. Accepts either a bare filename (resolved relative to the current session folder) or a full absolute path (e.g. /Volumes/catalog/schema/volume/folder/file.pdf). Use a full path when the user provides one or the file lives outside the current session folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Bare filename relative to the session folder, or a full absolute path to any file in the volume."}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_to_session",
            "description": "Copy a file from another session into the current session folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string", "description": "Absolute path to source file"},
                    "source_session_id": {"type": "string", "description": "Source session ID"},
                    "filename": {"type": "string", "description": "Filename under source_session_id"},
                    "target_filename": {"type": "string", "description": "Target filename (optional)"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_volume_files",
            "description": "List files in a Unity Catalog Volume directory. Defaults to the current session folder. Pass an absolute path to browse any other directory (e.g. /Volumes/catalog/schema/volume/some/folder). Use this to locate files when the user provides a partial path or you need to find a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to list (e.g. /Volumes/catalog/schema/volume/folder). Omit to list the current session folder."
                    }
                }
            }
        }
    }
]


def _looks_like_base64(s: str) -> bool:
    """Return True if the string is almost certainly base64-encoded binary data.

    Samples the first 200 non-whitespace characters. Real text always contains
    spaces, punctuation, or characters outside the base64 alphabet; encoded
    binary never does.
    """
    if len(s) < 128:
        return False
    sample = s[:200].replace("\n", "").replace("\r", "")
    return all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in sample)


def handle_tool_call(
    config: AgentConfig,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_context: ToolContext | None = None,
) -> str:
    """Handle a tool call from the LLM."""
    if tool_context is None:
        tool_context = ToolContext()

    if tool_name == "list_skills":
        result = list_skills(config)
        if not result["skills"]:
            return f"No skills found in: {', '.join(result['skill_directories'])}"
        lines = [f"- {s['name']} ({s['id']}): {s['description']}" for s in result["skills"]]
        return "Available skills:\n" + "\n".join(lines)

    elif tool_name == "load_skill":
        skill_name = tool_args.get("skill_name", "")
        skill_lookup: dict[str, str] = {}
        for skill_id in config.available_skills:
            metadata = config.load_skill_metadata(skill_id)
            skill_lookup[skill_id] = skill_id
            skill_lookup[metadata.get("name", skill_id)] = skill_id

        if skill_name in skill_lookup:
            resolved_id = skill_lookup[skill_name]
            skill_dir = config.get_skill_path(resolved_id)
            content = load_skill_instructions(config, resolved_id)
            return (
                f"Loaded skill: {resolved_id}\n"
                f"Skill directory: {skill_dir}\n"
                f"Scripts path: {skill_dir}/scripts\n\n"
                f"{content}"
            )

        return f"Skill '{skill_name}' not found. Available: {', '.join(config.available_skills)}"

    elif tool_name == "execute_python":
        code = tool_args.get("code", "")
        exec_context: dict[str, Any] = {}
        if tool_context.last_read_from_volume.get("content_base64"):
            try:
                source_bytes = base64.b64decode(tool_context.last_read_from_volume["content_base64"])
                exec_context.update({
                    "source_doc_bytes": source_bytes,
                    "source_doc_base64": tool_context.last_read_from_volume["content_base64"],
                    "source_doc_filename": tool_context.last_read_from_volume.get("filename", ""),
                    "source_doc_path": tool_context.last_read_from_volume.get("path", ""),
                })
            except Exception:
                pass

        result = execute_python_code(code, context=exec_context if exec_context else None)
        if result["success"]:
            output = "Code executed successfully."
            if result["result"] is not None:
                result_value = result["result"]
                if isinstance(result_value, bytes):
                    encoded = base64.b64encode(result_value).decode("utf-8")
                    tool_context.last_execute_result["content"] = encoded
                    output += f"\nResult: <{len(result_value)} bytes>. Use save_to_volume to save."
                else:
                    result_str = str(result_value)
                    # Always stash so save_to_volume can access it if needed.
                    tool_context.last_execute_result["content"] = result_str
                    if _looks_like_base64(result_str):
                        # Encoded binary (e.g. a processed document). Don't show
                        # the raw base64 — it's useless tokens. Just signal to save.
                        output += f"\nResult: <{len(result_str)} chars, base64-encoded binary>. Call save_to_volume to persist."
                    elif len(result_str) > 8000:
                        # Very long plain text — show a truncated preview.
                        output += (
                            f"\nResult ({len(result_str)} chars, showing first 8000):\n"
                            f"{result_str[:8000]}\n...(truncated)"
                        )
                    else:
                        output += f"\nResult:\n{result_str}"
            return output
        return f"Code execution failed: {result['error']}"

    elif tool_name == "execute_bash":
        command = tool_args.get("command", "")
        timeout = tool_args.get("timeout", 120)

        if tool_context.bash_working_directory is None:
            tool_context.bash_working_directory = tempfile.mkdtemp(prefix="agent_bash_")

        result = execute_bash_command(
            command,
            working_directory=tool_context.bash_working_directory,
            timeout=timeout,
        )

        tool_context.bash_working_directory = result.get(
            "working_directory", tool_context.bash_working_directory
        )

        output_parts = []
        if result["success"]:
            output_parts.append("Command executed successfully.")
        else:
            output_parts.append(f"Command failed (exit code {result['returncode']}).")
            if result.get("error"):
                output_parts.append(f"Error: {result['error']}")

        if result.get("stdout"):
            output_parts.append(f"stdout:\n{result['stdout']}")
        if result.get("stderr"):
            output_parts.append(f"stderr:\n{result['stderr']}")

        output_parts.append(f"Working directory: {result['working_directory']}")
        return "\n".join(output_parts)

    elif tool_name == "save_to_volume":
        content = tool_args.get("content_base64", "")
        if not content and tool_context.last_execute_result.get("content"):
            content = tool_context.last_execute_result["content"]
            tool_context.last_execute_result.clear()

        result = save_to_uc_volume(
            config,
            tool_args.get("filename", "output.bin"),
            content,
            tool_args.get("content_type", "application/octet-stream")
        )
        if result["success"]:
            return f"File saved: {result['path']}"
        return f"Failed to save: {result['error']}"

    elif tool_name == "read_from_volume":
        result = read_from_uc_volume(config, tool_args.get("filename", ""), return_base64=True)
        if result["success"]:
            tool_context.last_read_from_volume.clear()
            tool_context.last_read_from_volume.update({
                "filename": tool_args.get("filename", ""),
                "path": result.get("path", ""),
                "content_base64": result.get("content", ""),
                "size_bytes": result.get("size_bytes", 0),
            })
            return f"File read ({result['size_bytes']} bytes). Available as source_doc_bytes/source_doc_base64."
        return f"Failed to read: {result['error']}"

    elif tool_name == "copy_to_session":
        result = copy_file_to_current_session(
            config,
            source_path=tool_args.get("source_path"),
            source_session_id=tool_args.get("source_session_id"),
            filename=tool_args.get("filename"),
            target_filename=tool_args.get("target_filename"),
        )
        if result["success"]:
            return f"Copied: {result['target_path']}"
        return f"Failed to copy: {result['error']}"

    elif tool_name == "list_volume_files":
        result = list_uc_volume_files(config, path=tool_args.get("path"))
        if result["success"]:
            if not result["files"]:
                return f"No files in {result['path']}"
            # Show the full path so the agent can pass it directly to read_from_volume.
            file_list = "\n".join([f"- {f['path']} ({f['size_bytes']} bytes)" for f in result["files"]])
            return f"Files in {result['path']} ({result['count']} total):\n{file_list}"
        return f"Failed to list: {result['error']}"

    return f"Unknown tool: {tool_name}"

"""Configuration for the Agent Skills demo."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SKILLS_DIR = "./.claude/skills"


@dataclass
class AgentConfig:
    """Configuration for the Databricks agent with Claude Skills."""
    databricks_profile: str = ""
    model_endpoint: str = "databricks-gpt-5-2"
    uc_volume_path: str = "/Volumes/hls_amer_catalog/appeals-review/created_docs"
    local_output_dir: str = "./output"
    output_mode: str = "auto"  # auto | uc_volume | local
    skills_directory: Path = field(default_factory=lambda: Path(DEFAULT_SKILLS_DIR))
    def __setattr__(self, name: str, value):
        """Convert skills_directory to Path if string is passed."""
        if name == "skills_directory" and isinstance(value, str):
            value = Path(value)
        super().__setattr__(name, value)

    # Max agent iterations
    max_iterations: int = 10

    # Per-LLM-call HTTP timeout in seconds. Prevents the OpenAI client from
    # retrying indefinitely when the endpoint is slow or unreachable.
    llm_timeout: int = 120

    # Session ID for organizing outputs (auto-generated if not provided)
    session_id: Optional[str] = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _normalize_uc_volume_path(value: str) -> str:
        """Accept either /Volumes path or catalog.schema.volume and normalize."""
        cleaned = value.strip()
        if not cleaned:
            return cleaned
        if cleaned.startswith("/Volumes/"):
            return cleaned
        if "." in cleaned and "/" not in cleaned and len(cleaned.split(".")) == 3:
            return "/Volumes/" + cleaned.replace(".", "/")
        return cleaned

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build config from environment variables.

        Supports both AGENT_* variables and legacy names used by app.yaml.
        """
        return cls(
            databricks_profile=os.getenv("DATABRICKS_CONFIG_PROFILE", ""),
            model_endpoint=os.getenv(
                "AGENT_MODEL_ENDPOINT",
                os.getenv("SERVING_ENDPOINT_NAME", "databricks-gpt-5-2"),
            ),
            uc_volume_path=cls._normalize_uc_volume_path(
                os.getenv(
                    "AGENT_UC_VOLUME_PATH",
                    os.getenv(
                        "UC_VOLUME_PATH", "/Volumes/hls_amer_catalog/appeals-review/created_docs"
                    ),
                )
            ),
            local_output_dir=os.getenv("AGENT_LOCAL_OUTPUT_DIR", "./output"),
            output_mode=os.getenv("AGENT_OUTPUT_MODE", "auto").strip().lower(),
            skills_directory=os.getenv(
                "AGENT_SKILLS_DIR",
                os.getenv("SKILLS_DIR", DEFAULT_SKILLS_DIR),
            ),
            max_iterations=cls._env_int("AGENT_MAX_ITERATIONS", 10),
            llm_timeout=cls._env_int("AGENT_LLM_TIMEOUT", 120),
            session_id=os.getenv("AGENT_SESSION_ID"),
        )

    def __post_init__(self):
        """Validate and normalize configuration values."""
        if self.max_iterations < 1:
            self.max_iterations = 1

        if self.output_mode not in {"auto", "uc_volume", "local"}:
            self.output_mode = "auto"

        # Resolve relative skills path against project root so discovery is not CWD-dependent.
        if not self.skills_directory.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            self.skills_directory = (project_root / self.skills_directory).resolve()

        # Generate session ID if not provided
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())[:8]

    @property
    def is_running_in_databricks(self) -> bool:
        """Check if we're running inside Databricks runtime."""
        # Databricks runtime sets DATABRICKS_RUNTIME_VERSION
        # Model Serving/Apps set additional env vars depending on execution environment.
        return (
            "DATABRICKS_RUNTIME_VERSION" in os.environ
            or "IS_SERVERLESS" in os.environ
            or "DATABRICKS_APP_NAME" in os.environ
            or "DATABRICKS_APP_ID" in os.environ
        )

    @property
    def session_output_path(self) -> str:
        """Get the output path for this session.

        Uses UC Volume path in Databricks, local directory otherwise.
        """
        if self.output_mode == "uc_volume":
            return f"{self.uc_volume_path}/{self.session_id}"
        if self.output_mode == "local":
            return f"{self.local_output_dir}/{self.session_id}"
        if self.is_running_in_databricks:
            return f"{self.uc_volume_path}/{self.session_id}"
        return f"{self.local_output_dir}/{self.session_id}"

    @property
    def skill_directories(self) -> list[Path]:
        """Return configured skill directory."""
        return [self.skills_directory]

    @property
    def available_skills(self) -> list[str]:
        """List available skills discovered across configured skill directories."""
        skills: list[str] = []
        seen: set[str] = set()
        for skills_dir in self.skill_directories:
            if not skills_dir.exists():
                continue
            for entry in sorted(skills_dir.iterdir()):
                if not entry.is_dir() or not (entry / "SKILL.md").exists():
                    continue
                if entry.name in seen:
                    continue
                seen.add(entry.name)
                skills.append(entry.name)
        return skills

    def get_skill_path(self, skill_name: str) -> Path:
        """Get the path to a specific skill from known skill directories."""
        for skills_dir in self.skill_directories:
            candidate = skills_dir / skill_name
            if (candidate / "SKILL.md").exists():
                return candidate
        return self.skills_directory / skill_name

    def load_skill_metadata(self, skill_name: str) -> dict:
        """Load skill metadata from SKILL.md frontmatter."""
        skill_path = self.get_skill_path(skill_name) / "SKILL.md"
        if not skill_path.exists():
            return {}

        content = skill_path.read_text()

        # Parse YAML frontmatter using pyyaml for robust parsing
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                frontmatter = content[3:end_idx].strip()
                try:
                    metadata = yaml.safe_load(frontmatter)
                    return metadata if isinstance(metadata, dict) else {}
                except yaml.YAMLError:
                    return {}
        return {}

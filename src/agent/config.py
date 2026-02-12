"""Configuration for the Agent Skills demo."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the Databricks agent with Claude Skills."""
    
    # Databricks configuration
    databricks_profile: str = "FE-EAST"
    model_endpoint: str = "databricks-gpt-5-2"
    
    # Unity Catalog Volume configuration for file output (used in Databricks)
    uc_volume_path: str = "/Volumes/btbeal/docx_agent_skills_demo/created_docs"
    
    # Local output directory (used when running outside Databricks)
    local_output_dir: str = "./output"
    
    # Skills configuration - accepts str or Path
    skills_directory: Path = field(default_factory=lambda: Path(".claude/skills"))
    
    def __setattr__(self, name: str, value):
        """Convert skills_directory to Path if string is passed."""
        if name == "skills_directory" and isinstance(value, str):
            value = Path(value)
        super().__setattr__(name, value)
    
    # Agent configuration
    max_iterations: int = 10
    
    # Session ID for organizing outputs (auto-generated if not provided)
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Set up Databricks SDK profile via environment variable."""
        # Only set profile if provided and not running in Databricks
        # (Databricks runtime uses default authentication)
        if self.databricks_profile and not self.is_running_in_databricks:
            os.environ["DATABRICKS_CONFIG_PROFILE"] = self.databricks_profile
        
        # Generate session ID if not provided
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())[:8]
    
    @property
    def is_running_in_databricks(self) -> bool:
        """Check if we're running inside Databricks runtime."""
        # Databricks runtime sets DATABRICKS_RUNTIME_VERSION
        # Model Serving also sets specific env vars
        return (
            "DATABRICKS_RUNTIME_VERSION" in os.environ
            or "IS_SERVERLESS" in os.environ  # Model Serving
        )
    
    @property
    def session_output_path(self) -> str:
        """Get the output path for this session.
        
        Uses UC Volume path in Databricks, local directory otherwise.
        """
        if self.is_running_in_databricks:
            return f"{self.uc_volume_path}/{self.session_id}"
        else:
            return f"{self.local_output_dir}/{self.session_id}"
    
    @property
    def available_skills(self) -> list[str]:
        """List available skills in the skills directory."""
        if not self.skills_directory.exists():
            return []
        return [
            d.name for d in self.skills_directory.iterdir() 
            if d.is_dir() and (d / "SKILL.md").exists()
        ]
    
    def get_skill_path(self, skill_name: str) -> Path:
        """Get the path to a specific skill."""
        return self.skills_directory / skill_name
    
    def load_skill_metadata(self, skill_name: str) -> dict:
        """Load skill metadata from SKILL.md frontmatter."""
        skill_path = self.get_skill_path(skill_name) / "SKILL.md"
        if not skill_path.exists():
            return {}
        
        content = skill_path.read_text()
        
        # Parse YAML frontmatter
        if content.startswith("---"):
            end_idx = content.find("---", 3)
            if end_idx != -1:
                frontmatter = content[3:end_idx].strip()
                metadata = {}
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"')
                return metadata
        return {}


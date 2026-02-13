"""Start script for the MLflow Agent Server."""

# Import handlers so @invoke and @stream are registered.
import app  # noqa: F401

import logging
import os

import mlflow
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

logger = logging.getLogger(__name__)

mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_ID"))

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Optional helper to link traces to git revisions.
setup_mlflow_git_based_version_tracking()


def main() -> None:
    """Run the Agent Server locally."""
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()


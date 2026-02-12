"""Start script for the MLflow Agent Server."""

# Import handlers so @invoke and @stream are registered.
import app  # noqa: F401

import logging
import os

import mlflow
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

logger = logging.getLogger(__name__)


def configure_mlflow_tracking() -> None:
    """Configure MLflow tracking destination and experiment for traces."""
    is_databricks_runtime = (
        "DATABRICKS_RUNTIME_VERSION" in os.environ or "IS_SERVERLESS" in os.environ
    )
    configured_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "")

    if configured_tracking_uri:
        mlflow.set_tracking_uri(configured_tracking_uri)
        logger.info("MLflow tracking URI set from MLFLOW_TRACKING_URI")
    elif is_databricks_runtime:
        # In Databricks runtime, default workspace tracking is expected.
        mlflow.set_tracking_uri("databricks")
        logger.info("MLflow tracking URI set to Databricks workspace")
    elif profile:
        mlflow.set_tracking_uri(f"databricks://{profile}")
        logger.info("MLflow tracking URI set to Databricks profile '%s'", profile)
    else:
        logger.info("MLflow tracking URI not configured; using local default")

    experiment_path = os.getenv("MLFLOW_EXPERIMENT_PATH")
    if experiment_path:
        mlflow.set_experiment(experiment_path)
        logger.info("MLflow experiment set to '%s'", experiment_path)
    else:
        logger.info("MLflow experiment path not set; using default experiment")


configure_mlflow_tracking()

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Optional helper to link traces to git revisions.
setup_mlflow_git_based_version_tracking()


def main() -> None:
    """Run the Agent Server locally."""
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()


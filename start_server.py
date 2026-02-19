"""Start script for the MLflow Agent Server."""

# Import handlers so @invoke and @stream are registered.
import app  # noqa: F401

import logging
import os
from pathlib import Path

import mlflow
from mlflow.genai.agent_server import AgentServer
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(_static_dir / "index.html"))

    @app.get("/api/workspace-host")
    async def workspace_host():
        host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        if host and not host.startswith("http"):
            host = "https://" + host
        return JSONResponse({"host": host})

mlflow.set_tracking_uri("databricks")
exp_id = os.getenv("MLFLOW_EXPERIMENT_ID")
if exp_id:
    try:
        logger.info("Setting MLflow experiment to %s", exp_id)
        mlflow.set_experiment(experiment_id=exp_id)
        logger.info("MLflow experiment set by ID from MLFLOW_EXPERIMENT_ID")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to set MLflow experiment from MLFLOW_EXPERIMENT_ID='%s': %s",
            exp_id,
            exc,
        )


def main() -> None:
    """Run the Agent Server locally."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Agent Server")
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()


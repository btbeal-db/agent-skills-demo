import json
import sys
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("databricks://FEVM")


args = sys.argv[1:]
if len(args) != 1:
    print("Usage: python get_traces.py <trace_id>")
    sys.exit(1)

trace_id = args[0]

client = MlflowClient()
trace = client.get_trace(trace_id)
print(json.dumps(trace.to_dict(), indent=2))
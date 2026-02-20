#!/usr/bin/env python3
"""Streaming diagnostic script.

Hits /invocations with stream:true and prints each SSE line as it arrives
with a timestamp, so you can see if tokens trickle in or arrive all at once.

Usage:
  uv run bin/test_stream.py
  uv run bin/test_stream.py "your prompt here"
"""

import http.client
import json
import subprocess
import sys
import time
import urllib.parse
import uuid

APP_URL = "https://docx-skills-agent-1602460480284688.aws.databricksapps.com"
PROFILE = "FEVM"
LOCAL_URL = "http://localhost:8000"


def _get_token() -> str:
    result = subprocess.run(
        ["databricks", "auth", "token", "--profile", PROFILE, "-o", "json"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print("Failed to get token — falling back to no auth (local only)", file=sys.stderr)
        return ""
    return json.loads(result.stdout)["access_token"]


def run(prompt: str, base_url: str, token: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    use_ssl = parsed.scheme == "https"
    host = parsed.netloc
    path = "/invocations"

    payload = json.dumps({
        "input": [{"role": "user", "content": prompt}],
        "custom_inputs": {"conversation_id": str(uuid.uuid4())},
        "stream": True,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Content-Length": str(len(payload)),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    conn = http.client.HTTPSConnection(host, timeout=300) if use_ssl \
        else http.client.HTTPConnection(host, timeout=300)

    print(f"→ POST {base_url}{path}  (stream=true)")
    print(f"→ Prompt: {prompt!r}\n")

    t0 = time.monotonic()
    conn.request("POST", path, body=payload, headers=headers)
    response = conn.getresponse()

    print(f"← HTTP {response.status}  ({time.monotonic() - t0:.2f}s to first byte)\n")

    if response.status != 200:
        print(response.read().decode())
        return

    event_count = 0
    delta_count = 0
    buf = b""

    while True:
        chunk = response.read(256)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.rstrip(b"\r").decode("utf-8", errors="replace")
            if not line:
                continue
            elapsed = time.monotonic() - t0
            print(f"  [{elapsed:6.2f}s] {line}")
            if line.startswith("data: ") and line != "data: [DONE]":
                event_count += 1
                try:
                    ev = json.loads(line[6:])
                    if ev.get("type") == "response.output_text.delta":
                        delta_count += 1
                except Exception:
                    pass

    elapsed = time.monotonic() - t0
    print(f"\n← Done in {elapsed:.2f}s | {event_count} data events | {delta_count} delta chunks")
    conn.close()


def main() -> None:
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Say exactly: one two three four five"

    # Try local first, fall back to deployed app
    try:
        conn = http.client.HTTPConnection("localhost", 8000, timeout=2)
        conn.request("GET", "/health")
        conn.getresponse()
        conn.close()
        base_url = LOCAL_URL
        token = ""
        print("Using local server at localhost:8000\n")
    except Exception:
        base_url = APP_URL
        token = _get_token()
        print(f"Using deployed app at {APP_URL}\n")

    run(prompt, base_url, token)


if __name__ == "__main__":
    main()

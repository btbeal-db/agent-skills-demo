#!/usr/bin/env python3
"""Test client for the deployed app server.

Usage:
  Single-turn:   uv run bin/test_app_endpoint.py "make me a docx report"
  Interactive:   uv run bin/test_app_endpoint.py
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request
import uuid

APP_URL = "https://docx-skills-agent-1602460480284688.aws.databricksapps.com/invocations"
PROFILE = "FEVM"


def _get_access_token() -> str:
    result = subprocess.run(
        ["databricks", "auth", "token", "--profile", PROFILE, "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("Failed to fetch Databricks token for profile FEVM.", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(1)

    return json.loads(result.stdout)["access_token"]


def _send(prompt: str, token: str, conversation_id: str) -> dict:
    payload = {
        "input": [{"role": "user", "content": prompt}],
        "custom_inputs": {"conversation_id": conversation_id},
    }
    request = urllib.request.Request(
        APP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _extract_text(body: dict) -> str:
    try:
        return body["output"][0]["content"][0]["text"]
    except (KeyError, IndexError):
        return json.dumps(body, indent=2)


def _print_meta(body: dict) -> None:
    outputs = body.get("custom_outputs") or {}
    if outputs:
        print(f"  [session={outputs.get('session_id')}  path={outputs.get('output_path')}]")


def run_single(prompt: str, token: str) -> int:
    conversation_id = str(uuid.uuid4())
    try:
        body = _send(prompt, token, conversation_id)
        print(_extract_text(body))
        _print_meta(body)
        return 0
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1


def run_repl(token: str) -> int:
    conversation_id = str(uuid.uuid4())
    print(f"Multi-turn session  conversation_id={conversation_id}")
    print("Type your message and press Enter. Use 'quit' or Ctrl-D to exit.\n")

    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            print("Bye.")
            return 0

        try:
            body = _send(prompt, token, conversation_id)
            print(f"\nagent> {_extract_text(body)}")
            _print_meta(body)
            print()
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode()}", file=sys.stderr)
        except urllib.error.URLError as exc:
            print(f"Request failed: {exc}", file=sys.stderr)


def main() -> int:
    token = _get_access_token()
    if len(sys.argv) > 1:
        return run_single(" ".join(sys.argv[1:]), token)
    return run_repl(token)


if __name__ == "__main__":
    raise SystemExit(main())

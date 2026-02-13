#!/usr/bin/env python3
"""Simple FEVM-authenticated test call to the deployed app server."""

import json
import subprocess
import sys
import urllib.error
import urllib.request

APP_URL = "https://docx-skills-agent-1602460480284688.aws.databricksapps.com/invocations"
PROFILE = "FEVM"
PAYLOAD = {"input": [{"role": "user", "content": "Say hello in one sentence."}]}


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

    token_data = json.loads(result.stdout)
    return token_data["access_token"]


def main() -> int:
    token = _get_access_token()
    request = urllib.request.Request(
        APP_URL,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"Response status: {response.status}")
            print(json.dumps(json.loads(body), indent=2))
            return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Request failed with HTTP {exc.code}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


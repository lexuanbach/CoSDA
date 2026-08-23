from __future__ import annotations

import time
from typing import Any

import requests


def post_json_with_retries(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: int = 180,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            response = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
            if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(min(30, 2**attempt))
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text[:1000]
                raise requests.HTTPError(f"{exc}; response={detail}", response=response) from exc
            return response.json()
        except Exception as exc:  # requests has several transient exception subclasses.
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"POST failed after {retries} attempts: {last_error}") from last_error

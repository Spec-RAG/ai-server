from __future__ import annotations

import json
from urllib import error, request


def embed_text(api_key: str, model: str, text: str) -> list[float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={api_key}"
    body = {
        "model": model,
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT",
    }
    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except error.URLError as e:
        raise RuntimeError(f"URL error: {e}") from e

    values = payload.get("embedding", {}).get("values")
    if not isinstance(values, list):
        raise RuntimeError(f"Unexpected response payload: {payload}")
    return values


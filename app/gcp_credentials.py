from __future__ import annotations

import json
import os
import tempfile
from typing import Optional


def ensure_google_application_credentials() -> Optional[str]:
    """
    Bootstrap Google Application Default Credentials from an env var.

    Use-case: deploy environments (Railway/Docker) where you cannot mount a JSON file.
    Provide the service account JSON content via env `GOOGLE_APPLICATION_CREDENTIALS_JSON`.

    If `GOOGLE_APPLICATION_CREDENTIALS` is already set, this function does nothing.
    Returns the credential file path when created, else None.
    """

    if (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip():
        return None

    raw = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or "").strip()
    if not raw:
        return None

    try:
        # Validate JSON early to avoid confusing downstream errors.
        data = json.loads(raw)
        content = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError("Env GOOGLE_APPLICATION_CREDENTIALS_JSON bukan JSON yang valid.") from e

    tmp = tempfile.NamedTemporaryFile(prefix="gcp-sa-", suffix=".json", delete=False)
    try:
        tmp.write(content.encode("utf-8"))
        tmp.flush()
    finally:
        tmp.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    return tmp.name


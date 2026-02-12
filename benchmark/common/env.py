"""Environment-variable parsing and timestamp helpers.

Reusable across all benchmark versions for reading typed
environment variables and generating UTC ISO timestamps.
"""

import os
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def safe_env_int(name: str, default: int) -> int:
    """Read an environment variable as a positive integer.

    Returns *default* when the variable is unset, non-numeric,
    or not positive.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def safe_env_float(name: str, default: float) -> float:
    """Read an environment variable as a float.

    Returns *default* when the variable is unset or non-numeric.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_env_bool(name: str, default: bool) -> bool:
    """Read an environment variable as a boolean.

    Recognises ``1 / true / yes / on`` → True and
    ``0 / false / no / off`` → False (case-insensitive).
    Returns *default* for unset or unrecognised values.
    """
    value = os.getenv(name)
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default

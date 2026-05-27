import os
import json
from datetime import datetime


RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TELEMETRY_PATH = os.path.join(RESULTS_DIR, "telemetry.jsonl")


def _write(entry: dict) -> None:
    try:
        with open(TELEMETRY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Best-effort telemetry; never raise from logger
        pass


def log_telemetry(event: str, data: dict) -> None:
    payload = {
        "ts": datetime.utcnow().isoformat(),
        "event": event,
        "data": data,
    }
    _write(payload)


def record_groq_usage(response, context: str = "unknown") -> dict | None:
    """
    Best-effort extraction of token/usage information from Groq responses.
    Returns the extracted usage object if available and logs it.
    """
    usage = None
    try:
        if hasattr(response, "usage"):
            usage = response.usage
        elif isinstance(response, dict) and "usage" in response:
            usage = response.get("usage")
    except Exception:
        usage = None

    if usage:
        log_telemetry("groq_usage", {"context": context, "usage": usage})
    return usage

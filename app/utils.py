import json
import re
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return [token for token in cleaned.split() if token]


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

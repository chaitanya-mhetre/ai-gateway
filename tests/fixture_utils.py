from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURES / name).read_text())
    assert isinstance(data, dict)
    return data


def load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()

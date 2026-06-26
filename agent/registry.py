from __future__ import annotations
from typing import Any

_gateway: Any = None


def set_gateway(gw: Any) -> None:
    global _gateway
    _gateway = gw


def get_gateway() -> Any:
    if _gateway is None:
        raise RuntimeError("Gateway not initialized. Call set_gateway() first.")
    return _gateway

"""Minimal safetensors torch shim for offline testing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json


def save_file(state_dict: Dict[str, Any], path: str) -> None:
    """Persist a PyTorch-like state dict as JSON for tests."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Convert tensors to lists when possible
    serialisable = {}
    for key, value in state_dict.items():
        if hasattr(value, 'tolist'):
            serialisable[key] = value.tolist()
        else:
            serialisable[key] = value
    output.write_text(json.dumps(serialisable), encoding='utf-8')


__all__ = ['save_file']

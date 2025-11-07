from pathlib import Path
import json
import sys

import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional in CI
    torch = None
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlora.model_converter import convert_checkpoint_to_safetensors, build_tensorrt_engine


@pytest.mark.skipif(torch is None, reason="torch not available in test environment")
def test_convert_checkpoint_to_safetensors(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    state_dict = {"linear.weight": torch.randn(2, 2)}
    torch.save(state_dict, checkpoint)

    output = tmp_path / "model.safetensors"
    result = convert_checkpoint_to_safetensors(checkpoint, output)

    assert result.exists(), "Safetensors file should be created"


def test_build_tensorrt_engine_placeholder(tmp_path: Path):
    safetensors_path = tmp_path / "model.safetensors"
    safetensors_path.write_bytes(b"dummy")

    plan_path = tmp_path / "model.plan"
    result = build_tensorrt_engine(safetensors_path, plan_path)

    assert result.exists(), "Plan file should exist even without TensorRT"
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["safetensors"].endswith("model.safetensors")

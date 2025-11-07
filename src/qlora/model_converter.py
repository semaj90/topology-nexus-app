"""Utilities for converting checkpoints into TensorRT-LLM friendly formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import importlib
import importlib.util
import json
import logging

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import torch
from safetensors.torch import save_file as save_safetensors


logger = logging.getLogger(__name__)


@dataclass
class ConversionSummary:
    """Describes the outputs of the conversion pipeline."""

    safetensors_path: Optional[Path]
    engine_plan_path: Optional[Path]
    notes: Dict[str, str]


def _ensure_package(name: str, install_hint: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Package '{name}' is required. Install via: {install_hint}")


def convert_checkpoint_to_safetensors(checkpoint_path: Path, output_path: Path) -> Path:
    """Convert a PyTorch checkpoint (``.bin`` or ``.pt``) to ``.safetensors``."""

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    if torch is None:
        raise RuntimeError("PyTorch is required to convert checkpoints to safetensors")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a state_dict dictionary in the checkpoint")

    logger.info("Writing safetensors payload to %s", output_path)
    save_safetensors(state_dict, str(output_path))
    return output_path


def build_tensorrt_engine(safetensors_path: Path, output_path: Path, config_path: Optional[Path] = None) -> Path:
    """Create a TensorRT engine plan file from ``safetensors_path``.

    If TensorRT-LLM is not available in the environment we fall back to writing a
    metadata-only placeholder.  This keeps the desktop workflow responsive while
    making the dependency requirements explicit to the operator.
    """

    safetensors_path = Path(safetensors_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_tensorrt_llm = importlib.util.find_spec("tensorrt_llm") is not None
    notes: Dict[str, str] = {}

    if not has_tensorrt_llm:
        notes["status"] = "placeholder"
        notes["message"] = (
            "TensorRT-LLM is not installed. Generated a placeholder plan file "
            "containing environment instructions."
        )
        payload = {
            "safetensors": str(safetensors_path),
            "config": str(config_path) if config_path else None,
            "instructions": "Install TensorRT-LLM 0.9+ with CUDA 12.6 to build the engine.",
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.warning(notes["message"])
        return output_path

    _ensure_package(
        "tensorrt_llm",
        "Follow NVIDIA's installation guide for TensorRT-LLM >=0.9 and ensure CUDA 12.6 is available.",
    )

    tensorrt_llm = importlib.import_module("tensorrt_llm")

    network_config = None
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            network_config = json.loads(config_path.read_text(encoding="utf-8"))

    logger.info("Building TensorRT engine at %s", output_path)

    builder = tensorrt_llm.Builder()
    network = builder.create_network()
    parser = tensorrt_llm.parser.TrtLlmParser(network)
    parser.parse(safetensors_path)  # type: ignore[arg-type]

    if network_config:
        builder.apply_network_config(network, network_config)

    engine = builder.build_engine(network)
    with open(output_path, "wb") as handle:
        handle.write(engine.serialize())

    logger.info("TensorRT engine written to %s", output_path)
    return output_path


def convert_model_pipeline(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    engine_name: str = "model.plan",
    safetensors_name: str = "model.safetensors",
    config_path: Optional[Path] = None,
) -> ConversionSummary:
    """Full conversion pipeline from PyTorch checkpoints to TensorRT plans."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = output_dir / safetensors_name
    engine_path = output_dir / engine_name

    safetensors_file = convert_checkpoint_to_safetensors(checkpoint_path, safetensors_path)

    try:
        plan_file = build_tensorrt_engine(safetensors_file, engine_path, config_path=config_path)
    except RuntimeError as error:
        notes = {"error": str(error)}
        return ConversionSummary(safetensors_file, None, notes)

    return ConversionSummary(safetensors_file, plan_file, notes={})


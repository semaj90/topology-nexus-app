"""Definitions for describing QLoRA adapter training tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json


DEFAULT_TRAINING_SCRIPT = "src/qlora/run_qlora.py"


@dataclass
class TrainingHyperparameters:
    learning_rate: float = 2e-4
    micro_batch_size: int = 4
    max_steps: int = 100
    warmup_steps: int = 10
    gradient_checkpointing: bool = True
    logging_steps: int = 10


@dataclass
class QLoRATrainingSpec:
    base_model: str
    dataset_path: Path
    output_dir: Path
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    tokenizer_name: Optional[str] = None
    embedding_model: str = "embeddinggemma:latest"

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        return payload

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    def build_accelerate_command(self, python_executable: str = "python") -> str:
        config_path = self.output_dir / "training_spec.json"
        return (
            f"{python_executable} {DEFAULT_TRAINING_SCRIPT} "
            f"--config {config_path}"
        )


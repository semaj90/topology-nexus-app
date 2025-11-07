"""Minimal QLoRA training entry point.

The script expects a JSON configuration created via
``src.qlora.training_spec.QLoRATrainingSpec``.  It performs dependency checks at
runtime so that the desktop application can provide actionable feedback when the
environment is missing GPU drivers, PyTorch, or TensorRT-LLM.  Training defaults
to CPU execution, but when CUDA is available the user can opt-in via the
``CUDA_VISIBLE_DEVICES`` environment variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import argparse
import importlib
import importlib.util
import json
import logging
import os


logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_dependency(name: str, message: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(message)


def _resolve_device() -> str:
    if importlib.util.find_spec("torch") is None:
        return "cpu"

    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_trainer(config: Dict[str, object]):
    _check_dependency(
        "transformers",
        "transformers is required. Install with `pip install transformers accelerate peft bitsandbytes`.",
    )
    _check_dependency(
        "peft",
        "peft is required. Install with `pip install peft`.",
    )
    _check_dependency(
        "datasets",
        "datasets is required. Install with `pip install datasets`.",
    )

    transformers = importlib.import_module("transformers")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")

    dataset_path = Path(config["dataset_path"])
    base_model = config["base_model"]
    output_dir = Path(config["output_dir"])

    tokenizer_name = config.get("tokenizer_name") or base_model
    logger.info("Loading tokenizer %s", tokenizer_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading dataset from %s", dataset_path)
    dataset = datasets.load_dataset("json", data_files={"train": str(dataset_path)})["train"]

    def tokenize(example: Dict[str, str]) -> Dict[str, object]:
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=tokenizer.model_max_length)

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    logger.info("Loading base model %s", base_model)
    torch_module = importlib.import_module("torch") if importlib.util.find_spec("torch") else None
    torch_dtype = torch_module.float16 if (torch_module and _resolve_device() == "cuda") else None
    model = transformers.AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)

    lora_config = peft.LoraConfig(
        r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        task_type=peft.TaskType.CAUSAL_LM,
    )

    model = peft.get_peft_model(model, lora_config)

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("hyperparameters", {}).get("micro_batch_size", 4),
        learning_rate=config.get("hyperparameters", {}).get("learning_rate", 2e-4),
        max_steps=config.get("hyperparameters", {}).get("max_steps", 100),
        warmup_steps=config.get("hyperparameters", {}).get("warmup_steps", 10),
        logging_steps=config.get("hyperparameters", {}).get("logging_steps", 10),
        gradient_checkpointing=config.get("hyperparameters", {}).get("gradient_checkpointing", True),
        fp16=_resolve_device() == "cuda",
        report_to=[],
    )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    return trainer


def run_training(config_path: Path) -> None:
    config = _load_config(config_path)
    trainer = _prepare_trainer(config)
    device = _resolve_device()
    logger.info("Starting training on %s", device)
    trainer.train()
    trainer.save_model()
    logger.info("Training completed. Adapter saved to %s", config.get("output_dir"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QLoRA adapter training")
    parser.add_argument("--config", required=True, help="Path to QLoRA training JSON config")
    parser.add_argument("--log-level", default=os.getenv("QLOTRA_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    run_training(Path(args.config))


if __name__ == "__main__":
    main()


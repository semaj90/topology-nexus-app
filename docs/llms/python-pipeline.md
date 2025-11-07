# Python Pipeline Modules

The Python package under `src/qlora` provides focused utilities for dataset
preparation, embedding capture, QLoRA training, and model conversion. The key
modules exposed to the desktop studio and CLI are:

- `dataset_builder.py`
  - Normalises Markdown, HTML, JSON, JSONL, and plaintext files into `DocumentChunk`
    objects.
  - Supports adjustable chunk sizes with overlap handling to preserve context.
  - Writes structured JSONL output enriched with metadata.
- `ollama_embeddings.py`
  - Streams JSONL dataset rows to an Ollama endpoint for embedding generation.
  - Includes helpers for batching requests and persisting the resulting vectors.
- `training_spec.py`
  - Produces adapter configuration JSON describing training hyperparameters,
    dataset references, and hardware expectations (CUDA 12.6 + TensorRT 9.5).
- `run_qlora.py`
  - Wraps the Hugging Face QLoRA training loop with environment checks so that
    WSL2 deployments can surface actionable error messages before invoking
    PyTorch.
- `model_converter.py`
  - Converts trained adapters and base models into `.safetensors` or TensorRT-LLM
    `.plan` artifacts, ensuring that exported engines reference the generated
    adapter metadata.

The CLI (see `main.py`) exposes high-level commands that delegate to each module
so they can be executed programmatically from the desktop UI, from shell scripts,
or inside automated pipelines.

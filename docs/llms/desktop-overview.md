# Desktop Studio Overview

The Tauri + SvelteKit desktop studio orchestrates dataset preparation, embedding
jobs, and checkpoint conversion tasks by calling into the Python CLI exposed in
`main.py`. The UI renders forms that mirror the available CLI options for:

- Building JSONL corpora from uploaded documents through
  `src/qlora/dataset_builder.py`.
- Submitting embedding batches to an Ollama runtime via
  `src/qlora/ollama_embeddings.py`.
- Creating QLoRA adapter specifications and conversion pipelines powered by
  `src/qlora/training_spec.py`, `src/qlora/run_qlora.py`, and
  `src/qlora/model_converter.py`.

The desktop route `/` wires these operations together so that a creator can:

1. Select raw `.txt`, `.md`, `.html`, `.json`, or `.jsonl` files for ingestion.
2. Configure chunking controls and persist the resulting dataset as JSONL.
3. Forward the saved dataset into Ollama for embedding generation using the
   `embeddinggemma:latest` model and capture the vector responses.
4. Generate training specs and adapter plans that point at the dataset and the
   uploaded base checkpoint. The Python backend handles conversion to
   TensorRT-LLM `.plan` engines or `.safetensors` weights depending on the
   chosen export mode.

Each action writes status updates into the UI log to make it easy to copy the
results for further automation or ingestion into additional pipelines.

# Topology Nexus Desktop

The desktop companion application wraps the Python tooling in a Tauri shell so datasets, embeddings, and TensorRT conversions can be orchestrated from Windows while executing inside WSL2.

## Prerequisites

- Node.js 18+
- Rust toolchain (for compiling the Tauri backend)
- Python environment with project dependencies installed (see `requirements.txt`)
- Optional: `PYTHON_EXECUTABLE` environment variable pointing to the interpreter inside your WSL2 environment
- Ollama running locally with the `embeddinggemma:latest` model pulled (`ollama pull embeddinggemma:latest`)

## Development

```bash
cd desktop
npm install
npm run tauri
```

The Tauri backend proxies heavy operations to the Python modules:

- `build_dataset` → `src/qlora/dataset_builder.chunk_documents`
- `convert_checkpoint` → `src/qlora/model_converter.convert_model_pipeline`
- `embed_texts` → Ollama embeddings API (`http://localhost:11434/api/embeddings`)

TensorRT plan generation requires the environment described in `config/tensorrt_llm_environment.json`.

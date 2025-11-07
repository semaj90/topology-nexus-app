#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Serialize)]
struct ConversionSummary {
    safetensors_path: String,
    engine_plan_path: Option<String>,
    notes: HashMap<String, String>,
}

#[derive(Serialize)]
struct EmbeddingResult {
    text: String,
    vector: Vec<f32>,
    model: String,
    elapsed_ms: f64,
}

fn find_repo_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;
    for _ in 0..6 {
        if current.join("main.py").exists() {
            return Ok(current);
        }
        if !current.pop() {
            break;
        }
    }
    Err(anyhow!("Unable to locate repository root (main.py not found)"))
}

fn python_executable() -> String {
    std::env::var("PYTHON_EXECUTABLE")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python".to_string())
}

#[tauri::command]
fn build_dataset(inputs: Vec<String>, output: String, chunk_size: usize, overlap: usize) -> Result<String, String> {
    let repo_root = find_repo_root().map_err(|e| e.to_string())?;
    let python = python_executable();

    let inputs_json = serde_json::to_string(&inputs).map_err(|e| e.to_string())?;
    let output_json = serde_json::to_string(&output).map_err(|e| e.to_string())?;

    let script = format!(
        r#"
import json
from pathlib import Path
from src.qlora.dataset_builder import chunk_documents

inputs = json.loads({inputs})
resolved_inputs = []
for path in inputs:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    resolved_inputs.append(str(p))

output_path = Path({output})
if not output_path.is_absolute():
    output_path = Path.cwd() / output_path

chunk_documents(resolved_inputs, str(output_path), chunk_size={chunk_size}, overlap={overlap})
print(str(output_path))
"#,
        inputs = inputs_json,
        output = output_json,
        chunk_size = chunk_size,
        overlap = overlap
    );

    let command_output = Command::new(python)
        .arg("-c")
        .arg(script)
        .current_dir(&repo_root)
        .output()
        .map_err(|e| e.to_string())?;

    if !command_output.status.success() {
        let stderr = String::from_utf8_lossy(&command_output.stderr);
        return Err(stderr.trim().to_string());
    }

    let stdout = String::from_utf8_lossy(&command_output.stdout);
    Ok(stdout.trim().to_string())
}

#[tauri::command]
fn convert_checkpoint(checkpoint: String, output_dir: String, config: Option<String>) -> Result<ConversionSummary, String> {
    let repo_root = find_repo_root().map_err(|e| e.to_string())?;
    let python = python_executable();

    let checkpoint_json = serde_json::to_string(&checkpoint).map_err(|e| e.to_string())?;
    let output_dir_json = serde_json::to_string(&output_dir).map_err(|e| e.to_string())?;
    let config_json = serde_json::to_string(&config.unwrap_or_default()).map_err(|e| e.to_string())?;
    let config_flag = if config.is_some() { "True" } else { "False" };

    let script = format!(
        r#"
import json
from pathlib import Path
from src.qlora.model_converter import convert_model_pipeline

checkpoint = Path({checkpoint})
output_dir = Path({output_dir})
if not output_dir.is_absolute():
    output_dir = Path.cwd() / output_dir
output_dir.mkdir(parents=True, exist_ok=True)

config_path = Path({config}) if {config_flag} else None

summary = convert_model_pipeline(checkpoint, output_dir, config_path=config_path)
print(json.dumps({{
    "safetensorsPath": str(summary.safetensors_path),
    "enginePlanPath": str(summary.engine_plan_path) if summary.engine_plan_path else None,
    "notes": summary.notes
}}))
"#,
        checkpoint = checkpoint_json,
        output_dir = output_dir_json,
        config = config_json,
        config_flag = config_flag
    );

    let command_output = Command::new(python)
        .arg("-c")
        .arg(script)
        .current_dir(&repo_root)
        .output()
        .map_err(|e| e.to_string())?;

    if !command_output.status.success() {
        let stderr = String::from_utf8_lossy(&command_output.stderr);
        return Err(stderr.trim().to_string());
    }

    let stdout = String::from_utf8_lossy(&command_output.stdout);
    let value: Value = serde_json::from_str(stdout.trim()).map_err(|e| e.to_string())?;

    let safetensors_path = value
        .get("safetensorsPath")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing safetensorsPath".to_string())?
        .to_string();
    let engine_plan_path = value
        .get("enginePlanPath")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let notes = value
        .get("notes")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let notes_map: HashMap<String, String> = notes
        .as_object()
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok(ConversionSummary {
        safetensors_path,
        engine_plan_path,
        notes: notes_map,
    })
}

#[tauri::command]
fn embed_texts(texts: Vec<String>, model: String) -> Result<Vec<EmbeddingResult>, String> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for text in texts {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        let payload = json!({
            "model": model,
            "prompt": trimmed
        });

        let start = std::time::Instant::now();
        let response = client
            .post("http://localhost:11434/api/embeddings")
            .json(&payload)
            .send()
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err(format!("Ollama error: {}", response.status()));
        }

        let value: Value = response.json().map_err(|e| e.to_string())?;
        let embedding = value
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "Invalid embedding payload".to_string())?;

        let vector: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|v| v as f32)
            .collect();

        results.push(EmbeddingResult {
            text: trimmed,
            vector,
            model: model.clone(),
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
    }

    Ok(results)
}

#[tauri::command]
fn load_semantic_index() -> Result<Value, String> {
    let repo_root = find_repo_root().map_err(|e| e.to_string())?;
    let index_path = repo_root.join("data").join("semantic_index.json");
    if !index_path.exists() {
        return Ok(Value::Array(vec![]));
    }

    let contents = fs::read_to_string(&index_path).map_err(|e| e.to_string())?;
    let value: Value = serde_json::from_str(&contents).map_err(|e| e.to_string())?;
    Ok(value)
}

#[tauri::command]
fn refresh_semantic_index(dataset: Option<String>) -> Result<usize, String> {
    let repo_root = find_repo_root().map_err(|e| e.to_string())?;
    let python = python_executable();
    let dataset_path = dataset.unwrap_or_else(|| "docs/llms/crawled/ts_drizzle_uno.jsonl".to_string());
    let dataset_json = serde_json::to_string(&dataset_path).map_err(|e| e.to_string())?;

    let script = format!(
        r#"
from pathlib import Path
from src.semantic.index_builder import build_semantic_index

dataset = Path({dataset})
if not dataset.is_absolute():
    dataset = Path.cwd() / dataset
output_path = Path('data/semantic_index.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
count = build_semantic_index(dataset, output_path)
print(count)
"#,
        dataset = dataset_json,
    );

    let command_output = Command::new(python)
        .arg("-c")
        .arg(script)
        .current_dir(&repo_root)
        .output()
        .map_err(|e| e.to_string())?;

    if !command_output.status.success() {
        let stderr = String::from_utf8_lossy(&command_output.stderr);
        return Err(stderr.trim().to_string());
    }

    let stdout = String::from_utf8_lossy(&command_output.stdout);
    let trimmed = stdout.trim();
    let count: usize = trimmed.parse().map_err(|e| e.to_string())?;
    Ok(count)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            build_dataset,
            convert_checkpoint,
            embed_texts,
            load_semantic_index,
            refresh_semantic_index
        ])
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![build_dataset, convert_checkpoint, embed_texts])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

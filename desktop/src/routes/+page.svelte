<script lang="ts">
  import {
    buildDataset,
    convertCheckpoint,
    embedTexts,
    refreshSemanticIndex,
    type ConversionSummary,
    type EmbeddingResult,
    type SemanticIndexEntry
  } from '$lib/api';
  import { initialiseSemanticIndex, runSemanticSearch, semanticIndexStore } from '$lib/search';
  import { onDestroy, onMount } from 'svelte';

  let datasetInputs: FileList | null = null;
  let datasetOutput = '';
  let chunkSize = 1024;
  let overlap = 200;
  let datasetStatus = '';

  let embeddingText = '';
  let embeddingModel = 'embeddinggemma:latest';
  let embeddingResults: EmbeddingResult[] = [];
  let embeddingStatus = '';

  let checkpointFile: File | null = null;
  let outputDirectory = '';
  let conversionConfig = '';
  let conversionSummary: ConversionSummary | null = null;
  let conversionStatus = '';

  let semanticStatus = '';
  let searchQuery = '';
  let searchResults: SemanticIndexEntry[] = [];
  let semanticRecords: SemanticIndexEntry[] = [];
  const unsubscribe = semanticIndexStore.subscribe((records) => {
    semanticRecords = records;
    if (!searchQuery.trim()) {
      searchResults = records.slice(0, 5);
    }
  });

  onDestroy(() => unsubscribe());

  onMount(async () => {
    semanticStatus = 'Loading semantic index...';
    try {
      const records = await initialiseSemanticIndex();
      semanticStatus = `Indexed ${records.length} documentation chunks from Qdrant/PGVector exports.`;
      searchResults = records.slice(0, 5);
    } catch (error) {
      semanticStatus = `Unable to load semantic index: ${error}`;
    }
  });

  $: searchResults = searchQuery.trim()
    ? runSemanticSearch(searchQuery)
    : semanticRecords.slice(0, Math.min(5, semanticRecords.length));

  async function handleDatasetBuild() {
    if (!datasetInputs || datasetInputs.length === 0) {
      datasetStatus = 'Select at least one document.';
      return;
    }
    if (!datasetOutput) {
      datasetStatus = 'Provide an output JSONL path.';
      return;
    }

    datasetStatus = 'Building dataset...';
    const inputPaths: string[] = [];
    for (const file of Array.from(datasetInputs)) {
      inputPaths.push(file.path ?? file.name);
    }

    try {
      const result = await buildDataset({
        output: datasetOutput,
        inputs: inputPaths,
        chunkSize,
        overlap
      });
      datasetStatus = `Dataset written to ${result}`;
    } catch (error) {
      datasetStatus = `Failed: ${error}`;
    }
  }

  async function handleEmbeddings() {
    if (!embeddingText.trim()) {
      embeddingStatus = 'Provide text to embed.';
      return;
    }
    embeddingStatus = 'Requesting embeddings from Ollama...';
    try {
      embeddingResults = await embedTexts({
        texts: [embeddingText],
        model: embeddingModel
      });
      embeddingStatus = `Received ${embeddingResults.length} embedding(s).`;
    } catch (error) {
      embeddingStatus = `Failed: ${error}`;
    }
  }

  async function handleConversion() {
    if (!checkpointFile) {
      conversionStatus = 'Select a checkpoint file.';
      return;
    }
    if (!outputDirectory) {
      conversionStatus = 'Provide an output directory.';
      return;
    }

    conversionStatus = 'Converting checkpoint...';
    try {
      conversionSummary = await convertCheckpoint({
        checkpoint: checkpointFile.path ?? checkpointFile.name,
        outputDir: outputDirectory,
        config: conversionConfig || null
      });
      conversionStatus = 'Conversion complete.';
    } catch (error) {
      conversionStatus = `Failed: ${error}`;
    }
  }

  async function handleIndexRefresh() {
    semanticStatus = 'Refreshing semantic index...';
    try {
      const count = await refreshSemanticIndex();
      const records = await initialiseSemanticIndex();
      semanticStatus = `Semantic index refreshed with ${count} entries.`;
      if (searchQuery.trim()) {
        searchResults = runSemanticSearch(searchQuery);
      } else {
        searchResults = records.slice(0, 5);
      }
    } catch (error) {
      semanticStatus = `Failed to refresh index: ${error}`;
    }
  }
</script>

<main class="container">
  <section>
    <h1>Topology Nexus Desktop</h1>
    <p>
      Build QLoRA adapter datasets, request embeddings from Ollama, and convert checkpoints into
      TensorRT-LLM engine plans—all from a single desktop interface.
    </p>
  </section>

  <section>
    <h2>1. Build JSONL Dataset</h2>
    <label>
      Documents
      <input type="file" multiple bind:files={datasetInputs} />
    </label>
    <label>
      Output JSONL Path
      <input type="text" bind:value={datasetOutput} placeholder="/path/to/output.jsonl" />
    </label>
    <div class="row">
      <label>
        Chunk Size
        <input type="number" bind:value={chunkSize} min="128" />
      </label>
      <label>
        Overlap
        <input type="number" bind:value={overlap} min="0" />
      </label>
    </div>
    <button on:click|preventDefault={handleDatasetBuild}>Build Dataset</button>
    <p class="status">{datasetStatus}</p>
  </section>

  <section>
    <h2>2. Generate Embeddings</h2>
    <label>
      Text
      <textarea rows="4" bind:value={embeddingText} placeholder="Paste the text to embed"></textarea>
    </label>
    <label>
      Ollama Model
      <input type="text" bind:value={embeddingModel} />
    </label>
    <button on:click|preventDefault={handleEmbeddings}>Generate</button>
    <p class="status">{embeddingStatus}</p>

    {#if embeddingResults.length}
      <div class="card">
        <h3>Embedding Preview</h3>
        <p>Model: {embeddingResults[0].model}</p>
        <p>Vector length: {embeddingResults[0].vector.length}</p>
      </div>
    {/if}
  </section>

  <section>
    <h2>3. Convert Checkpoint</h2>
    <label>
      Checkpoint (.pt/.bin)
      <input type="file" bind:files={(files) => (checkpointFile = files?.[0] ?? null)} />
    </label>
    <label>
      Output Directory
      <input type="text" bind:value={outputDirectory} placeholder="/path/to/output" />
    </label>
    <label>
      TensorRT Network Config (optional)
      <input type="text" bind:value={conversionConfig} placeholder="/path/to/config.json" />
    </label>
    <button on:click|preventDefault={handleConversion}>Convert</button>
    <p class="status">{conversionStatus}</p>

    {#if conversionSummary}
      <div class="card">
        <h3>Conversion Outputs</h3>
        <p>Safetensors: {conversionSummary.safetensorsPath}</p>
        {#if conversionSummary.enginePlanPath}
          <p>TensorRT Engine: {conversionSummary.enginePlanPath}</p>
        {:else if conversionSummary.notes}
          <pre>{JSON.stringify(conversionSummary.notes, null, 2)}</pre>
        {/if}
      </div>
    {/if}
  </section>

  <section>
    <h2>4. Semantic Search & Indexed Memory</h2>
    <p>
      Indexed DB storage mirrors XRabbit's server-side memory queue so Fuse.js can perform low-latency semantic
      lookups across the LangExtract corpus tagged in Qdrant/PGVector.
    </p>
    <div class="row search-bar">
      <input
        type="text"
        bind:value={searchQuery}
        placeholder="Search TypeScript, Drizzle ORM, UnoCSS snippets..."
      />
      <button on:click|preventDefault={handleIndexRefresh}>Refresh Index</button>
    </div>
    <p class="status">{semanticStatus}</p>

    {#if searchResults.length}
      <ul class="search-results">
        {#each searchResults as result}
          <li class="result-item">
            <h3>{result.title}</h3>
            {#if result.topic}
              <p class="topic">Topic: {result.topic}</p>
            {/if}
            <p>{result.content}</p>
            <div class="tags">
              {#each result.tags.slice(0, 6) as tag}
                <span class="tag">{tag}</span>
              {/each}
            </div>
            {#if result.score !== undefined}
              <p class="score">Relevance: {(1 - (result.score ?? 0)).toFixed(3)}</p>
            {/if}
          </li>
        {/each}
      </ul>
    {:else if searchQuery.trim()}
      <p class="status">No semantic matches for "{searchQuery}".</p>
    {/if}
  </section>
</main>

<style>
  .container {
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1.5rem 4rem;
    display: flex;
    flex-direction: column;
    gap: 2.5rem;
  }

  section {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
  }

  h1,
  h2 {
    margin-bottom: 1rem;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
  }

  input,
  textarea {
    padding: 0.75rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    background: rgba(18, 18, 18, 0.9);
    color: #f3f4f6;
  }

  textarea {
    resize: vertical;
  }

  button {
    align-self: flex-start;
    padding: 0.75rem 1.5rem;
    border-radius: 999px;
    border: none;
    background: linear-gradient(135deg, #6366f1, #14b8a6);
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease;
  }

  button:hover {
    transform: translateY(-1px);
  }

  .row {
    display: flex;
    gap: 1rem;
  }

  .row label {
    flex: 1;
  }

  .search-bar {
    align-items: center;
  }

  .search-bar input {
    flex: 1;
  }

  .search-results {
    list-style: none;
    display: grid;
    gap: 1rem;
    margin-top: 1rem;
    padding: 0;
  }

  .result-item {
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1rem;
    background: rgba(15, 23, 42, 0.6);
  }

  .result-item h3 {
    margin-bottom: 0.5rem;
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .tag {
    background: rgba(99, 102, 241, 0.2);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
  }

  .topic {
    font-size: 0.85rem;
    color: #fbbf24;
  }

  .score {
    font-size: 0.8rem;
    margin-top: 0.5rem;
    color: #38bdf8;
  }

  .status {
    min-height: 1.5rem;
    font-size: 0.9rem;
    color: #93c5fd;
  }

  .card {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.3);
  }
</style>

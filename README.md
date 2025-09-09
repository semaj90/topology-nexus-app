# Topology Nexus App

A comprehensive platform for contextual engineering that fetches webpages, parses them into JSON/JSONL, uses semantic understanding for intelligent text splitting, and creates distilled QLoRA JavaScript modules for memory-efficient contextual prompting.

## Features

- **Web Scraping & Parsing**: Fetches webpages and converts them to structured JSON/JSONL format
- **Semantic Processing**: Uses LangChain and semantic splitting for intelligent text understanding
- **QLoRA Modules**: Memory-efficient loading/unloading of distilled modules based on user queries
- **Interactive Studio**: Testing environment for features and mini-dataset creation
- **Topology Training**: Train on different neural network topologies (transformer, graph, hierarchical, hybrid)
- **GPU Support**: Neural network operations with CUDA support for training and inference
- **Contextual Prompting**: Dynamic context selection based on semantic similarity
- **Multi-Database Architecture**: PostgreSQL, MinIO, and CouchDB integration with Docker Compose
- **RESTful API**: Complete API endpoints for all functionality with Node.js and Python examples

## Architecture

```
├── src/
│   ├── scraper/          # Web scraping and data extraction
│   ├── semantic/         # LangChain-based text processing
│   ├── qlora/           # QLoRA memory management (JavaScript)
│   ├── studio/          # Interactive testing environment
│   └── trainer/         # Topology training and GPU operations
├── examples/             # Integration examples
│   ├── nodejs/          # Node.js database integration examples
│   └── python/          # Python database integration examples
├── docker/              # Docker configuration files
├── config/              # Configuration files
├── data/               # Data storage and datasets
└── tests/              # Test suites
```

## Docker Compose Setup

### Quick Start with Databases

1. **Start all services (PostgreSQL, MinIO, CouchDB):**
   ```bash
   docker-compose up -d
   ```

2. **Install dependencies:**
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. **Test database connections:**
   ```bash
   npm run example:pg     # Test PostgreSQL
   npm run example:minio  # Test MinIO
   npm run example:couchdb # Test CouchDB
   ```

4. **Start API server:**
   ```bash
   npm run api
   ```

### Service Access

- **PostgreSQL**: localhost:5432 (nexus/changeme)
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **CouchDB Admin**: http://localhost:5984/_utils (admin/changeme)
- **API Documentation**: http://localhost:3000/api

For detailed setup instructions, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/semaj90/topology-nexus-app.git
   cd topology-nexus-app
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

## Quick Start

### 1. Run Complete Demo Pipeline
```bash
python main.py demo https://example.com https://httpbin.org/html
```

### 2. Interactive Studio Mode
```bash
python main.py studio
```
or
```bash
npm run studio
```

### 3. Scrape and Process URLs
```bash
python main.py scrape https://example.com --output processed_data.json
```

### 4. Train a Topology
```bash
python main.py train transformer processed_data.json --epochs 5
```

## Components

### Web Scraper (`src/scraper/web_scraper.py`)
- Fetches webpages with rate limiting and error handling
- Extracts structured content, metadata, and links
- Outputs JSON and JSONL formats

**Example:**
```python
from src.scraper.web_scraper import WebScraper

scraper = WebScraper(delay=1.0)
data = scraper.scrape_urls(['https://example.com'])
scraper.save_to_jsonl(data, 'output.jsonl')
```

### Semantic Processor (`src/semantic/text_processor.py`)
- Uses LangChain for intelligent text splitting
- Semantic chunking with embeddings
- Key concept extraction and similarity search

**Example:**
```python
from src.semantic.text_processor import SemanticProcessor

processor = SemanticProcessor()
chunks = processor.process_scraped_data(scraped_data)
similar = processor.find_similar_chunks("machine learning", chunks)
```

### QLoRA Memory Manager (`src/qlora/memory_manager.js`)
- Efficient module loading/unloading based on memory constraints
- Contextual module swapping based on user queries
- Low-rank compression for memory optimization

**Example:**
```javascript
const QLoRAMemoryManager = require('./src/qlora/memory_manager');

const manager = new QLoRAMemoryManager({ maxMemoryMB: 512 });
await manager.loadModule('module-1', moduleData);
const relevant = await manager.findRelevantModules(query, availableModules);
```

### Topology Trainer (`src/trainer/topology_trainer.py`)
- Multiple neural network architectures (transformer, graph, hierarchical, hybrid)
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- GPU acceleration with CUDA support
- Model checkpointing and swapping

**Example:**
```python
from src.trainer.topology_trainer import TopologyTrainer, DEFAULT_TOPOLOGIES

trainer = TopologyTrainer()
trainer.create_topology(DEFAULT_TOPOLOGIES['transformer'])
stats = trainer.train_topology('transformer', training_data)
```

### Studio Interface (`src/studio/studio.py`)
- Interactive dataset creation and management
- Semantic querying and contextual prompt generation
- QLoRA module export functionality
- Dataset statistics and visualization

## Studio Commands

In interactive mode:
- `create <name> <url1,url2,...> [description]` - Create new dataset
- `list` - List all datasets
- `info <name>` - Show dataset information  
- `query <dataset> <query>` - Query dataset semantically
- `prompt <dataset> <query>` - Generate contextual prompt
- `export <dataset> [output_dir]` - Export QLoRA modules
- `exit` - Exit studio

## Configuration

Edit `config/default_config.json` to customize:

```json
{
  "scraper": {
    "delay": 1.0,
    "timeout": 30
  },
  "semantic": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 1000
  },
  "qlora": {
    "max_memory_mb": 512,
    "compression_ratio": 0.1
  },
  "trainer": {
    "learning_rate": 1e-4,
    "batch_size": 16
  }
}
```

## Testing

### Run JavaScript Tests
```bash
npm test
```

### Run Python Components
```bash
python -m pytest tests/ # (if pytest tests are added)
```

## GPU Requirements

- **CUDA-compatible GPU** (recommended for training)
- **PyTorch with CUDA support**
- **Minimum 4GB GPU memory** for small models
- **8GB+ recommended** for larger topologies

## Memory Management

The QLoRA system automatically manages memory by:
- Compressing modules using low-rank approximation
- Unloading least-recently-used modules when memory is full
- Contextually loading relevant modules based on queries
- Providing memory usage statistics and alerts

## Topology Types

1. **Transformer**: Standard transformer architecture with multi-head attention
2. **Graph**: Graph neural network for structured data relationships
3. **Hierarchical**: Hierarchical attention for document-level understanding
4. **Hybrid**: Combination of multiple architectures

## Example Workflow

1. **Data Collection**: Scrape relevant websites
2. **Processing**: Convert to semantic chunks with embeddings
3. **Module Creation**: Export as QLoRA modules
4. **Training**: Fine-tune topologies on domain-specific data
5. **Deployment**: Load relevant modules contextually for inference
6. **Querying**: Use semantic similarity for context retrieval

## Database Integration

The Topology Nexus App integrates three complementary databases:

### PostgreSQL - Relational Data
- **Users, topologies, modules, and training results**
- **Semantic chunks with vector embeddings**
- **ACID compliance for critical data integrity**

```python
from examples.python.pg_example import TopologyDB

with TopologyDB() as db:
    user = db.create_user('scientist', 'scientist@example.com')
    topology = db.create_topology('my-transformer', 'Custom model', 'transformer', user['id'])
    results = db.save_training_results(topology['id'], {...}, [0.8, 0.3, 0.1], 0.94, 5)
```

### MinIO - Object Storage  
- **Model files, datasets, logs, and artifacts**
- **S3-compatible API with bucket organization**
- **Efficient large file handling and presigned URLs**

```javascript
const { TopologyStorage } = require('./examples/nodejs/minio-example');

const storage = new TopologyStorage();
await storage.uploadModel('my-model', modelData, {version: '1.0', accuracy: 0.94});
const url = await storage.getPresignedUrl('models', 'my-model.json', 3600);
```

### CouchDB - Document Storage
- **Engineering context, annotations, and experiments**  
- **Flexible schema for evolving metadata**
- **Full-text search and view-based queries**

```python
from examples.python.couchdb_example import TopologyContext

context = TopologyContext()
context.add_context('insight', 'Model shows attention bias in layer 3', 'topology-123')
experiments = context.get_experiments_by_topology('topology-123')
```

### API Integration

Complete RESTful API with all database operations:

```bash
# Start API server
npm run api

# Create topology with automatic context logging
curl -X POST http://localhost:3000/api/topologies \
  -H "Content-Type: application/json" \
  -d '{"name": "my-model", "architecture": "transformer", "created_by": 1}'

# Upload model to MinIO  
curl -X POST http://localhost:3000/api/models \
  -H "Content-Type: application/json" \
  -d '{"name": "trained-v1", "model_data": {...}}'

# Search across all context
curl "http://localhost:3000/api/context/search?q=overfitting&limit=10"
```

For detailed integration examples, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

## Performance Tips

- Use GPU acceleration for training (`torch.cuda.is_available()`)
- Adjust `chunk_size` based on your content type
- Set appropriate `max_memory_mb` for your system
- Use `batch_size` that fits your GPU memory
- Enable `mixed_precision` for faster training

## Troubleshooting

### Common Issues:
- **CUDA out of memory**: Reduce `batch_size` or `max_memory_mb`
- **Import errors**: Ensure all requirements are installed
- **Slow scraping**: Increase `delay` to avoid rate limiting
- **Large memory usage**: Reduce `compression_ratio` or enable auto-unloading

### Debug Mode:
```bash
python main.py --log-level DEBUG <command>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

ISC License - see LICENSE file for details

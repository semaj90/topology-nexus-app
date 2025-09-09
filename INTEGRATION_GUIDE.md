# Database Integration Guide

This guide provides detailed examples and best practices for integrating PostgreSQL, MinIO, and CouchDB with the Topology Nexus App.

## Architecture Overview

The Topology Nexus App uses a multi-database architecture:

- **PostgreSQL** - Relational data (users, topologies, modules, training results)
- **MinIO** - Object storage (models, datasets, logs, artifacts)
- **CouchDB** - Document storage (context, annotations, sessions, experiments)
- **Redis** (Optional) - Caching and session management

## PostgreSQL Integration

### Schema Design

```sql
-- Core entities
users → topologies → qlora_modules
     → datasets → semantic_chunks
     → training_results

-- Key relationships
- Users create topologies and datasets
- Topologies have modules and training results  
- Datasets contain semantic chunks
- All entities have audit timestamps
```

### Node.js Example

```javascript
const { TopologyDB } = require('./examples/nodejs/pg-example');

// Initialize connection
const db = new TopologyDB();
await db.connect();

// Create user and topology
const user = await db.createUser('developer', 'dev@example.com');
const topology = await db.createTopology(
  'my-transformer', 
  'Custom transformer model',
  'transformer',
  user.id,
  { layers: 6, dimensions: 512 }
);

// Track training results
const results = await db.saveTrainingResults(
  topology.id,
  { final_model: 'path/to/model.pt' },
  [0.8, 0.5, 0.3, 0.15], // loss values
  0.94, // accuracy
  4     // epochs
);
```

### Python Example

```python
from examples.python.pg_example import TopologyDB

# Use context manager for automatic connection handling
with TopologyDB() as db:
    # Create and query data
    user = db.create_user('scientist', 'scientist@example.com')
    
    topologies = db.get_topologies(user['id'])
    analytics = db.get_topology_analytics(topology_id)
    
    # Add semantic chunks for processed data
    chunk = db.create_semantic_chunk(
        dataset_id=dataset['id'],
        content="Machine learning fundamentals...",
        embedding=[0.1, 0.2, ...], # 384-dim vector
        metadata={'source': 'textbook', 'chapter': 1}
    )
```

## MinIO Integration

### Bucket Organization

```
topology-models/
├── models/
│   ├── transformer-v1.json
│   └── graph-network-v2.json
└── checkpoints/
    ├── topology-123/
    │   ├── epoch-5.pt
    │   └── epoch-10.pt

topology-datasets/
├── datasets/
│   ├── web-scrape-2024/
│   │   ├── chunk_001.jsonl
│   │   └── metadata.json

topology-logs/
├── logs/
│   ├── 2024-01-15/
│   │   ├── training-session-1.log
│   │   └── error-reports.log

topology-artifacts/
├── artifacts/
│   ├── experiment-results/
│   └── exported-modules/
```

### Node.js Example

```javascript
const { TopologyStorage } = require('./examples/nodejs/minio-example');

const storage = new TopologyStorage();
await storage.initializeBuckets();

// Upload model with metadata
const modelPath = await storage.uploadModel('my-model', {
  architecture: 'transformer',
  parameters: { layers: 6, dim: 512 },
  weights: modelWeights
}, {
  version: '1.0',
  accuracy: 0.94,
  compression_ratio: 0.1
});

// Upload dataset files
const files = [
  { name: 'chunk_001.jsonl', data: jsonlData, type: 'application/x-ndjson' },
  { name: 'metadata.json', data: JSON.stringify(metadata) }
];
await storage.uploadDataset('my-dataset', files);

// Generate presigned URLs for secure access
const url = await storage.getPresignedUrl(
  storage.buckets.models, 
  'models/my-model.json',
  3600 // 1 hour expiry
);
```

### Python Example

```python
from examples.python.minio_example import TopologyStorage

storage = TopologyStorage()
storage.initialize_buckets()

# Upload training checkpoint
checkpoint_data = {
    'epoch': 10,
    'loss': 0.123,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
checkpoint_path = storage.upload_checkpoint(
    'best-model-checkpoint',
    checkpoint_data,
    topology_id='topology-123'
)

# Download and use model
model_data = storage.download_model('my-model')
print(f"Model accuracy: {model_data['training_info']['accuracy']}")

# Bulk operations for large datasets
dataset_files = [
    {'name': f'chunk_{i:03d}.json', 'data': chunk_data}
    for i, chunk_data in enumerate(processed_chunks)
]
paths = storage.upload_dataset('processed-web-data', dataset_files)
```

## CouchDB Integration

### Document Design

```javascript
// Context Document
{
  "_id": "ctx_2024_001",
  "type": "engineering_note",
  "data": "Model shows overfitting after epoch 8...",
  "topology_id": "topology-123", 
  "metadata": {
    "priority": "high",
    "author": "ml-engineer",
    "tags": ["overfitting", "training"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}

// Experiment Document
{
  "_id": "exp_hyperopt_001",
  "name": "Hyperparameter Optimization",
  "topology_id": "topology-123",
  "status": "completed",
  "parameters": {
    "learning_rate": [0.001, 0.0001],
    "batch_size": [16, 32, 64]
  },
  "metrics": [
    {"epoch": 1, "val_acc": 0.82, "timestamp": "..."},
    {"epoch": 2, "val_acc": 0.89, "timestamp": "..."}
  ],
  "results": {
    "best_params": {"lr": 0.001, "batch_size": 32},
    "best_accuracy": 0.94
  }
}
```

### Node.js Example

```javascript
const { TopologyContext } = require('./examples/nodejs/couchdb-example');

const context = new TopologyContext();
await context.initializeDatabases();

// Add engineering context
const note = await context.addContext(
  'performance_insight',
  'Attention head 3 focuses on syntactic patterns while head 7 handles semantics',
  'topology-123',
  { analysis_type: 'attention_visualization', confidence: 0.89 }
);

// Track experiments
const experiment = await context.createExperiment(
  'Learning Rate Sweep',
  'topology-123',
  { learning_rates: [0.1, 0.01, 0.001, 0.0001] }
);

// Update with training metrics
await context.updateExperimentMetrics(experiment.id, {
  epoch: 5,
  train_loss: 0.234,
  val_loss: 0.267,
  learning_rate: 0.001
});

// Search across all context
const results = await context.searchContext('overfitting', 10);
```

### Python Example

```python
from examples.python.couchdb_example import TopologyContext

context_store = TopologyContext()
context_store.initialize_databases()

# Add structured context
context_store.add_context(
    'system_metrics',
    {
        'gpu_utilization': 0.87,
        'memory_usage_gb': 4.2,
        'temperature_c': 78
    },
    topology_id='topology-123',
    metadata={'alert_level': 'normal', 'source': 'nvidia-smi'}
)

# Track user sessions
session = context_store.create_session(
    user_id='user-456',
    session_type='model_development',
    metadata={'project': 'transformer-optimization'}
)

# Annotate models and results
context_store.add_annotation(
    target_id='topology-123',
    target_type='topology', 
    annotation='Excellent performance on classification, poor on generation',
    created_by='user-456',
    tags=['performance', 'classification', 'generation']
)

# Bulk insert for large context datasets
bulk_contexts = [
    {
        'type': 'metric',
        'data': {'throughput': 1250, 'latency_ms': 15},
        'topology_id': 'topology-123'
    }
    for _ in range(100)
]
context_store.bulk_insert_context(bulk_contexts)
```

## API Integration

### RESTful Endpoints

The `examples/nodejs/api-example.js` provides a complete Express.js API:

```javascript
// Start the API server
npm run api

// Example API calls
POST /api/topologies - Create topology
GET /api/topologies/123 - Get topology with context and analytics
POST /api/models - Upload model to MinIO
GET /api/context?topology_id=123 - Get context for topology
POST /api/annotations - Add annotation
GET /api/experiments/by_topology/123 - Get experiments
```

### API Usage Examples

```bash
# Create a new topology
curl -X POST http://localhost:3000/api/topologies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-transformer",
    "description": "Custom transformer model", 
    "architecture": "transformer",
    "created_by": 1,
    "config": {"layers": 6, "dimensions": 512}
  }'

# Upload a model
curl -X POST http://localhost:3000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "trained-model-v1",
    "model_data": {...},
    "metadata": {"accuracy": 0.94, "compression": 0.1}
  }'

# Add context
curl -X POST http://localhost:3000/api/context \
  -H "Content-Type: application/json" \
  -d '{
    "type": "training_note",
    "data": "Model converged after 50 epochs",
    "topology_id": "123",
    "metadata": {"priority": "medium"}
  }'

# Search context
curl "http://localhost:3000/api/context/search?q=overfitting&limit=5"

# Get topology analytics
curl "http://localhost:3000/api/analytics/topologies/123"
```

## Best Practices

### Data Consistency

```javascript
// Use transactions for related operations
await db.beginTransaction();
try {
  const topology = await db.createTopology(...);
  const module = await db.createQLoRAModule(..., topology.id);
  await context.addContext('topology_created', ..., topology.id);
  await db.commitTransaction();
} catch (error) {
  await db.rollbackTransaction();
  throw error;
}
```

### Error Handling

```javascript
// Robust error handling with retries
async function uploadModelWithRetry(name, data, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await storage.uploadModel(name, data);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await delay(1000 * (i + 1)); // Exponential backoff
    }
  }
}
```

### Performance Optimization

```javascript
// Use connection pooling
const pool = new Pool({
  host: 'localhost',
  database: 'topology',
  user: 'nexus',
  password: 'changeme',
  max: 20, // Maximum connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Batch operations for efficiency
const chunks = await Promise.all(
  chunkBatch.map(chunk => 
    db.addSemanticChunk(datasetId, chunk.content, chunk.embedding)
  )
);

// Use presigned URLs for large file transfers
const uploadUrl = await storage.getPresignedUrl(bucket, key, 3600);
// Client uploads directly to MinIO using the presigned URL
```

### Security Considerations

```javascript
// Input validation
function validateTopologyConfig(config) {
  const schema = Joi.object({
    layers: Joi.number().min(1).max(50).required(),
    dimensions: Joi.number().min(64).max(4096).required(),
    attention_heads: Joi.number().min(1).max(32).required()
  });
  return schema.validate(config);
}

// Sanitize user inputs
const sanitizedQuery = query.replace(/[<>]/g, '').trim();

// Use parameterized queries
const result = await db.query(
  'SELECT * FROM topologies WHERE created_by = $1 AND name ILIKE $2',
  [userId, `%${sanitizedName}%`]
);
```

## Monitoring and Logging

### Application Monitoring

```javascript
// Log all database operations
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Performance monitoring
const startTime = Date.now();
const result = await db.complexQuery();
const duration = Date.now() - startTime;
logger.info(`Query completed in ${duration}ms`);

// Upload logs to MinIO for analysis
await storage.uploadLog('app-performance', performanceData);
```

### Health Checks

```javascript
// Comprehensive health check
async function healthCheck() {
  const results = {};
  
  try {
    await db.query('SELECT 1');
    results.postgres = 'healthy';
  } catch (e) {
    results.postgres = 'unhealthy';
  }
  
  try {
    await storage.client.listBuckets();
    results.minio = 'healthy';
  } catch (e) {
    results.minio = 'unhealthy';
  }
  
  try {
    await context.session.get(context.base_url);
    results.couchdb = 'healthy';
  } catch (e) {
    results.couchdb = 'unhealthy';
  }
  
  return results;
}
```

## Migration and Deployment

### Database Migrations

```sql
-- migrations/001_initial_schema.sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- migrations/002_add_embedding_column.sql
ALTER TABLE semantic_chunks 
ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

CREATE INDEX IF NOT EXISTS idx_semantic_chunks_embedding 
ON semantic_chunks USING ivfflat (embedding vector_cosine_ops);
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: topology
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

secrets:
  postgres_password:
    external: true
    
volumes:
  postgres_prod_data:
    external: true
```

For complete working examples, see the files in `examples/nodejs/` and `examples/python/`.
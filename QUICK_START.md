# Quick Start Guide

Get up and running with the complete Topology Nexus App stack in minutes.

## 1. Start the Database Stack

```bash
# Start PostgreSQL, MinIO, and CouchDB
docker compose up -d

# Verify all services are running
docker compose ps
```

**Services will be available at:**
- PostgreSQL: `localhost:5432` 
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- CouchDB Admin: http://localhost:5984/_utils (admin/changeme)

## 2. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies  
pip install -r requirements.txt
```

## 3. Test Database Connections

```bash
# Test PostgreSQL integration
npm run example:pg

# Test MinIO object storage
npm run example:minio

# Test CouchDB document storage
npm run example:couchdb
```

## 4. Run the Original Demo

```bash
# Test core QLoRA functionality
python demo.py
```

## 5. Start the API Server

```bash
# Start the RESTful API
npm run api

# Access API documentation
open http://localhost:3000/api
```

## 6. Test the Complete Workflow

### Create a Topology

```bash
curl -X POST http://localhost:3000/api/topologies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-transformer",
    "description": "My custom transformer", 
    "architecture": "transformer",
    "created_by": 1,
    "config": {"layers": 6, "dimensions": 512}
  }'
```

### Upload a Model

```bash
curl -X POST http://localhost:3000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "trained-model",
    "model_data": {
      "architecture": "transformer",
      "weights": [0.1, 0.2, 0.3],
      "accuracy": 0.94
    },
    "metadata": {"version": "1.0", "compression": 0.1}
  }'
```

### Add Context

```bash
curl -X POST http://localhost:3000/api/context \
  -H "Content-Type: application/json" \
  -d '{
    "type": "engineering_note",
    "data": "Model shows excellent performance on classification tasks",
    "topology_id": "1",
    "metadata": {"priority": "high"}
  }'
```

### Search Context

```bash
curl "http://localhost:3000/api/context/search?q=performance&limit=5"
```

## 7. Explore the Web Interfaces

- **MinIO Console**: http://localhost:9001
  - Login: minioadmin / minioadmin
  - View uploaded models and datasets

- **CouchDB Admin**: http://localhost:5984/_utils  
  - Login: admin / changeme
  - Browse context documents and experiments

- **API Documentation**: http://localhost:3000/api
  - Complete list of available endpoints

## 8. Stop the Stack

```bash
# Stop all services
docker compose down

# Stop and remove volumes (careful - this deletes data!)
docker compose down -v
```

## What's Included

### 🗃️ **PostgreSQL Database**
- Users, topologies, QLoRA modules
- Training results and datasets  
- Semantic chunks with embeddings

### 📦 **MinIO Object Storage**
- Model files and checkpoints
- Dataset files and artifacts
- Training logs and exports

### 📄 **CouchDB Document Store**  
- Engineering context and insights
- User annotations and comments
- Experiment tracking and sessions

### 🚀 **RESTful API**
- Complete CRUD operations
- File upload/download
- Context search and analytics

### 🧠 **QLoRA Memory Management**
- Efficient module loading/unloading
- Context-aware memory management
- Compression and optimization

## Next Steps

- Read [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed configuration
- Check [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for advanced patterns  
- Explore the `examples/` directory for integration code
- Customize `config/default_config.json` for your needs

## Troubleshooting

### Port Conflicts
```bash
# Check what's using the ports
netstat -tlnp | grep -E ':(5432|9000|5984|3000)'
```

### Reset Everything
```bash
# Nuclear option - removes all containers and volumes
docker compose down -v
docker system prune -f
docker compose up -d
```

### Check Service Health
```bash
# View logs
docker compose logs -f

# Check specific service
docker compose logs postgres
docker compose logs minio
docker compose logs couchdb
```

### Test API Health  
```bash
curl http://localhost:3000/health
```

The complete system is now ready for contextual engineering with multi-database persistence!
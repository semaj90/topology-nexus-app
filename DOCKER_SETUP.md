# Docker Compose Setup for Topology Nexus App

This guide explains how to set up PostgreSQL, MinIO, and CouchDB using Docker Compose for the Topology Nexus App.

## Quick Start

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Check service status:**
   ```bash
   docker-compose ps
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop all services:**
   ```bash
   docker-compose down
   ```

## Service Access

### PostgreSQL
- **Host:** localhost
- **Port:** 5432
- **Database:** topology
- **Username:** nexus
- **Password:** changeme
- **Connection URL:** `postgresql://nexus:changeme@localhost:5432/topology`

### MinIO Object Storage
- **API Endpoint:** http://localhost:9000
- **Console:** http://localhost:9001
- **Access Key:** minioadmin
- **Secret Key:** minioadmin
- **Web Console Login:** minioadmin / minioadmin

### CouchDB
- **API Endpoint:** http://localhost:5984
- **Admin Interface:** http://localhost:5984/_utils
- **Username:** admin
- **Password:** changeme

### Redis (Optional)
- **Host:** localhost
- **Port:** 6379
- **No password required**

### Nginx Reverse Proxy
- **Main Page:** http://localhost
- **MinIO Console:** http://console.localhost
- **CouchDB Admin:** http://couchdb.localhost
- **MinIO S3 API:** http://s3.localhost

## Database Schemas

### PostgreSQL Schema

The PostgreSQL database is automatically initialized with the following schema:

#### Tables:
- **users** - User accounts and authentication
- **topologies** - Neural network topology definitions
- **qlora_modules** - QLoRA module metadata and file paths
- **training_results** - Training metrics and results
- **datasets** - Dataset metadata and statistics
- **semantic_chunks** - Processed text chunks with embeddings

#### Key Relationships:
- Users create topologies and datasets
- Topologies have associated QLoRA modules and training results
- Datasets contain semantic chunks for processing

### MinIO Buckets

The following buckets are automatically created:

- **topology-models** - Trained model files and checkpoints
- **topology-datasets** - Raw and processed dataset files
- **topology-logs** - Training and application logs
- **topology-artifacts** - Experiment results and reports
- **topology-checkpoints** - Training checkpoints and snapshots

### CouchDB Databases

The following databases are created with design documents:

- **topology-context** - Engineering notes, insights, and contextual information
- **topology-annotations** - User annotations and comments
- **topology-sessions** - User sessions and activity tracking  
- **topology-experiments** - Experiment tracking and metrics

## Configuration

### Environment Variables

You can customize the setup by setting environment variables:

```bash
# PostgreSQL
POSTGRES_DB=topology
POSTGRES_USER=nexus
POSTGRES_PASSWORD=changeme

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# CouchDB
COUCHDB_USER=admin
COUCHDB_PASSWORD=changeme
```

### Custom Configuration

Edit `config/default_config.json` to customize database connections:

```json
{
  "databases": {
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "topology",
      "user": "nexus",
      "password": "changeme"
    },
    "minio": {
      "endpoint": "localhost:9000",
      "access_key": "minioadmin",
      "secret_key": "minioadmin",
      "secure": false
    },
    "couchdb": {
      "base_url": "http://localhost:5984",
      "username": "admin",
      "password": "changeme"
    }
  }
}
```

## Data Persistence

All data is persisted in Docker volumes:

- `postgres_data` - PostgreSQL database files
- `minio_data` - MinIO object storage
- `couchdb_data` - CouchDB database files
- `redis_data` - Redis cache data

### Backup and Restore

#### PostgreSQL Backup:
```bash
docker exec topology-nexus-app_postgres_1 pg_dump -U nexus topology > backup.sql
```

#### PostgreSQL Restore:
```bash
docker exec -i topology-nexus-app_postgres_1 psql -U nexus topology < backup.sql
```

#### MinIO Backup:
```bash
# Use MinIO client (mc) for backup
docker exec topology-nexus-app_minio_1 mc mirror /data /backup
```

#### CouchDB Backup:
```bash
# Create backup of all databases
curl -X GET http://admin:changeme@localhost:5984/_all_dbs | \
jq -r '.[]' | \
xargs -I {} curl -X GET "http://admin:changeme@localhost:5984/{}/_all_docs?include_docs=true" > couchdb_backup.json
```

## Monitoring and Health Checks

### Health Check Endpoints

- **PostgreSQL:** Connect using `psql` or database client
- **MinIO:** http://localhost:9000/minio/health/live
- **CouchDB:** http://admin:changeme@localhost:5984/
- **Redis:** `redis-cli ping`
- **Nginx:** http://localhost/health

### Container Health Status

```bash
# Check all container health
docker-compose ps

# Check specific service logs
docker-compose logs postgres
docker-compose logs minio
docker-compose logs couchdb
```

### Resource Usage

```bash
# Monitor resource usage
docker stats

# Check volume usage
docker system df
docker volume ls
```

## Security Considerations

### For Production Use:

1. **Change Default Passwords:**
   - Update PostgreSQL password
   - Change MinIO access keys
   - Set strong CouchDB admin password

2. **Enable SSL/TLS:**
   - Configure MinIO with TLS certificates
   - Enable PostgreSQL SSL
   - Set up HTTPS for CouchDB

3. **Network Security:**
   - Use internal Docker networks
   - Configure firewall rules
   - Implement VPN access

4. **Backup Strategy:**
   - Regular automated backups
   - Off-site backup storage
   - Test restore procedures

## Troubleshooting

### Common Issues:

1. **Port Conflicts:**
   ```bash
   # Check what's using ports
   netstat -tlnp | grep -E ':(5432|9000|5984|6379|80)'
   
   # Stop conflicting services
   sudo systemctl stop postgresql
   sudo systemctl stop redis
   ```

2. **Permission Issues:**
   ```bash
   # Fix Docker permissions
   sudo chown -R $USER:$USER docker/
   
   # Reset volumes if needed
   docker-compose down -v
   docker-compose up -d
   ```

3. **Memory Issues:**
   ```bash
   # Increase Docker memory limit
   # Edit Docker Desktop settings or /etc/docker/daemon.json
   
   # Check container memory usage
   docker stats --no-stream
   ```

4. **Connection Issues:**
   ```bash
   # Test database connections
   npm run example:pg
   npm run example:minio  
   npm run example:couchdb
   ```

### Useful Commands:

```bash
# Restart specific service
docker-compose restart postgres

# View real-time logs
docker-compose logs -f --tail=100

# Execute commands in containers
docker-compose exec postgres psql -U nexus topology
docker-compose exec minio mc admin info local
docker-compose exec couchdb curl http://localhost:5984

# Clean up everything
docker-compose down -v --remove-orphans
docker system prune -f
```

## Next Steps

1. **Run Example Applications:**
   ```bash
   npm install
   npm run example:pg
   npm run example:minio
   npm run example:couchdb
   ```

2. **Start API Server:**
   ```bash
   npm run api
   ```

3. **Explore Web Interfaces:**
   - MinIO Console: http://localhost:9001
   - CouchDB Admin: http://localhost:5984/_utils
   - API Documentation: http://localhost:3000/api

For more detailed integration examples, see the files in `examples/nodejs/` and `examples/python/`.
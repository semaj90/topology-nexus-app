-- Initialize Topology Nexus Database Schema

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Topologies table
CREATE TABLE IF NOT EXISTS topologies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    architecture VARCHAR(50) NOT NULL, -- transformer, graph, hierarchical, hybrid
    created_by INTEGER REFERENCES users(id),
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- QLoRA Modules table
CREATE TABLE IF NOT EXISTS qlora_modules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    topology_id INTEGER REFERENCES topologies(id),
    config_json JSONB,
    file_path VARCHAR(255), -- Path to module file in MinIO
    compression_ratio FLOAT DEFAULT 0.1,
    memory_size_mb INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training Results table
CREATE TABLE IF NOT EXISTS training_results (
    id SERIAL PRIMARY KEY,
    topology_id INTEGER REFERENCES topologies(id),
    result_json JSONB,
    loss_values FLOAT[],
    accuracy FLOAT,
    epochs INTEGER,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    source_urls TEXT[],
    created_by INTEGER REFERENCES users(id),
    statistics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Semantic Chunks table (for processed text data)
CREATE TABLE IF NOT EXISTS semantic_chunks (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    content TEXT NOT NULL,
    embedding VECTOR(384), -- For sentence-transformers/all-MiniLM-L6-v2
    metadata JSONB,
    chunk_index INTEGER,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_topologies_architecture ON topologies(architecture);
CREATE INDEX IF NOT EXISTS idx_topologies_created_by ON topologies(created_by);
CREATE INDEX IF NOT EXISTS idx_qlora_modules_topology_id ON qlora_modules(topology_id);
CREATE INDEX IF NOT EXISTS idx_training_results_topology_id ON training_results(topology_id);
CREATE INDEX IF NOT EXISTS idx_semantic_chunks_dataset_id ON semantic_chunks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_datasets_created_by ON datasets(created_by);

-- Insert sample data
INSERT INTO users (username, email) VALUES 
    ('admin', 'admin@topology-nexus.local'),
    ('demo_user', 'demo@topology-nexus.local')
ON CONFLICT (username) DO NOTHING;

INSERT INTO topologies (name, description, architecture, created_by, config) VALUES 
    ('default_transformer', 'Default transformer topology for general use', 'transformer', 1, 
     '{"dimensions": 512, "layers": 6, "attention_heads": 8, "dropout": 0.1}'),
    ('graph_network', 'Graph neural network for structured data', 'graph', 1,
     '{"dimensions": 256, "layers": 4, "node_features": 128, "edge_features": 64}')
ON CONFLICT DO NOTHING;

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_topologies_updated_at BEFORE UPDATE ON topologies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_qlora_modules_updated_at BEFORE UPDATE ON qlora_modules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
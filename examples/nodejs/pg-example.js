const { Pool } = require('pg');

/**
 * PostgreSQL Integration Example for Topology Nexus App
 * Demonstrates CRUD operations for topologies, modules, and datasets
 */

const pool = new Pool({
  user: 'nexus',
  host: 'localhost',
  database: 'topology',
  password: 'changeme',
  port: 5432,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

class TopologyDB {
  constructor() {
    this.pool = pool;
  }

  // User operations
  async createUser(username, email) {
    const query = 'INSERT INTO users (username, email) VALUES ($1, $2) RETURNING id, username, email, created_at';
    try {
      const result = await this.pool.query(query, [username, email]);
      return result.rows[0];
    } catch (error) {
      console.error('Error creating user:', error);
      throw error;
    }
  }

  async getUser(userId) {
    const query = 'SELECT * FROM users WHERE id = $1';
    try {
      const result = await this.pool.query(query, [userId]);
      return result.rows[0];
    } catch (error) {
      console.error('Error fetching user:', error);
      throw error;
    }
  }

  // Topology operations
  async createTopology(name, description, architecture, createdBy, config = {}) {
    const query = `
      INSERT INTO topologies (name, description, architecture, created_by, config) 
      VALUES ($1, $2, $3, $4, $5) 
      RETURNING id, name, architecture, created_at
    `;
    try {
      const result = await this.pool.query(query, [name, description, architecture, createdBy, JSON.stringify(config)]);
      return result.rows[0];
    } catch (error) {
      console.error('Error creating topology:', error);
      throw error;
    }
  }

  async getTopologies(userId = null) {
    let query = 'SELECT t.*, u.username FROM topologies t LEFT JOIN users u ON t.created_by = u.id';
    let params = [];

    if (userId) {
      query += ' WHERE t.created_by = $1';
      params = [userId];
    }

    query += ' ORDER BY t.created_at DESC';

    try {
      const result = await this.pool.query(query, params);
      return result.rows;
    } catch (error) {
      console.error('Error fetching topologies:', error);
      throw error;
    }
  }

  // QLoRA Module operations
  async createQLoRAModule(name, topologyId, configJson, filePath, compressionRatio = 0.1, memorySizeMb) {
    const query = `
      INSERT INTO qlora_modules (name, topology_id, config_json, file_path, compression_ratio, memory_size_mb)
      VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING *
    `;
    try {
      const result = await this.pool.query(query, [
        name, topologyId, JSON.stringify(configJson), filePath, compressionRatio, memorySizeMb
      ]);
      return result.rows[0];
    } catch (error) {
      console.error('Error creating QLoRA module:', error);
      throw error;
    }
  }

  async getQLoRAModules(topologyId) {
    const query = 'SELECT * FROM qlora_modules WHERE topology_id = $1 ORDER BY created_at DESC';
    try {
      const result = await this.pool.query(query, [topologyId]);
      return result.rows;
    } catch (error) {
      console.error('Error fetching QLoRA modules:', error);
      throw error;
    }
  }

  // Dataset operations
  async createDataset(name, description, sourceUrls, createdBy, statistics = {}) {
    const query = `
      INSERT INTO datasets (name, description, source_urls, created_by, statistics)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;
    try {
      const result = await this.pool.query(query, [
        name, description, sourceUrls, createdBy, JSON.stringify(statistics)
      ]);
      return result.rows[0];
    } catch (error) {
      console.error('Error creating dataset:', error);
      throw error;
    }
  }

  // Training results operations
  async saveTrainingResults(topologyId, resultJson, lossValues, accuracy, epochs) {
    const query = `
      INSERT INTO training_results (topology_id, result_json, loss_values, accuracy, epochs)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING *
    `;
    try {
      const result = await this.pool.query(query, [
        topologyId, JSON.stringify(resultJson), lossValues, accuracy, epochs
      ]);
      return result.rows[0];
    } catch (error) {
      console.error('Error saving training results:', error);
      throw error;
    }
  }

  async getTrainingHistory(topologyId) {
    const query = 'SELECT * FROM training_results WHERE topology_id = $1 ORDER BY completed_at DESC';
    try {
      const result = await this.pool.query(query, [topologyId]);
      return result.rows;
    } catch (error) {
      console.error('Error fetching training history:', error);
      throw error;
    }
  }

  // Close connection
  async close() {
    await this.pool.end();
  }
}

// Example usage
async function demonstrateUsage() {
  const db = new TopologyDB();
  
  try {
    console.log('=== PostgreSQL Integration Demo ===\n');

    // Create a user
    const user = await db.createUser('test_user', 'test@example.com');
    console.log('Created user:', user);

    // Create a topology
    const topology = await db.createTopology(
      'demo_transformer', 
      'Demo transformer for testing', 
      'transformer', 
      user.id,
      { dimensions: 512, layers: 6, attention_heads: 8 }
    );
    console.log('Created topology:', topology);

    // Create a QLoRA module
    const module = await db.createQLoRAModule(
      'demo_module',
      topology.id,
      { type: 'classification', classes: 10 },
      'modules/demo_module.json',
      0.1,
      256
    );
    console.log('Created QLoRA module:', module);

    // Create a dataset
    const dataset = await db.createDataset(
      'demo_dataset',
      'Demo dataset for testing',
      ['https://example.com/data1', 'https://example.com/data2'],
      user.id,
      { chunks: 100, total_size: '10MB' }
    );
    console.log('Created dataset:', dataset);

    // Save training results
    const trainingResult = await db.saveTrainingResults(
      topology.id,
      { final_loss: 0.1234, metrics: { accuracy: 0.95 } },
      [0.5, 0.3, 0.2, 0.1234],
      0.95,
      4
    );
    console.log('Saved training results:', trainingResult);

    // Fetch all topologies
    const topologies = await db.getTopologies();
    console.log('All topologies:', topologies);

    console.log('\n=== Demo completed successfully! ===');

  } catch (error) {
    console.error('Demo failed:', error);
  } finally {
    await db.close();
  }
}

module.exports = { TopologyDB, demonstrateUsage };

// Run demo if this file is executed directly
if (require.main === module) {
  demonstrateUsage();
}
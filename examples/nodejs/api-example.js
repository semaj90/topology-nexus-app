const express = require('express');
const { TopologyDB } = require('./pg-example');
const { TopologyStorage } = require('./minio-example');
const { TopologyContext } = require('./couchdb-example');

/**
 * Express API Example for Topology Nexus App
 * Demonstrates RESTful API endpoints integrating PostgreSQL, MinIO, and CouchDB
 */

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Initialize database connections
let topologyDB, topologyStorage, topologyContext;

async function initializeServices() {
  try {
    topologyDB = new TopologyDB();
    await topologyDB.connect();
    
    topologyStorage = new TopologyStorage();
    await topologyStorage.initializeBuckets();
    
    topologyContext = new TopologyContext();
    await topologyContext.initializeDatabases();
    
    console.log('All services initialized successfully');
  } catch (error) {
    console.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    services: {
      postgres: 'connected',
      minio: 'connected',
      couchdb: 'connected'
    }
  });
});

// User endpoints
app.post('/api/users', async (req, res) => {
  try {
    const { username, email } = req.body;
    
    if (!username || !email) {
      return res.status(400).json({ error: 'Username and email are required' });
    }
    
    const user = await topologyDB.createUser(username, email);
    res.status(201).json(user);
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ error: 'Failed to create user' });
  }
});

app.get('/api/users/:id', async (req, res) => {
  try {
    const user = await topologyDB.getUser(parseInt(req.params.id));
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(user);
  } catch (error) {
    console.error('Error fetching user:', error);
    res.status(500).json({ error: 'Failed to fetch user' });
  }
});

// Topology endpoints
app.post('/api/topologies', async (req, res) => {
  try {
    const { name, description, architecture, created_by, config } = req.body;
    
    if (!name || !architecture || !created_by) {
      return res.status(400).json({ 
        error: 'Name, architecture, and created_by are required' 
      });
    }
    
    const topology = await topologyDB.createTopology(
      name, 
      description || '', 
      architecture, 
      created_by, 
      config || {}
    );
    
    // Add context entry for topology creation
    await topologyContext.addContext(
      'topology_created',
      `New ${architecture} topology '${name}' created`,
      topology.id.toString(),
      { 
        creator_id: created_by,
        architecture,
        config_params: Object.keys(config || {}).length
      }
    );
    
    res.status(201).json(topology);
  } catch (error) {
    console.error('Error creating topology:', error);
    res.status(500).json({ error: 'Failed to create topology' });
  }
});

app.get('/api/topologies', async (req, res) => {
  try {
    const { user_id } = req.query;
    const userId = user_id ? parseInt(user_id) : null;
    
    const topologies = await topologyDB.getTopologies(userId);
    res.json(topologies);
  } catch (error) {
    console.error('Error fetching topologies:', error);
    res.status(500).json({ error: 'Failed to fetch topologies' });
  }
});

app.get('/api/topologies/:id', async (req, res) => {
  try {
    const topology = await topologyDB.getTopology(parseInt(req.params.id));
    
    if (!topology) {
      return res.status(404).json({ error: 'Topology not found' });
    }
    
    // Get related context and analytics
    const [context, analytics] = await Promise.all([
      topologyContext.getContextByTopology(req.params.id, 10),
      topologyDB.getTopologyAnalytics(parseInt(req.params.id))
    ]);
    
    res.json({
      ...topology,
      recent_context: context,
      analytics
    });
  } catch (error) {
    console.error('Error fetching topology:', error);
    res.status(500).json({ error: 'Failed to fetch topology' });
  }
});

app.put('/api/topologies/:id', async (req, res) => {
  try {
    const { name, description, architecture, config } = req.body;
    const updates = {};
    
    if (name) updates.name = name;
    if (description) updates.description = description;
    if (architecture) updates.architecture = architecture;
    if (config) updates.config = config;
    
    const success = await topologyDB.updateTopology(parseInt(req.params.id), updates);
    
    if (!success) {
      return res.status(404).json({ error: 'Topology not found' });
    }
    
    // Log the update
    await topologyContext.addContext(
      'topology_updated',
      `Topology updated with changes: ${Object.keys(updates).join(', ')}`,
      req.params.id,
      { updated_fields: Object.keys(updates) }
    );
    
    const updatedTopology = await topologyDB.getTopology(parseInt(req.params.id));
    res.json(updatedTopology);
  } catch (error) {
    console.error('Error updating topology:', error);
    res.status(500).json({ error: 'Failed to update topology' });
  }
});

// QLoRA Module endpoints
app.post('/api/topologies/:id/modules', async (req, res) => {
  try {
    const { name, config_json, file_path, compression_ratio, memory_size_mb } = req.body;
    const topologyId = parseInt(req.params.id);
    
    if (!name || !config_json) {
      return res.status(400).json({ 
        error: 'Name and config_json are required' 
      });
    }
    
    const module = await topologyDB.createQLoRAModule(
      name,
      topologyId,
      config_json,
      file_path || `modules/${name}.json`,
      compression_ratio || 0.1,
      memory_size_mb
    );
    
    // Add context for module creation
    await topologyContext.addContext(
      'module_created',
      `QLoRA module '${name}' created for topology ${topologyId}`,
      topologyId.toString(),
      { 
        module_id: module.id,
        module_name: name,
        compression_ratio: module.compression_ratio
      }
    );
    
    res.status(201).json(module);
  } catch (error) {
    console.error('Error creating QLoRA module:', error);
    res.status(500).json({ error: 'Failed to create QLoRA module' });
  }
});

app.get('/api/topologies/:id/modules', async (req, res) => {
  try {
    const modules = await topologyDB.getQLoRAModules(parseInt(req.params.id));
    res.json(modules);
  } catch (error) {
    console.error('Error fetching QLoRA modules:', error);
    res.status(500).json({ error: 'Failed to fetch QLoRA modules' });
  }
});

// Model storage endpoints
app.post('/api/models', async (req, res) => {
  try {
    const { name, model_data, metadata } = req.body;
    
    if (!name || !model_data) {
      return res.status(400).json({ 
        error: 'Name and model_data are required' 
      });
    }
    
    const modelPath = await topologyStorage.uploadModel(name, model_data, metadata);
    
    res.status(201).json({
      name,
      path: modelPath,
      uploaded_at: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error uploading model:', error);
    res.status(500).json({ error: 'Failed to upload model' });
  }
});

app.get('/api/models/:name', async (req, res) => {
  try {
    const model = await topologyStorage.downloadModel(req.params.name);
    res.json(model);
  } catch (error) {
    console.error('Error downloading model:', error);
    res.status(404).json({ error: 'Model not found' });
  }
});

app.get('/api/models', async (req, res) => {
  try {
    const models = await topologyStorage.listModels();
    res.json(models);
  } catch (error) {
    console.error('Error listing models:', error);
    res.status(500).json({ error: 'Failed to list models' });
  }
});

app.get('/api/models/:name/url', async (req, res) => {
  try {
    const { expiry } = req.query;
    const expirySeconds = expiry ? parseInt(expiry) : 3600;
    
    const url = await topologyStorage.getPresignedUrl(
      topologyStorage.buckets.models,
      `models/${req.params.name}.json`,
      expirySeconds
    );
    
    res.json({
      url,
      expires_in: expirySeconds,
      expires_at: new Date(Date.now() + expirySeconds * 1000).toISOString()
    });
  } catch (error) {
    console.error('Error generating presigned URL:', error);
    res.status(500).json({ error: 'Failed to generate presigned URL' });
  }
});

// Dataset endpoints
app.post('/api/datasets', async (req, res) => {
  try {
    const { name, description, source_urls, created_by, statistics, files } = req.body;
    
    if (!name || !created_by) {
      return res.status(400).json({ 
        error: 'Name and created_by are required' 
      });
    }
    
    // Create dataset record in PostgreSQL
    const dataset = await topologyDB.createDataset(
      name,
      description || '',
      source_urls || [],
      created_by,
      statistics || {}
    );
    
    // Upload files to MinIO if provided
    let uploadedFiles = [];
    if (files && files.length > 0) {
      uploadedFiles = await topologyStorage.uploadDataset(name, files);
    }
    
    // Add context for dataset creation
    await topologyContext.addContext(
      'dataset_created',
      `Dataset '${name}' created with ${files ? files.length : 0} files`,
      null,
      { 
        dataset_id: dataset.id,
        dataset_name: name,
        file_count: files ? files.length : 0,
        creator_id: created_by
      }
    );
    
    res.status(201).json({
      ...dataset,
      uploaded_files: uploadedFiles
    });
  } catch (error) {
    console.error('Error creating dataset:', error);
    res.status(500).json({ error: 'Failed to create dataset' });
  }
});

app.get('/api/datasets', async (req, res) => {
  try {
    const { user_id } = req.query;
    const userId = user_id ? parseInt(user_id) : null;
    
    const datasets = await topologyDB.getDatasets(userId);
    res.json(datasets);
  } catch (error) {
    console.error('Error fetching datasets:', error);
    res.status(500).json({ error: 'Failed to fetch datasets' });
  }
});

app.get('/api/datasets/:name/files', async (req, res) => {
  try {
    const files = await topologyStorage.listDatasetFiles(req.params.name);
    res.json(files);
  } catch (error) {
    console.error('Error listing dataset files:', error);
    res.status(500).json({ error: 'Failed to list dataset files' });
  }
});

// Training endpoints
app.post('/api/topologies/:id/training', async (req, res) => {
  try {
    const { result_json, loss_values, accuracy, epochs } = req.body;
    const topologyId = parseInt(req.params.id);
    
    if (!result_json || !loss_values || accuracy === undefined || !epochs) {
      return res.status(400).json({ 
        error: 'result_json, loss_values, accuracy, and epochs are required' 
      });
    }
    
    const trainingResult = await topologyDB.saveTrainingResults(
      topologyId,
      result_json,
      loss_values,
      accuracy,
      epochs
    );
    
    // Add context for training completion
    await topologyContext.addContext(
      'training_completed',
      `Training completed for topology ${topologyId}. Final accuracy: ${accuracy}, Final loss: ${loss_values[loss_values.length - 1]}`,
      topologyId.toString(),
      { 
        final_accuracy: accuracy,
        final_loss: loss_values[loss_values.length - 1],
        epochs: epochs,
        training_id: trainingResult.id
      }
    );
    
    res.status(201).json(trainingResult);
  } catch (error) {
    console.error('Error saving training results:', error);
    res.status(500).json({ error: 'Failed to save training results' });
  }
});

app.get('/api/topologies/:id/training', async (req, res) => {
  try {
    const trainingHistory = await topologyDB.getTrainingHistory(parseInt(req.params.id));
    res.json(trainingHistory);
  } catch (error) {
    console.error('Error fetching training history:', error);
    res.status(500).json({ error: 'Failed to fetch training history' });
  }
});

app.get('/api/topologies/:id/training/best', async (req, res) => {
  try {
    const bestResult = await topologyDB.getBestTrainingResult(parseInt(req.params.id));
    
    if (!bestResult) {
      return res.status(404).json({ error: 'No training results found' });
    }
    
    res.json(bestResult);
  } catch (error) {
    console.error('Error fetching best training result:', error);
    res.status(500).json({ error: 'Failed to fetch best training result' });
  }
});

// Context endpoints
app.post('/api/context', async (req, res) => {
  try {
    const { type, data, topology_id, metadata } = req.body;
    
    if (!type || !data) {
      return res.status(400).json({ 
        error: 'Type and data are required' 
      });
    }
    
    const context = await topologyContext.addContext(type, data, topology_id, metadata);
    res.status(201).json(context);
  } catch (error) {
    console.error('Error adding context:', error);
    res.status(500).json({ error: 'Failed to add context' });
  }
});

app.get('/api/context', async (req, res) => {
  try {
    const { type, topology_id, limit = 20 } = req.query;
    
    let contexts;
    if (type) {
      contexts = await topologyContext.getContextByType(type, parseInt(limit));
    } else if (topology_id) {
      contexts = await topologyContext.getContextByTopology(topology_id, parseInt(limit));
    } else {
      return res.status(400).json({ 
        error: 'Either type or topology_id parameter is required' 
      });
    }
    
    res.json(contexts);
  } catch (error) {
    console.error('Error fetching context:', error);
    res.status(500).json({ error: 'Failed to fetch context' });
  }
});

app.get('/api/context/search', async (req, res) => {
  try {
    const { q, limit = 20 } = req.query;
    
    if (!q) {
      return res.status(400).json({ error: 'Query parameter q is required' });
    }
    
    const results = await topologyContext.searchContext(q, parseInt(limit));
    res.json(results);
  } catch (error) {
    console.error('Error searching context:', error);
    res.status(500).json({ error: 'Failed to search context' });
  }
});

// Annotation endpoints
app.post('/api/annotations', async (req, res) => {
  try {
    const { target_id, target_type, annotation, created_by, tags } = req.body;
    
    if (!target_id || !target_type || !annotation || !created_by) {
      return res.status(400).json({ 
        error: 'target_id, target_type, annotation, and created_by are required' 
      });
    }
    
    const annotationDoc = await topologyContext.addAnnotation(
      target_id,
      target_type,
      annotation,
      created_by,
      tags
    );
    
    res.status(201).json(annotationDoc);
  } catch (error) {
    console.error('Error adding annotation:', error);
    res.status(500).json({ error: 'Failed to add annotation' });
  }
});

app.get('/api/annotations/:target_id', async (req, res) => {
  try {
    const annotations = await topologyContext.getAnnotations(req.params.target_id);
    res.json(annotations);
  } catch (error) {
    console.error('Error fetching annotations:', error);
    res.status(500).json({ error: 'Failed to fetch annotations' });
  }
});

// Session endpoints
app.post('/api/sessions', async (req, res) => {
  try {
    const { user_id, session_type, metadata } = req.body;
    
    if (!user_id) {
      return res.status(400).json({ error: 'user_id is required' });
    }
    
    const session = await topologyContext.createSession(
      user_id,
      session_type || 'general',
      metadata
    );
    
    res.status(201).json(session);
  } catch (error) {
    console.error('Error creating session:', error);
    res.status(500).json({ error: 'Failed to create session' });
  }
});

app.put('/api/sessions/:id', async (req, res) => {
  try {
    const updates = req.body;
    const updatedSession = await topologyContext.updateSession(req.params.id, updates);
    res.json(updatedSession);
  } catch (error) {
    console.error('Error updating session:', error);
    res.status(500).json({ error: 'Failed to update session' });
  }
});

app.delete('/api/sessions/:id', async (req, res) => {
  try {
    const endedSession = await topologyContext.endSession(req.params.id);
    res.json(endedSession);
  } catch (error) {
    console.error('Error ending session:', error);
    res.status(500).json({ error: 'Failed to end session' });
  }
});

app.get('/api/sessions/active', async (req, res) => {
  try {
    const { user_id } = req.query;
    const activeSessions = await topologyContext.getActiveSessions(user_id);
    res.json(activeSessions);
  } catch (error) {
    console.error('Error fetching active sessions:', error);
    res.status(500).json({ error: 'Failed to fetch active sessions' });
  }
});

// Analytics endpoints
app.get('/api/analytics/topologies/:id', async (req, res) => {
  try {
    const analytics = await topologyDB.getTopologyAnalytics(parseInt(req.params.id));
    res.json(analytics);
  } catch (error) {
    console.error('Error fetching topology analytics:', error);
    res.status(500).json({ error: 'Failed to fetch topology analytics' });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully...');
  
  if (topologyDB) {
    await topologyDB.disconnect();
  }
  
  process.exit(0);
});

// Start server
async function startServer() {
  await initializeServices();
  
  app.listen(port, () => {
    console.log(`Topology Nexus API server running on port ${port}`);
    console.log(`Health check: http://localhost:${port}/health`);
    console.log(`API docs: http://localhost:${port}/api`);
  });
}

// API documentation endpoint
app.get('/api', (req, res) => {
  res.json({
    name: 'Topology Nexus API',
    version: '1.0.0',
    description: 'RESTful API for Topology Nexus App with PostgreSQL, MinIO, and CouchDB integration',
    endpoints: {
      health: 'GET /health',
      users: {
        create: 'POST /api/users',
        get: 'GET /api/users/:id'
      },
      topologies: {
        create: 'POST /api/topologies',
        list: 'GET /api/topologies',
        get: 'GET /api/topologies/:id',
        update: 'PUT /api/topologies/:id'
      },
      modules: {
        create: 'POST /api/topologies/:id/modules',
        list: 'GET /api/topologies/:id/modules'
      },
      models: {
        upload: 'POST /api/models',
        download: 'GET /api/models/:name',
        list: 'GET /api/models',
        presigned_url: 'GET /api/models/:name/url'
      },
      datasets: {
        create: 'POST /api/datasets',
        list: 'GET /api/datasets',
        files: 'GET /api/datasets/:name/files'
      },
      training: {
        save: 'POST /api/topologies/:id/training',
        history: 'GET /api/topologies/:id/training',
        best: 'GET /api/topologies/:id/training/best'
      },
      context: {
        add: 'POST /api/context',
        get: 'GET /api/context',
        search: 'GET /api/context/search'
      },
      annotations: {
        add: 'POST /api/annotations',
        get: 'GET /api/annotations/:target_id'
      },
      sessions: {
        create: 'POST /api/sessions',
        update: 'PUT /api/sessions/:id',
        end: 'DELETE /api/sessions/:id',
        active: 'GET /api/sessions/active'
      },
      analytics: {
        topology: 'GET /api/analytics/topologies/:id'
      }
    }
  });
});

if (require.main === module) {
  startServer().catch(console.error);
}

module.exports = { app, startServer };
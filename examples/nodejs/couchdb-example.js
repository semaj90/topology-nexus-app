const nano = require('nano');

/**
 * CouchDB Integration Example for Topology Nexus App
 * Demonstrates document storage for context, annotations, and metadata
 */

const couchClient = nano('http://admin:changeme@localhost:5984');

class TopologyContext {
  constructor() {
    this.client = couchClient;
    this.databases = {
      context: 'topology-context',
      annotations: 'topology-annotations',
      sessions: 'topology-sessions',
      experiments: 'topology-experiments'
    };
  }

  // Initialize databases
  async initializeDatabases() {
    console.log('Initializing CouchDB databases...');
    
    for (const [name, dbName] of Object.entries(this.databases)) {
      try {
        const db = this.client.db.use(dbName);
        await db.info();
        console.log(`Database already exists: ${dbName}`);
      } catch (error) {
        if (error.statusCode === 404) {
          try {
            await this.client.db.create(dbName);
            console.log(`Created database: ${dbName}`);
            
            // Create initial design documents for each database
            await this.createDesignDocuments(dbName);
          } catch (createError) {
            console.error(`Error creating database ${dbName}:`, createError);
          }
        } else {
          console.error(`Error checking database ${dbName}:`, error);
        }
      }
    }
  }

  // Create design documents with views and indexes
  async createDesignDocuments(dbName) {
    const db = this.client.db.use(dbName);
    
    const designDocs = {
      'topology-context': {
        _id: '_design/context',
        views: {
          by_type: {
            map: "function(doc) { if(doc.type) emit(doc.type, doc); }"
          },
          by_topology_id: {
            map: "function(doc) { if(doc.topology_id) emit(doc.topology_id, doc); }"
          },
          by_timestamp: {
            map: "function(doc) { if(doc.timestamp) emit(doc.timestamp, doc); }"
          }
        }
      },
      'topology-annotations': {
        _id: '_design/annotations',
        views: {
          by_target: {
            map: "function(doc) { if(doc.target_id) emit(doc.target_id, doc); }"
          },
          by_user: {
            map: "function(doc) { if(doc.created_by) emit(doc.created_by, doc); }"
          }
        }
      },
      'topology-sessions': {
        _id: '_design/sessions',
        views: {
          by_user: {
            map: "function(doc) { if(doc.user_id) emit(doc.user_id, doc); }"
          },
          active_sessions: {
            map: "function(doc) { if(doc.status === 'active') emit(doc.started_at, doc); }"
          }
        }
      },
      'topology-experiments': {
        _id: '_design/experiments',
        views: {
          by_status: {
            map: "function(doc) { if(doc.status) emit(doc.status, doc); }"
          },
          by_topology: {
            map: "function(doc) { if(doc.topology_id) emit(doc.topology_id, doc); }"
          }
        }
      }
    };

    if (designDocs[dbName]) {
      try {
        await db.insert(designDocs[dbName]);
        console.log(`Created design document for ${dbName}`);
      } catch (error) {
        if (error.statusCode !== 409) { // 409 = conflict (already exists)
          console.error(`Error creating design document for ${dbName}:`, error);
        }
      }
    }
  }

  // Context operations
  async addContext(type, data, topologyId = null, metadata = {}) {
    const db = this.client.db.use(this.databases.context);
    
    const doc = {
      type,
      data,
      topology_id: topologyId,
      metadata,
      timestamp: new Date().toISOString(),
      created_at: new Date().toISOString()
    };

    try {
      const response = await db.insert(doc);
      console.log(`Added context document: ${response.id}`);
      return { id: response.id, rev: response.rev, ...doc };
    } catch (error) {
      console.error('Error adding context:', error);
      throw error;
    }
  }

  async getContextByType(type, limit = 10) {
    const db = this.client.db.use(this.databases.context);
    
    try {
      const response = await db.view('context', 'by_type', {
        key: type,
        limit,
        include_docs: true,
        descending: true
      });
      
      console.log(`Retrieved ${response.rows.length} context documents of type: ${type}`);
      return response.rows.map(row => row.doc);
    } catch (error) {
      console.error('Error getting context by type:', error);
      throw error;
    }
  }

  async getContextByTopology(topologyId, limit = 50) {
    const db = this.client.db.use(this.databases.context);
    
    try {
      const response = await db.view('context', 'by_topology_id', {
        key: topologyId,
        limit,
        include_docs: true,
        descending: true
      });
      
      console.log(`Retrieved ${response.rows.length} context documents for topology: ${topologyId}`);
      return response.rows.map(row => row.doc);
    } catch (error) {
      console.error('Error getting context by topology:', error);
      throw error;
    }
  }

  // Annotation operations
  async addAnnotation(targetId, targetType, annotation, createdBy, tags = []) {
    const db = this.client.db.use(this.databases.annotations);
    
    const doc = {
      target_id: targetId,
      target_type: targetType,
      annotation,
      created_by: createdBy,
      tags,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    try {
      const response = await db.insert(doc);
      console.log(`Added annotation: ${response.id}`);
      return { id: response.id, rev: response.rev, ...doc };
    } catch (error) {
      console.error('Error adding annotation:', error);
      throw error;
    }
  }

  async getAnnotations(targetId) {
    const db = this.client.db.use(this.databases.annotations);
    
    try {
      const response = await db.view('annotations', 'by_target', {
        key: targetId,
        include_docs: true
      });
      
      console.log(`Retrieved ${response.rows.length} annotations for target: ${targetId}`);
      return response.rows.map(row => row.doc);
    } catch (error) {
      console.error('Error getting annotations:', error);
      throw error;
    }
  }

  // Session operations
  async createSession(userId, sessionType = 'training', metadata = {}) {
    const db = this.client.db.use(this.databases.sessions);
    
    const doc = {
      user_id: userId,
      session_type: sessionType,
      status: 'active',
      metadata,
      started_at: new Date().toISOString(),
      last_activity: new Date().toISOString()
    };

    try {
      const response = await db.insert(doc);
      console.log(`Created session: ${response.id}`);
      return { id: response.id, rev: response.rev, ...doc };
    } catch (error) {
      console.error('Error creating session:', error);
      throw error;
    }
  }

  async updateSession(sessionId, updates) {
    const db = this.client.db.use(this.databases.sessions);
    
    try {
      const doc = await db.get(sessionId);
      const updatedDoc = {
        ...doc,
        ...updates,
        last_activity: new Date().toISOString()
      };
      
      const response = await db.insert(updatedDoc);
      console.log(`Updated session: ${sessionId}`);
      return { id: response.id, rev: response.rev, ...updatedDoc };
    } catch (error) {
      console.error('Error updating session:', error);
      throw error;
    }
  }

  async endSession(sessionId) {
    return this.updateSession(sessionId, {
      status: 'completed',
      ended_at: new Date().toISOString()
    });
  }

  // Experiment tracking
  async createExperiment(name, topologyId, parameters, description = '') {
    const db = this.client.db.use(this.databases.experiments);
    
    const doc = {
      name,
      topology_id: topologyId,
      parameters,
      description,
      status: 'running',
      results: {},
      metrics: [],
      created_at: new Date().toISOString(),
      started_at: new Date().toISOString()
    };

    try {
      const response = await db.insert(doc);
      console.log(`Created experiment: ${response.id}`);
      return { id: response.id, rev: response.rev, ...doc };
    } catch (error) {
      console.error('Error creating experiment:', error);
      throw error;
    }
  }

  async updateExperimentMetrics(experimentId, metrics) {
    const db = this.client.db.use(this.databases.experiments);
    
    try {
      const doc = await db.get(experimentId);
      doc.metrics = doc.metrics || [];
      doc.metrics.push({
        timestamp: new Date().toISOString(),
        ...metrics
      });
      doc.updated_at = new Date().toISOString();
      
      const response = await db.insert(doc);
      console.log(`Updated experiment metrics: ${experimentId}`);
      return { id: response.id, rev: response.rev };
    } catch (error) {
      console.error('Error updating experiment metrics:', error);
      throw error;
    }
  }

  async completeExperiment(experimentId, results) {
    const db = this.client.db.use(this.databases.experiments);
    
    try {
      const doc = await db.get(experimentId);
      doc.status = 'completed';
      doc.results = results;
      doc.completed_at = new Date().toISOString();
      doc.updated_at = new Date().toISOString();
      
      const response = await db.insert(doc);
      console.log(`Completed experiment: ${experimentId}`);
      return { id: response.id, rev: response.rev, ...doc };
    } catch (error) {
      console.error('Error completing experiment:', error);
      throw error;
    }
  }

  // Search and query operations
  async searchContext(query, limit = 20) {
    const db = this.client.db.use(this.databases.context);
    
    try {
      // Simple text search (in production, you'd use CouchDB's search capabilities)
      const response = await db.find({
        selector: {
          $or: [
            { "data": { "$regex": query } },
            { "metadata.description": { "$regex": query } }
          ]
        },
        limit,
        sort: [{ "timestamp": "desc" }]
      });
      
      console.log(`Found ${response.docs.length} context documents matching: ${query}`);
      return response.docs;
    } catch (error) {
      console.error('Error searching context:', error);
      throw error;
    }
  }

  // Bulk operations
  async bulkInsertContext(contexts) {
    const db = this.client.db.use(this.databases.context);
    
    const docs = contexts.map(context => ({
      ...context,
      timestamp: context.timestamp || new Date().toISOString(),
      created_at: context.created_at || new Date().toISOString()
    }));

    try {
      const response = await db.bulk({ docs });
      console.log(`Bulk inserted ${response.length} context documents`);
      return response;
    } catch (error) {
      console.error('Error bulk inserting context:', error);
      throw error;
    }
  }
}

// Example usage
async function demonstrateUsage() {
  const contextStore = new TopologyContext();
  
  try {
    console.log('=== CouchDB Integration Demo ===\n');

    // Initialize databases
    await contextStore.initializeDatabases();

    // Add various types of context
    const note1 = await contextStore.addContext(
      'engineering_note',
      'Initial transformer architecture shows promising results with 95% accuracy on validation set.',
      'topology-123',
      { priority: 'high', category: 'performance' }
    );
    console.log('Added engineering note:', note1.id);

    const log1 = await contextStore.addContext(
      'training_log',
      'Epoch 5/10 completed. Loss: 0.1234, Accuracy: 0.952',
      'topology-123',
      { epoch: 5, loss: 0.1234, accuracy: 0.952 }
    );
    console.log('Added training log:', log1.id);

    const insight1 = await contextStore.addContext(
      'insight',
      'Attention heads 3 and 7 seem to focus on different linguistic patterns. Consider increasing head count.',
      'topology-123',
      { suggested_action: 'architecture_modification' }
    );
    console.log('Added insight:', insight1.id);

    // Add annotations
    const annotation1 = await contextStore.addAnnotation(
      'topology-123',
      'topology',
      'This topology performs well on classification tasks but struggles with sequence generation.',
      'user-456',
      ['performance', 'classification', 'generation']
    );
    console.log('Added annotation:', annotation1.id);

    // Create and manage a session
    const session = await contextStore.createSession(
      'user-456',
      'experiment',
      { experiment_type: 'hyperparameter_tuning' }
    );
    console.log('Created session:', session.id);

    await contextStore.updateSession(session.id, {
      metadata: {
        ...session.metadata,
        current_step: 'data_preprocessing',
        progress: 0.1
      }
    });

    // Create an experiment
    const experiment = await contextStore.createExperiment(
      'Transformer Hyperparameter Tuning',
      'topology-123',
      {
        learning_rate: 0.001,
        batch_size: 32,
        hidden_size: 512,
        num_heads: 8
      },
      'Testing different hyperparameters for optimal performance'
    );
    console.log('Created experiment:', experiment.id);

    // Update experiment with metrics
    await contextStore.updateExperimentMetrics(experiment.id, {
      epoch: 1,
      train_loss: 0.89,
      val_loss: 0.92,
      train_acc: 0.72,
      val_acc: 0.70
    });

    await contextStore.updateExperimentMetrics(experiment.id, {
      epoch: 2,
      train_loss: 0.67,
      val_loss: 0.71,
      train_acc: 0.84,
      val_acc: 0.81
    });

    // Complete the experiment
    await contextStore.completeExperiment(experiment.id, {
      best_epoch: 2,
      best_val_acc: 0.81,
      final_model_path: 'models/transformer-tuned-v2.json',
      conclusions: 'Lower learning rate improves convergence stability'
    });

    // Retrieve context by type
    const engineeringNotes = await contextStore.getContextByType('engineering_note', 5);
    console.log('Engineering notes count:', engineeringNotes.length);

    // Retrieve context by topology
    const topologyContext = await contextStore.getContextByTopology('topology-123', 10);
    console.log('Topology context count:', topologyContext.length);

    // Get annotations for the topology
    const annotations = await contextStore.getAnnotations('topology-123');
    console.log('Annotations count:', annotations.length);

    // Search context
    const searchResults = await contextStore.searchContext('accuracy', 5);
    console.log('Search results count:', searchResults.length);

    // End the session
    await contextStore.endSession(session.id);

    // Bulk insert multiple context items
    const bulkContexts = [
      {
        type: 'metric',
        data: 'GPU utilization: 87%',
        topology_id: 'topology-123',
        metadata: { metric_type: 'resource', value: 0.87 }
      },
      {
        type: 'metric', 
        data: 'Memory usage: 4.2GB',
        topology_id: 'topology-123',
        metadata: { metric_type: 'resource', value: 4.2, unit: 'GB' }
      },
      {
        type: 'checkpoint',
        data: 'Model checkpoint saved at epoch 10',
        topology_id: 'topology-123',
        metadata: { epoch: 10, checkpoint_path: 'checkpoints/model-epoch-10.pt' }
      }
    ];

    const bulkResults = await contextStore.bulkInsertContext(bulkContexts);
    console.log('Bulk insert results:', bulkResults.length);

    console.log('\n=== Demo completed successfully! ===');

  } catch (error) {
    console.error('Demo failed:', error);
  }
}

module.exports = { TopologyContext, demonstrateUsage };

// Run demo if this file is executed directly
if (require.main === module) {
  demonstrateUsage();
}
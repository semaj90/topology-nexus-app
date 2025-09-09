const Minio = require('minio');
const fs = require('fs');
const path = require('path');

/**
 * MinIO Integration Example for Topology Nexus App
 * Demonstrates object storage for models, datasets, and logs
 */

const minioClient = new Minio.Client({
  endPoint: 'localhost',
  port: 9000,
  useSSL: false,
  accessKey: 'minioadmin',
  secretKey: 'minioadmin',
});

class TopologyStorage {
  constructor() {
    this.client = minioClient;
    this.buckets = {
      models: 'topology-models',
      datasets: 'topology-datasets', 
      logs: 'topology-logs',
      artifacts: 'topology-artifacts'
    };
  }

  // Initialize buckets
  async initializeBuckets() {
    console.log('Initializing MinIO buckets...');
    
    for (const [name, bucket] of Object.entries(this.buckets)) {
      try {
        const exists = await this.client.bucketExists(bucket);
        if (!exists) {
          await this.client.makeBucket(bucket, 'us-east-1');
          console.log(`Created bucket: ${bucket}`);
        } else {
          console.log(`Bucket already exists: ${bucket}`);
        }
      } catch (error) {
        console.error(`Error with bucket ${bucket}:`, error);
      }
    }
  }

  // Model operations
  async uploadModel(modelName, modelData, metadata = {}) {
    const objectName = `models/${modelName}.json`;
    const bucket = this.buckets.models;

    try {
      // Add metadata
      const metaData = {
        'Content-Type': 'application/json',
        'X-Amz-Meta-Upload-Date': new Date().toISOString(),
        'X-Amz-Meta-Model-Type': metadata.type || 'qlora',
        'X-Amz-Meta-Architecture': metadata.architecture || 'transformer',
        ...metadata
      };

      const modelBuffer = Buffer.from(JSON.stringify(modelData));
      
      await this.client.putObject(bucket, objectName, modelBuffer, modelBuffer.length, metaData);
      
      console.log(`Uploaded model: ${objectName}`);
      return `${bucket}/${objectName}`;
    } catch (error) {
      console.error('Error uploading model:', error);
      throw error;
    }
  }

  async downloadModel(modelName) {
    const objectName = `models/${modelName}.json`;
    const bucket = this.buckets.models;

    try {
      const dataStream = await this.client.getObject(bucket, objectName);
      
      let data = '';
      return new Promise((resolve, reject) => {
        dataStream.on('data', (chunk) => {
          data += chunk;
        });
        
        dataStream.on('end', () => {
          try {
            const modelData = JSON.parse(data);
            console.log(`Downloaded model: ${objectName}`);
            resolve(modelData);
          } catch (error) {
            reject(error);
          }
        });
        
        dataStream.on('error', reject);
      });
    } catch (error) {
      console.error('Error downloading model:', error);
      throw error;
    }
  }

  // Dataset operations
  async uploadDataset(datasetName, files) {
    const results = [];
    
    for (const file of files) {
      const objectName = `datasets/${datasetName}/${file.name}`;
      const bucket = this.buckets.datasets;

      try {
        const metaData = {
          'Content-Type': file.type || 'application/octet-stream',
          'X-Amz-Meta-Dataset': datasetName,
          'X-Amz-Meta-Upload-Date': new Date().toISOString(),
        };

        let fileStream;
        if (typeof file.data === 'string') {
          fileStream = Buffer.from(file.data);
        } else {
          fileStream = file.data;
        }

        await this.client.putObject(bucket, objectName, fileStream, fileStream.length, metaData);
        
        console.log(`Uploaded dataset file: ${objectName}`);
        results.push(`${bucket}/${objectName}`);
      } catch (error) {
        console.error(`Error uploading dataset file ${file.name}:`, error);
        throw error;
      }
    }
    
    return results;
  }

  async listDatasetFiles(datasetName) {
    const bucket = this.buckets.datasets;
    const prefix = `datasets/${datasetName}/`;

    try {
      const objectsStream = this.client.listObjects(bucket, prefix, true);
      const objects = [];

      return new Promise((resolve, reject) => {
        objectsStream.on('data', (obj) => {
          objects.push({
            name: obj.name,
            size: obj.size,
            lastModified: obj.lastModified,
            etag: obj.etag
          });
        });

        objectsStream.on('end', () => {
          console.log(`Listed ${objects.length} files for dataset: ${datasetName}`);
          resolve(objects);
        });

        objectsStream.on('error', reject);
      });
    } catch (error) {
      console.error('Error listing dataset files:', error);
      throw error;
    }
  }

  // Log operations
  async uploadLog(logName, logData, logLevel = 'info') {
    const objectName = `logs/${new Date().toISOString().split('T')[0]}/${logName}.log`;
    const bucket = this.buckets.logs;

    try {
      const metaData = {
        'Content-Type': 'text/plain',
        'X-Amz-Meta-Log-Level': logLevel,
        'X-Amz-Meta-Upload-Date': new Date().toISOString(),
      };

      const logBuffer = Buffer.from(typeof logData === 'string' ? logData : JSON.stringify(logData));
      
      await this.client.putObject(bucket, objectName, logBuffer, logBuffer.length, metaData);
      
      console.log(`Uploaded log: ${objectName}`);
      return `${bucket}/${objectName}`;
    } catch (error) {
      console.error('Error uploading log:', error);
      throw error;
    }
  }

  // Artifact operations (for training checkpoints, exports, etc.)
  async uploadArtifact(artifactName, data, type = 'checkpoint') {
    const objectName = `artifacts/${type}/${artifactName}`;
    const bucket = this.buckets.artifacts;

    try {
      const metaData = {
        'Content-Type': 'application/octet-stream',
        'X-Amz-Meta-Artifact-Type': type,
        'X-Amz-Meta-Upload-Date': new Date().toISOString(),
      };

      const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data));
      
      await this.client.putObject(bucket, objectName, dataBuffer, dataBuffer.length, metaData);
      
      console.log(`Uploaded artifact: ${objectName}`);
      return `${bucket}/${objectName}`;
    } catch (error) {
      console.error('Error uploading artifact:', error);
      throw error;
    }
  }

  // Get presigned URLs for direct access
  async getPresignedUrl(bucket, objectName, expiry = 24 * 60 * 60) {
    try {
      const url = await this.client.presignedGetObject(bucket, objectName, expiry);
      console.log(`Generated presigned URL for ${bucket}/${objectName}`);
      return url;
    } catch (error) {
      console.error('Error generating presigned URL:', error);
      throw error;
    }
  }

  // Get object metadata
  async getObjectMetadata(bucket, objectName) {
    try {
      const stat = await this.client.statObject(bucket, objectName);
      console.log(`Retrieved metadata for ${bucket}/${objectName}`);
      return stat;
    } catch (error) {
      console.error('Error getting object metadata:', error);
      throw error;
    }
  }

  // Delete object
  async deleteObject(bucket, objectName) {
    try {
      await this.client.removeObject(bucket, objectName);
      console.log(`Deleted object: ${bucket}/${objectName}`);
      return true;
    } catch (error) {
      console.error('Error deleting object:', error);
      throw error;
    }
  }
}

// Example usage
async function demonstrateUsage() {
  const storage = new TopologyStorage();
  
  try {
    console.log('=== MinIO Integration Demo ===\n');

    // Initialize buckets
    await storage.initializeBuckets();

    // Upload a sample model
    const sampleModel = {
      id: 'demo-transformer',
      architecture: 'transformer',
      parameters: {
        layers: 6,
        dimensions: 512,
        attention_heads: 8
      },
      weights: Array(1000).fill(0).map(() => Math.random()),
      metadata: {
        trained_on: 'demo-dataset',
        epochs: 10,
        loss: 0.1234
      }
    };

    const modelPath = await storage.uploadModel('demo-transformer', sampleModel, {
      type: 'qlora',
      architecture: 'transformer'
    });
    console.log('Model uploaded to:', modelPath);

    // Download the model back
    const downloadedModel = await storage.downloadModel('demo-transformer');
    console.log('Downloaded model ID:', downloadedModel.id);

    // Upload dataset files
    const datasetFiles = [
      { name: 'chunk_001.json', data: JSON.stringify({chunks: ['sample text 1', 'sample text 2']}), type: 'application/json' },
      { name: 'chunk_002.json', data: JSON.stringify({chunks: ['sample text 3', 'sample text 4']}), type: 'application/json' },
      { name: 'metadata.json', data: JSON.stringify({total_chunks: 4, source: 'demo'}), type: 'application/json' }
    ];

    const datasetPaths = await storage.uploadDataset('demo-dataset', datasetFiles);
    console.log('Dataset uploaded to:', datasetPaths);

    // List dataset files
    const datasetFileList = await storage.listDatasetFiles('demo-dataset');
    console.log('Dataset files:', datasetFileList.map(f => f.name));

    // Upload a log
    const logData = `[${new Date().toISOString()}] INFO: Demo training completed successfully\n[${new Date().toISOString()}] INFO: Final loss: 0.1234`;
    const logPath = await storage.uploadLog('demo-training', logData, 'info');
    console.log('Log uploaded to:', logPath);

    // Upload an artifact (training checkpoint)
    const checkpoint = {
      epoch: 10,
      loss: 0.1234,
      model_state: 'compressed_state_data_here',
      optimizer_state: 'optimizer_data_here'
    };
    const artifactPath = await storage.uploadArtifact('demo-checkpoint-epoch-10.json', checkpoint, 'checkpoint');
    console.log('Artifact uploaded to:', artifactPath);

    // Get presigned URL for model
    const presignedUrl = await storage.getPresignedUrl(storage.buckets.models, 'models/demo-transformer.json');
    console.log('Presigned URL generated (first 100 chars):', presignedUrl.substring(0, 100) + '...');

    // Get object metadata
    const metadata = await storage.getObjectMetadata(storage.buckets.models, 'models/demo-transformer.json');
    console.log('Object metadata:', {
      size: metadata.size,
      lastModified: metadata.lastModified,
      contentType: metadata.contentType
    });

    console.log('\n=== Demo completed successfully! ===');

  } catch (error) {
    console.error('Demo failed:', error);
  }
}

module.exports = { TopologyStorage, demonstrateUsage };

// Run demo if this file is executed directly
if (require.main === module) {
  demonstrateUsage();
}
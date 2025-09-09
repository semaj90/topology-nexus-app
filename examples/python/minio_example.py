from minio import Minio
from minio.error import S3Error
import json
import io
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, BinaryIO

"""
MinIO Integration Example for Topology Nexus App
Demonstrates object storage for models, datasets, logs, and artifacts using Python
"""

class TopologyStorage:
    def __init__(self, 
                 endpoint='localhost:9000',
                 access_key='minioadmin',
                 secret_key='minioadmin',
                 secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        self.buckets = {
            'models': 'topology-models',
            'datasets': 'topology-datasets',
            'logs': 'topology-logs', 
            'artifacts': 'topology-artifacts',
            'checkpoints': 'topology-checkpoints'
        }
    
    def initialize_buckets(self):
        """Create all required buckets if they don't exist"""
        print("Initializing MinIO buckets...")
        
        for name, bucket in self.buckets.items():
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    print(f"Created bucket: {bucket}")
                else:
                    print(f"Bucket already exists: {bucket}")
            except S3Error as e:
                print(f"Error with bucket {bucket}: {e}")
                raise
    
    # Model operations
    def upload_model(self, 
                    model_name: str, 
                    model_data: Dict, 
                    metadata: Optional[Dict] = None) -> str:
        """Upload a model to MinIO"""
        bucket = self.buckets['models']
        object_name = f"models/{model_name}.json"
        
        try:
            # Prepare metadata
            meta_data = {
                'Content-Type': 'application/json',
                'X-Amz-Meta-Upload-Date': datetime.now().isoformat(),
                'X-Amz-Meta-Model-Type': metadata.get('type', 'qlora') if metadata else 'qlora',
                'X-Amz-Meta-Architecture': metadata.get('architecture', 'transformer') if metadata else 'transformer'
            }
            
            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in ['type', 'architecture']:
                        meta_data[f'X-Amz-Meta-{key.replace("_", "-")}'] = str(value)
            
            # Convert model data to bytes
            model_json = json.dumps(model_data, indent=2)
            model_bytes = model_json.encode('utf-8')
            
            # Upload
            self.client.put_object(
                bucket,
                object_name,
                io.BytesIO(model_bytes),
                len(model_bytes),
                metadata=meta_data
            )
            
            print(f"Uploaded model: {object_name}")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            print(f"Error uploading model: {e}")
            raise
    
    def download_model(self, model_name: str) -> Dict:
        """Download a model from MinIO"""
        bucket = self.buckets['models']
        object_name = f"models/{model_name}.json"
        
        try:
            response = self.client.get_object(bucket, object_name)
            model_data = json.loads(response.data.decode('utf-8'))
            response.close()
            response.release_conn()
            
            print(f"Downloaded model: {object_name}")
            return model_data
            
        except S3Error as e:
            print(f"Error downloading model: {e}")
            raise
    
    def list_models(self, prefix: str = "models/") -> List[Dict]:
        """List all models in storage"""
        bucket = self.buckets['models']
        
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            model_list = []
            
            for obj in objects:
                # Get object metadata
                stat = self.client.stat_object(bucket, obj.object_name)
                
                model_info = {
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag,
                    'metadata': stat.metadata if hasattr(stat, 'metadata') else {}
                }
                model_list.append(model_info)
            
            print(f"Found {len(model_list)} models")
            return model_list
            
        except S3Error as e:
            print(f"Error listing models: {e}")
            raise
    
    # Dataset operations
    def upload_dataset_file(self, 
                           dataset_name: str, 
                           file_name: str, 
                           file_data: Union[str, bytes, BinaryIO],
                           content_type: str = 'application/json') -> str:
        """Upload a single dataset file"""
        bucket = self.buckets['datasets']
        object_name = f"datasets/{dataset_name}/{file_name}"
        
        try:
            metadata = {
                'Content-Type': content_type,
                'X-Amz-Meta-Dataset': dataset_name,
                'X-Amz-Meta-Upload-Date': datetime.now().isoformat(),
            }
            
            # Handle different data types
            if isinstance(file_data, str):
                data = file_data.encode('utf-8')
                data_stream = io.BytesIO(data)
                data_length = len(data)
            elif isinstance(file_data, bytes):
                data_stream = io.BytesIO(file_data)
                data_length = len(file_data)
            else:  # Assume it's a file-like object
                data_stream = file_data
                # Try to get file size
                current_pos = data_stream.tell()
                data_stream.seek(0, 2)  # Seek to end
                data_length = data_stream.tell()
                data_stream.seek(current_pos)  # Reset position
            
            self.client.put_object(
                bucket,
                object_name,
                data_stream,
                data_length,
                metadata=metadata
            )
            
            print(f"Uploaded dataset file: {object_name}")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            print(f"Error uploading dataset file: {e}")
            raise
    
    def upload_dataset(self, 
                      dataset_name: str, 
                      files: List[Dict]) -> List[str]:
        """Upload multiple files for a dataset"""
        results = []
        
        for file_info in files:
            path = self.upload_dataset_file(
                dataset_name,
                file_info['name'],
                file_info['data'],
                file_info.get('content_type', 'application/json')
            )
            results.append(path)
        
        return results
    
    def download_dataset_file(self, 
                             dataset_name: str, 
                             file_name: str) -> bytes:
        """Download a specific dataset file"""
        bucket = self.buckets['datasets']
        object_name = f"datasets/{dataset_name}/{file_name}"
        
        try:
            response = self.client.get_object(bucket, object_name)
            data = response.data
            response.close()
            response.release_conn()
            
            print(f"Downloaded dataset file: {object_name}")
            return data
            
        except S3Error as e:
            print(f"Error downloading dataset file: {e}")
            raise
    
    def list_dataset_files(self, dataset_name: str) -> List[Dict]:
        """List all files in a dataset"""
        bucket = self.buckets['datasets']
        prefix = f"datasets/{dataset_name}/"
        
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            file_list = []
            
            for obj in objects:
                file_info = {
                    'name': obj.object_name.replace(prefix, ''),
                    'full_path': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                }
                file_list.append(file_info)
            
            print(f"Found {len(file_list)} files in dataset: {dataset_name}")
            return file_list
            
        except S3Error as e:
            print(f"Error listing dataset files: {e}")
            raise
    
    # Log operations
    def upload_log(self, 
                  log_name: str, 
                  log_data: str, 
                  log_level: str = 'info') -> str:
        """Upload a log file"""
        bucket = self.buckets['logs']
        date_str = datetime.now().strftime('%Y-%m-%d')
        object_name = f"logs/{date_str}/{log_name}.log"
        
        try:
            metadata = {
                'Content-Type': 'text/plain',
                'X-Amz-Meta-Log-Level': log_level,
                'X-Amz-Meta-Upload-Date': datetime.now().isoformat(),
            }
            
            log_bytes = log_data.encode('utf-8')
            
            self.client.put_object(
                bucket,
                object_name,
                io.BytesIO(log_bytes),
                len(log_bytes),
                metadata=metadata
            )
            
            print(f"Uploaded log: {object_name}")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            print(f"Error uploading log: {e}")
            raise
    
    def download_log(self, log_name: str, date: Optional[str] = None) -> str:
        """Download a log file"""
        bucket = self.buckets['logs']
        date_str = date or datetime.now().strftime('%Y-%m-%d')
        object_name = f"logs/{date_str}/{log_name}.log"
        
        try:
            response = self.client.get_object(bucket, object_name)
            log_data = response.data.decode('utf-8')
            response.close()
            response.release_conn()
            
            print(f"Downloaded log: {object_name}")
            return log_data
            
        except S3Error as e:
            print(f"Error downloading log: {e}")
            raise
    
    # Checkpoint operations
    def upload_checkpoint(self, 
                         checkpoint_name: str, 
                         checkpoint_data: Dict,
                         topology_id: Optional[str] = None) -> str:
        """Upload a training checkpoint"""
        bucket = self.buckets['checkpoints']
        
        if topology_id:
            object_name = f"checkpoints/{topology_id}/{checkpoint_name}.json"
        else:
            object_name = f"checkpoints/{checkpoint_name}.json"
        
        try:
            metadata = {
                'Content-Type': 'application/json',
                'X-Amz-Meta-Checkpoint-Type': 'training',
                'X-Amz-Meta-Upload-Date': datetime.now().isoformat(),
            }
            
            if topology_id:
                metadata['X-Amz-Meta-Topology-Id'] = topology_id
            
            checkpoint_json = json.dumps(checkpoint_data, indent=2)
            checkpoint_bytes = checkpoint_json.encode('utf-8')
            
            self.client.put_object(
                bucket,
                object_name,
                io.BytesIO(checkpoint_bytes),
                len(checkpoint_bytes),
                metadata=metadata
            )
            
            print(f"Uploaded checkpoint: {object_name}")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            print(f"Error uploading checkpoint: {e}")
            raise
    
    def download_checkpoint(self, 
                           checkpoint_name: str, 
                           topology_id: Optional[str] = None) -> Dict:
        """Download a training checkpoint"""
        bucket = self.buckets['checkpoints']
        
        if topology_id:
            object_name = f"checkpoints/{topology_id}/{checkpoint_name}.json"
        else:
            object_name = f"checkpoints/{checkpoint_name}.json"
        
        try:
            response = self.client.get_object(bucket, object_name)
            checkpoint_data = json.loads(response.data.decode('utf-8'))
            response.close()
            response.release_conn()
            
            print(f"Downloaded checkpoint: {object_name}")
            return checkpoint_data
            
        except S3Error as e:
            print(f"Error downloading checkpoint: {e}")
            raise
    
    # Artifact operations
    def upload_artifact(self, 
                       artifact_name: str, 
                       data: Union[Dict, bytes], 
                       artifact_type: str = 'general') -> str:
        """Upload an artifact (generic storage)"""
        bucket = self.buckets['artifacts']
        object_name = f"artifacts/{artifact_type}/{artifact_name}"
        
        try:
            metadata = {
                'X-Amz-Meta-Artifact-Type': artifact_type,
                'X-Amz-Meta-Upload-Date': datetime.now().isoformat(),
            }
            
            # Handle different data types
            if isinstance(data, dict):
                metadata['Content-Type'] = 'application/json'
                data_bytes = json.dumps(data, indent=2).encode('utf-8')
            elif isinstance(data, bytes):
                metadata['Content-Type'] = 'application/octet-stream'
                data_bytes = data
            else:
                metadata['Content-Type'] = 'text/plain'
                data_bytes = str(data).encode('utf-8')
            
            self.client.put_object(
                bucket,
                object_name,
                io.BytesIO(data_bytes),
                len(data_bytes),
                metadata=metadata
            )
            
            print(f"Uploaded artifact: {object_name}")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            print(f"Error uploading artifact: {e}")
            raise
    
    # URL and metadata operations
    def get_presigned_url(self, 
                         bucket: str, 
                         object_name: str, 
                         expiry: int = 86400) -> str:
        """Get a presigned URL for direct access"""
        try:
            url = self.client.presigned_get_object(bucket, object_name, timedelta(seconds=expiry))
            print(f"Generated presigned URL for {bucket}/{object_name}")
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            raise
    
    def get_object_metadata(self, bucket: str, object_name: str) -> Dict:
        """Get metadata for an object"""
        try:
            stat = self.client.stat_object(bucket, object_name)
            
            metadata = {
                'size': stat.size,
                'last_modified': stat.last_modified,
                'etag': stat.etag,
                'content_type': stat.content_type,
                'custom_metadata': stat.metadata if hasattr(stat, 'metadata') else {}
            }
            
            print(f"Retrieved metadata for {bucket}/{object_name}")
            return metadata
            
        except S3Error as e:
            print(f"Error getting object metadata: {e}")
            raise
    
    def delete_object(self, bucket: str, object_name: str) -> bool:
        """Delete an object"""
        try:
            self.client.remove_object(bucket, object_name)
            print(f"Deleted object: {bucket}/{object_name}")
            return True
        except S3Error as e:
            print(f"Error deleting object: {e}")
            raise
    
    # Bulk operations
    def copy_object(self, 
                   src_bucket: str, 
                   src_object: str, 
                   dst_bucket: str, 
                   dst_object: str) -> bool:
        """Copy an object between buckets or within the same bucket"""
        try:
            from minio.commonconfig import CopySource
            
            copy_source = CopySource(src_bucket, src_object)
            self.client.copy_object(dst_bucket, dst_object, copy_source)
            
            print(f"Copied {src_bucket}/{src_object} to {dst_bucket}/{dst_object}")
            return True
            
        except S3Error as e:
            print(f"Error copying object: {e}")
            raise


def demonstrate_usage():
    """Demonstrate MinIO integration"""
    print("=== MinIO Integration Demo ===\n")
    
    try:
        storage = TopologyStorage()
        
        # Initialize buckets
        storage.initialize_buckets()
        
        # Upload a sample model
        sample_model = {
            'id': 'demo-transformer-py',
            'architecture': 'transformer',
            'config': {
                'layers': 6,
                'dimensions': 512,
                'attention_heads': 8,
                'dropout': 0.1
            },
            'parameters': {
                'total_params': 125000000,
                'trainable_params': 5000000
            },
            'weights': [0.1, 0.2, 0.3],  # Simplified for demo
            'training_info': {
                'epochs': 10,
                'final_loss': 0.1234,
                'accuracy': 0.95
            }
        }
        
        model_path = storage.upload_model(
            'demo-transformer-py', 
            sample_model,
            {
                'type': 'qlora',
                'architecture': 'transformer',
                'compression_ratio': 0.1,
                'memory_size_mb': 256
            }
        )
        print(f"Model uploaded to: {model_path}\n")
        
        # Download the model back
        downloaded_model = storage.download_model('demo-transformer-py')
        print(f"Downloaded model ID: {downloaded_model['id']}")
        print(f"Model parameters: {downloaded_model['parameters']}\n")
        
        # Upload dataset files
        dataset_files = [
            {
                'name': 'chunk_001.jsonl',
                'data': json.dumps({'chunks': ['Sample text chunk 1', 'Sample text chunk 2']}) + '\n' +
                       json.dumps({'chunks': ['Sample text chunk 3', 'Sample text chunk 4']}),
                'content_type': 'application/x-ndjson'
            },
            {
                'name': 'chunk_002.jsonl', 
                'data': json.dumps({'chunks': ['Sample text chunk 5', 'Sample text chunk 6']}) + '\n' +
                       json.dumps({'chunks': ['Sample text chunk 7', 'Sample text chunk 8']}),
                'content_type': 'application/x-ndjson'
            },
            {
                'name': 'metadata.json',
                'data': json.dumps({
                    'total_chunks': 8,
                    'source_urls': ['https://example.com/data1', 'https://example.com/data2'],
                    'processing_date': datetime.now().isoformat(),
                    'statistics': {
                        'avg_chunk_length': 150,
                        'total_tokens': 1200
                    }
                }),
                'content_type': 'application/json'
            }
        ]
        
        dataset_paths = storage.upload_dataset('demo-dataset-py', dataset_files)
        print(f"Dataset uploaded to: {dataset_paths}\n")
        
        # List dataset files
        dataset_file_list = storage.list_dataset_files('demo-dataset-py')
        print("Dataset files:")
        for file_info in dataset_file_list:
            print(f"  - {file_info['name']} ({file_info['size']} bytes)")
        print()
        
        # Download a dataset file
        metadata_content = storage.download_dataset_file('demo-dataset-py', 'metadata.json')
        metadata = json.loads(metadata_content.decode('utf-8'))
        print(f"Dataset metadata: {metadata}\n")
        
        # Upload training logs
        training_log = f"""[{datetime.now().isoformat()}] INFO: Starting training for demo-transformer-py
[{datetime.now().isoformat()}] INFO: Epoch 1/5 - Loss: 0.8567, Accuracy: 0.72
[{datetime.now().isoformat()}] INFO: Epoch 2/5 - Loss: 0.5432, Accuracy: 0.84
[{datetime.now().isoformat()}] INFO: Epoch 3/5 - Loss: 0.3456, Accuracy: 0.89
[{datetime.now().isoformat()}] INFO: Epoch 4/5 - Loss: 0.2345, Accuracy: 0.92
[{datetime.now().isoformat()}] INFO: Epoch 5/5 - Loss: 0.1234, Accuracy: 0.95
[{datetime.now().isoformat()}] INFO: Training completed successfully!"""
        
        log_path = storage.upload_log('demo-training-py', training_log, 'info')
        print(f"Training log uploaded to: {log_path}\n")
        
        # Upload a checkpoint
        checkpoint_data = {
            'epoch': 5,
            'loss': 0.1234,
            'accuracy': 0.95,
            'model_state_dict': 'base64_encoded_model_state_here',
            'optimizer_state_dict': 'base64_encoded_optimizer_state_here',
            'scheduler_state_dict': 'base64_encoded_scheduler_state_here',
            'training_config': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'weight_decay': 0.01
            },
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = storage.upload_checkpoint(
            'demo-checkpoint-epoch-5',
            checkpoint_data,
            'topology-123'
        )
        print(f"Checkpoint uploaded to: {checkpoint_path}\n")
        
        # Download the checkpoint
        downloaded_checkpoint = storage.download_checkpoint('demo-checkpoint-epoch-5', 'topology-123')
        print(f"Downloaded checkpoint - Epoch: {downloaded_checkpoint['epoch']}, "
              f"Loss: {downloaded_checkpoint['loss']}\n")
        
        # Upload artifacts
        experiment_results = {
            'experiment_id': 'exp-001',
            'topology_id': 'topology-123',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout': 0.1
            },
            'metrics': {
                'final_accuracy': 0.95,
                'best_f1_score': 0.94,
                'training_time_minutes': 45.7
            },
            'conclusions': [
                'Lower learning rate improved convergence stability',
                'Dropout of 0.1 provided optimal regularization',
                'Model ready for production deployment'
            ]
        }
        
        artifact_path = storage.upload_artifact(
            'experiment-001-results.json',
            experiment_results,
            'experiment-results'
        )
        print(f"Experiment results uploaded to: {artifact_path}\n")
        
        # List all models
        all_models = storage.list_models()
        print("Available models:")
        for model in all_models:
            print(f"  - {model['name']} ({model['size']} bytes, modified: {model['last_modified']})")
        print()
        
        # Get presigned URL for model access
        presigned_url = storage.get_presigned_url(
            storage.buckets['models'], 
            'models/demo-transformer-py.json',
            3600  # 1 hour expiry
        )
        print(f"Presigned URL (first 100 chars): {presigned_url[:100]}...\n")
        
        # Get object metadata
        model_metadata = storage.get_object_metadata(
            storage.buckets['models'],
            'models/demo-transformer-py.json'
        )
        print("Model metadata:")
        for key, value in model_metadata.items():
            if key != 'custom_metadata':
                print(f"  {key}: {value}")
        print(f"  custom_metadata: {model_metadata['custom_metadata']}\n")
        
        print("=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_usage()
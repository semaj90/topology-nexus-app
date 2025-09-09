import psycopg2
import psycopg2.extras
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

"""
PostgreSQL Integration Example for Topology Nexus App
Demonstrates CRUD operations for topologies, modules, and datasets using Python
"""

class TopologyDB:
    def __init__(self, 
                 host='localhost', 
                 port=5432, 
                 database='topology', 
                 user='nexus', 
                 password='changeme'):
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database, 
            'user': user,
            'password': password
        }
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            print("Connected to PostgreSQL database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Disconnected from database")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    # User operations
    def create_user(self, username: str, email: str) -> Dict:
        """Create a new user"""
        query = """
            INSERT INTO users (username, email) 
            VALUES (%s, %s) 
            RETURNING id, username, email, created_at
        """
        try:
            self.cursor.execute(query, (username, email))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Created user: {result['username']}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating user: {e}")
            raise
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        query = "SELECT * FROM users WHERE id = %s"
        try:
            self.cursor.execute(query, (user_id,))
            result = self.cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            print(f"Error fetching user: {e}")
            raise
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = %s"
        try:
            self.cursor.execute(query, (username,))
            result = self.cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            print(f"Error fetching user by username: {e}")
            raise
    
    # Topology operations
    def create_topology(self, 
                       name: str, 
                       description: str, 
                       architecture: str, 
                       created_by: int, 
                       config: Dict = None) -> Dict:
        """Create a new topology"""
        config = config or {}
        query = """
            INSERT INTO topologies (name, description, architecture, created_by, config) 
            VALUES (%s, %s, %s, %s, %s) 
            RETURNING id, name, architecture, created_at
        """
        try:
            self.cursor.execute(query, (name, description, architecture, created_by, json.dumps(config)))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Created topology: {result['name']}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating topology: {e}")
            raise
    
    def get_topologies(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all topologies, optionally filtered by user"""
        if user_id:
            query = """
                SELECT t.*, u.username 
                FROM topologies t 
                LEFT JOIN users u ON t.created_by = u.id 
                WHERE t.created_by = %s 
                ORDER BY t.created_at DESC
            """
            params = (user_id,)
        else:
            query = """
                SELECT t.*, u.username 
                FROM topologies t 
                LEFT JOIN users u ON t.created_by = u.id 
                ORDER BY t.created_at DESC
            """
            params = ()
        
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching topologies: {e}")
            raise
    
    def get_topology(self, topology_id: int) -> Optional[Dict]:
        """Get topology by ID"""
        query = """
            SELECT t.*, u.username 
            FROM topologies t 
            LEFT JOIN users u ON t.created_by = u.id 
            WHERE t.id = %s
        """
        try:
            self.cursor.execute(query, (topology_id,))
            result = self.cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            print(f"Error fetching topology: {e}")
            raise
    
    def update_topology(self, topology_id: int, updates: Dict) -> bool:
        """Update topology"""
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['name', 'description', 'architecture']:
                set_clauses.append(f"{key} = %s")
                params.append(value)
            elif key == 'config':
                set_clauses.append("config = %s")
                params.append(json.dumps(value))
        
        if not set_clauses:
            return False
        
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(topology_id)
        
        query = f"UPDATE topologies SET {', '.join(set_clauses)} WHERE id = %s"
        
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            print(f"Updated topology {topology_id}")
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating topology: {e}")
            raise
    
    # QLoRA Module operations
    def create_qlora_module(self, 
                           name: str, 
                           topology_id: int, 
                           config_json: Dict, 
                           file_path: str, 
                           compression_ratio: float = 0.1, 
                           memory_size_mb: Optional[int] = None) -> Dict:
        """Create a new QLoRA module"""
        query = """
            INSERT INTO qlora_modules 
            (name, topology_id, config_json, file_path, compression_ratio, memory_size_mb)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """
        try:
            self.cursor.execute(query, (
                name, topology_id, json.dumps(config_json), 
                file_path, compression_ratio, memory_size_mb
            ))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Created QLoRA module: {result['name']}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating QLoRA module: {e}")
            raise
    
    def get_qlora_modules(self, topology_id: int) -> List[Dict]:
        """Get all QLoRA modules for a topology"""
        query = "SELECT * FROM qlora_modules WHERE topology_id = %s ORDER BY created_at DESC"
        try:
            self.cursor.execute(query, (topology_id,))
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching QLoRA modules: {e}")
            raise
    
    def get_qlora_module(self, module_id: int) -> Optional[Dict]:
        """Get QLoRA module by ID"""
        query = "SELECT * FROM qlora_modules WHERE id = %s"
        try:
            self.cursor.execute(query, (module_id,))
            result = self.cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            print(f"Error fetching QLoRA module: {e}")
            raise
    
    # Dataset operations
    def create_dataset(self, 
                      name: str, 
                      description: str, 
                      source_urls: List[str], 
                      created_by: int, 
                      statistics: Dict = None) -> Dict:
        """Create a new dataset"""
        statistics = statistics or {}
        query = """
            INSERT INTO datasets (name, description, source_urls, created_by, statistics)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """
        try:
            self.cursor.execute(query, (
                name, description, source_urls, created_by, json.dumps(statistics)
            ))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Created dataset: {result['name']}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating dataset: {e}")
            raise
    
    def get_datasets(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all datasets, optionally filtered by user"""
        if user_id:
            query = """
                SELECT d.*, u.username 
                FROM datasets d 
                LEFT JOIN users u ON d.created_by = u.id 
                WHERE d.created_by = %s 
                ORDER BY d.created_at DESC
            """
            params = (user_id,)
        else:
            query = """
                SELECT d.*, u.username 
                FROM datasets d 
                LEFT JOIN users u ON d.created_by = u.id 
                ORDER BY d.created_at DESC
            """
            params = ()
        
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching datasets: {e}")
            raise
    
    # Training results operations
    def save_training_results(self, 
                             topology_id: int, 
                             result_json: Dict, 
                             loss_values: List[float], 
                             accuracy: float, 
                             epochs: int) -> Dict:
        """Save training results"""
        query = """
            INSERT INTO training_results 
            (topology_id, result_json, loss_values, accuracy, epochs)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """
        try:
            self.cursor.execute(query, (
                topology_id, json.dumps(result_json), loss_values, accuracy, epochs
            ))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Saved training results for topology {topology_id}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving training results: {e}")
            raise
    
    def get_training_history(self, topology_id: int) -> List[Dict]:
        """Get training history for a topology"""
        query = """
            SELECT * FROM training_results 
            WHERE topology_id = %s 
            ORDER BY completed_at DESC
        """
        try:
            self.cursor.execute(query, (topology_id,))
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching training history: {e}")
            raise
    
    def get_best_training_result(self, topology_id: int) -> Optional[Dict]:
        """Get best training result (highest accuracy) for a topology"""
        query = """
            SELECT * FROM training_results 
            WHERE topology_id = %s 
            ORDER BY accuracy DESC, completed_at DESC 
            LIMIT 1
        """
        try:
            self.cursor.execute(query, (topology_id,))
            result = self.cursor.fetchone()
            if result:
                return dict(result)
            return None
        except Exception as e:
            print(f"Error fetching best training result: {e}")
            raise
    
    # Semantic chunks operations (for processed text data)
    def add_semantic_chunk(self, 
                          dataset_id: int, 
                          content: str, 
                          embedding: Optional[List[float]] = None, 
                          metadata: Dict = None, 
                          chunk_index: Optional[int] = None, 
                          source_url: Optional[str] = None) -> Dict:
        """Add a semantic chunk to a dataset"""
        metadata = metadata or {}
        query = """
            INSERT INTO semantic_chunks 
            (dataset_id, content, embedding, metadata, chunk_index, source_url)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """
        try:
            self.cursor.execute(query, (
                dataset_id, content, embedding, json.dumps(metadata), chunk_index, source_url
            ))
            self.conn.commit()
            result = self.cursor.fetchone()
            print(f"Added semantic chunk to dataset {dataset_id}")
            return dict(result)
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding semantic chunk: {e}")
            raise
    
    def get_semantic_chunks(self, dataset_id: int, limit: int = 100) -> List[Dict]:
        """Get semantic chunks for a dataset"""
        query = """
            SELECT * FROM semantic_chunks 
            WHERE dataset_id = %s 
            ORDER BY chunk_index, created_at 
            LIMIT %s
        """
        try:
            self.cursor.execute(query, (dataset_id, limit))
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching semantic chunks: {e}")
            raise
    
    # Analytics and reporting
    def get_topology_analytics(self, topology_id: int) -> Dict:
        """Get analytics for a topology"""
        try:
            analytics = {}
            
            # Get module count
            self.cursor.execute(
                "SELECT COUNT(*) as module_count FROM qlora_modules WHERE topology_id = %s",
                (topology_id,)
            )
            analytics['module_count'] = self.cursor.fetchone()['module_count']
            
            # Get training runs count
            self.cursor.execute(
                "SELECT COUNT(*) as training_runs FROM training_results WHERE topology_id = %s",
                (topology_id,)
            )
            analytics['training_runs'] = self.cursor.fetchone()['training_runs']
            
            # Get best accuracy
            self.cursor.execute(
                "SELECT MAX(accuracy) as best_accuracy FROM training_results WHERE topology_id = %s",
                (topology_id,)
            )
            result = self.cursor.fetchone()
            analytics['best_accuracy'] = result['best_accuracy'] if result['best_accuracy'] else 0.0
            
            # Get average loss from latest training
            self.cursor.execute(
                """SELECT loss_values FROM training_results 
                   WHERE topology_id = %s 
                   ORDER BY completed_at DESC 
                   LIMIT 1""",
                (topology_id,)
            )
            result = self.cursor.fetchone()
            if result and result['loss_values']:
                analytics['latest_final_loss'] = result['loss_values'][-1]
            else:
                analytics['latest_final_loss'] = None
            
            return analytics
        except Exception as e:
            print(f"Error getting topology analytics: {e}")
            raise


def demonstrate_usage():
    """Demonstrate PostgreSQL integration"""
    print("=== PostgreSQL Integration Demo ===\n")
    
    try:
        with TopologyDB() as db:
            # Create a user
            user = db.create_user('demo_user_py', 'demo_py@example.com')
            print(f"Created user: {user}\n")
            
            # Create a topology
            topology = db.create_topology(
                'demo_transformer_py',
                'Demo transformer created from Python',
                'transformer',
                user['id'],
                {
                    'dimensions': 512,
                    'layers': 6,
                    'attention_heads': 8,
                    'dropout': 0.1,
                    'activation': 'relu'
                }
            )
            print(f"Created topology: {topology}\n")
            
            # Create a QLoRA module
            module = db.create_qlora_module(
                'demo_module_py',
                topology['id'],
                {
                    'type': 'classification',
                    'num_classes': 10,
                    'input_features': 512
                },
                'modules/demo_module_py.json',
                0.15,
                128
            )
            print(f"Created QLoRA module: {module}\n")
            
            # Create a dataset
            dataset = db.create_dataset(
                'demo_dataset_py',
                'Demo dataset created from Python',
                ['https://example.com/data1.json', 'https://example.com/data2.json'],
                user['id'],
                {
                    'total_chunks': 250,
                    'total_size_mb': 15.7,
                    'source_type': 'web_scraping'
                }
            )
            print(f"Created dataset: {dataset}\n")
            
            # Add some semantic chunks
            chunks_data = [
                {
                    'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
                    'metadata': {'category': 'definition', 'importance': 'high'},
                    'chunk_index': 0
                },
                {
                    'content': 'Deep learning uses neural networks with multiple layers to model complex patterns.',
                    'metadata': {'category': 'technique', 'importance': 'high'},
                    'chunk_index': 1
                },
                {
                    'content': 'Transformers have revolutionized natural language processing tasks.',
                    'metadata': {'category': 'architecture', 'importance': 'medium'},
                    'chunk_index': 2
                }
            ]
            
            for chunk_data in chunks_data:
                chunk = db.add_semantic_chunk(
                    dataset['id'],
                    chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    chunk_index=chunk_data['chunk_index'],
                    source_url='https://example.com/ml-basics'
                )
                print(f"Added semantic chunk {chunk_data['chunk_index']}")
            
            # Save training results
            training_result = db.save_training_results(
                topology['id'],
                {
                    'model_path': 'models/demo_transformer_py_final.pt',
                    'optimizer': 'AdamW',
                    'learning_rate': 0.001,
                    'batch_size': 32
                },
                [0.8, 0.5, 0.3, 0.2, 0.15],
                0.94,
                5
            )
            print(f"Saved training results: {training_result}\n")
            
            # Retrieve data
            print("=== Retrieving Data ===")
            
            # Get all topologies
            all_topologies = db.get_topologies()
            print(f"Total topologies: {len(all_topologies)}")
            
            # Get user's topologies
            user_topologies = db.get_topologies(user['id'])
            print(f"User's topologies: {len(user_topologies)}")
            
            # Get QLoRA modules for the topology
            modules = db.get_qlora_modules(topology['id'])
            print(f"Modules for topology: {len(modules)}")
            
            # Get semantic chunks
            chunks = db.get_semantic_chunks(dataset['id'])
            print(f"Semantic chunks in dataset: {len(chunks)}")
            
            # Get training history
            training_history = db.get_training_history(topology['id'])
            print(f"Training runs: {len(training_history)}")
            
            # Get best training result
            best_result = db.get_best_training_result(topology['id'])
            if best_result:
                print(f"Best accuracy: {best_result['accuracy']}")
            
            # Get analytics
            analytics = db.get_topology_analytics(topology['id'])
            print(f"Topology analytics: {analytics}")
            
            print("\n=== Demo completed successfully! ===")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_usage()
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import urllib.parse

"""
CouchDB Integration Example for Topology Nexus App
Demonstrates document storage for context, annotations, and metadata using Python
"""

class TopologyContext:
    def __init__(self, 
                 base_url='http://localhost:5984',
                 username='admin',
                 password='changeme'):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        
        self.databases = {
            'context': 'topology-context',
            'annotations': 'topology-annotations',
            'sessions': 'topology-sessions',
            'experiments': 'topology-experiments'
        }
    
    def initialize_databases(self):
        """Create all required databases and design documents"""
        print("Initializing CouchDB databases...")
        
        for name, db_name in self.databases.items():
            try:
                # Check if database exists
                response = self.session.head(f"{self.base_url}/{db_name}")
                
                if response.status_code == 404:
                    # Create database
                    create_response = self.session.put(f"{self.base_url}/{db_name}")
                    if create_response.status_code in [201, 202]:
                        print(f"Created database: {db_name}")
                        self._create_design_documents(db_name)
                    else:
                        print(f"Failed to create database {db_name}: {create_response.text}")
                elif response.status_code == 200:
                    print(f"Database already exists: {db_name}")
                else:
                    print(f"Error checking database {db_name}: {response.status_code}")
                    
            except Exception as e:
                print(f"Error with database {db_name}: {e}")
    
    def _create_design_documents(self, db_name: str):
        """Create design documents with views and indexes"""
        design_docs = {
            'topology-context': {
                '_id': '_design/context',
                'views': {
                    'by_type': {
                        'map': 'function(doc) { if(doc.type) emit(doc.type, doc); }'
                    },
                    'by_topology_id': {
                        'map': 'function(doc) { if(doc.topology_id) emit(doc.topology_id, doc); }'
                    },
                    'by_timestamp': {
                        'map': 'function(doc) { if(doc.timestamp) emit(doc.timestamp, doc); }'
                    },
                    'recent_by_type': {
                        'map': 'function(doc) { if(doc.type && doc.timestamp) emit([doc.type, doc.timestamp], doc); }'
                    }
                }
            },
            'topology-annotations': {
                '_id': '_design/annotations',
                'views': {
                    'by_target': {
                        'map': 'function(doc) { if(doc.target_id) emit(doc.target_id, doc); }'
                    },
                    'by_user': {
                        'map': 'function(doc) { if(doc.created_by) emit(doc.created_by, doc); }'
                    },
                    'by_tags': {
                        'map': 'function(doc) { if(doc.tags && doc.tags.length > 0) { for(var i in doc.tags) emit(doc.tags[i], doc); } }'
                    }
                }
            },
            'topology-sessions': {
                '_id': '_design/sessions',
                'views': {
                    'by_user': {
                        'map': 'function(doc) { if(doc.user_id) emit(doc.user_id, doc); }'
                    },
                    'active_sessions': {
                        'map': 'function(doc) { if(doc.status === "active") emit(doc.started_at, doc); }'
                    },
                    'by_status': {
                        'map': 'function(doc) { if(doc.status) emit(doc.status, doc); }'
                    }
                }
            },
            'topology-experiments': {
                '_id': '_design/experiments',
                'views': {
                    'by_status': {
                        'map': 'function(doc) { if(doc.status) emit(doc.status, doc); }'
                    },
                    'by_topology': {
                        'map': 'function(doc) { if(doc.topology_id) emit(doc.topology_id, doc); }'
                    },
                    'by_date': {
                        'map': 'function(doc) { if(doc.created_at) emit(doc.created_at, doc); }'
                    }
                }
            }
        }
        
        if db_name in design_docs:
            try:
                response = self.session.put(
                    f"{self.base_url}/{db_name}/_design/{'context' if 'context' in db_name else 'annotations' if 'annotations' in db_name else 'sessions' if 'sessions' in db_name else 'experiments'}",
                    json=design_docs[db_name]
                )
                if response.status_code in [201, 202]:
                    print(f"Created design document for {db_name}")
                elif response.status_code == 409:
                    pass  # Document already exists
                else:
                    print(f"Failed to create design document for {db_name}: {response.text}")
            except Exception as e:
                print(f"Error creating design document for {db_name}: {e}")
    
    # Context operations
    def add_context(self, 
                   context_type: str, 
                   data: Any, 
                   topology_id: Optional[str] = None, 
                   metadata: Optional[Dict] = None) -> Dict:
        """Add a context document"""
        db_name = self.databases['context']
        
        doc = {
            'type': context_type,
            'data': data,
            'topology_id': topology_id,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        try:
            response = self.session.post(f"{self.base_url}/{db_name}", json=doc)
            if response.status_code in [201, 202]:
                result = response.json()
                print(f"Added context document: {result['id']}")
                return {'id': result['id'], 'rev': result['rev'], **doc}
            else:
                print(f"Error adding context: {response.text}")
                raise Exception(f"Failed to add context: {response.text}")
        except Exception as e:
            print(f"Error adding context: {e}")
            raise
    
    def get_context_by_type(self, 
                           context_type: str, 
                           limit: int = 10,
                           descending: bool = True) -> List[Dict]:
        """Get context documents by type"""
        db_name = self.databases['context']
        
        params = {
            'key': f'"{context_type}"',
            'limit': limit,
            'include_docs': 'true',
            'descending': str(descending).lower()
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/context/_view/by_type",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                print(f"Retrieved {len(documents)} context documents of type: {context_type}")
                return documents
            else:
                print(f"Error getting context by type: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting context by type: {e}")
            raise
    
    def get_context_by_topology(self, 
                               topology_id: str, 
                               limit: int = 50) -> List[Dict]:
        """Get context documents for a specific topology"""
        db_name = self.databases['context']
        
        params = {
            'key': f'"{topology_id}"',
            'limit': limit,
            'include_docs': 'true',
            'descending': 'true'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/context/_view/by_topology_id",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                print(f"Retrieved {len(documents)} context documents for topology: {topology_id}")
                return documents
            else:
                print(f"Error getting context by topology: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting context by topology: {e}")
            raise
    
    def search_context(self, 
                      query: str, 
                      limit: int = 20) -> List[Dict]:
        """Search context documents (simple implementation)"""
        db_name = self.databases['context']
        
        # Use Mango query for text search
        search_query = {
            "selector": {
                "$or": [
                    {"data": {"$regex": f"(?i).*{query}.*"}},
                    {"metadata.description": {"$regex": f"(?i).*{query}.*"}}
                ]
            },
            "limit": limit,
            "sort": [{"timestamp": "desc"}]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/{db_name}/_find",
                json=search_query
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get('docs', [])
                print(f"Found {len(documents)} context documents matching: {query}")
                return documents
            else:
                print(f"Error searching context: {response.text}")
                return []
        except Exception as e:
            print(f"Error searching context: {e}")
            raise
    
    # Annotation operations
    def add_annotation(self, 
                      target_id: str, 
                      target_type: str, 
                      annotation: str, 
                      created_by: str, 
                      tags: Optional[List[str]] = None) -> Dict:
        """Add an annotation"""
        db_name = self.databases['annotations']
        
        doc = {
            'target_id': target_id,
            'target_type': target_type,
            'annotation': annotation,
            'created_by': created_by,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        try:
            response = self.session.post(f"{self.base_url}/{db_name}", json=doc)
            if response.status_code in [201, 202]:
                result = response.json()
                print(f"Added annotation: {result['id']}")
                return {'id': result['id'], 'rev': result['rev'], **doc}
            else:
                print(f"Error adding annotation: {response.text}")
                raise Exception(f"Failed to add annotation: {response.text}")
        except Exception as e:
            print(f"Error adding annotation: {e}")
            raise
    
    def get_annotations(self, target_id: str) -> List[Dict]:
        """Get annotations for a target"""
        db_name = self.databases['annotations']
        
        params = {
            'key': f'"{target_id}"',
            'include_docs': 'true'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/annotations/_view/by_target",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                print(f"Retrieved {len(documents)} annotations for target: {target_id}")
                return documents
            else:
                print(f"Error getting annotations: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting annotations: {e}")
            raise
    
    def get_annotations_by_tag(self, tag: str) -> List[Dict]:
        """Get annotations by tag"""
        db_name = self.databases['annotations']
        
        params = {
            'key': f'"{tag}"',
            'include_docs': 'true'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/annotations/_view/by_tags",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                print(f"Retrieved {len(documents)} annotations with tag: {tag}")
                return documents
            else:
                print(f"Error getting annotations by tag: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting annotations by tag: {e}")
            raise
    
    # Session operations
    def create_session(self, 
                      user_id: str, 
                      session_type: str = 'training', 
                      metadata: Optional[Dict] = None) -> Dict:
        """Create a new session"""
        db_name = self.databases['sessions']
        
        doc = {
            'user_id': user_id,
            'session_type': session_type,
            'status': 'active',
            'metadata': metadata or {},
            'started_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        
        try:
            response = self.session.post(f"{self.base_url}/{db_name}", json=doc)
            if response.status_code in [201, 202]:
                result = response.json()
                print(f"Created session: {result['id']}")
                return {'id': result['id'], 'rev': result['rev'], **doc}
            else:
                print(f"Error creating session: {response.text}")
                raise Exception(f"Failed to create session: {response.text}")
        except Exception as e:
            print(f"Error creating session: {e}")
            raise
    
    def update_session(self, session_id: str, updates: Dict) -> Dict:
        """Update a session"""
        db_name = self.databases['sessions']
        
        try:
            # Get current document
            get_response = self.session.get(f"{self.base_url}/{db_name}/{session_id}")
            if get_response.status_code != 200:
                raise Exception(f"Session not found: {session_id}")
            
            current_doc = get_response.json()
            
            # Update document
            updated_doc = {**current_doc, **updates}
            updated_doc['last_activity'] = datetime.now().isoformat()
            
            # Save updated document
            put_response = self.session.put(
                f"{self.base_url}/{db_name}/{session_id}",
                json=updated_doc
            )
            
            if put_response.status_code in [201, 202]:
                result = put_response.json()
                print(f"Updated session: {session_id}")
                return {'id': result['id'], 'rev': result['rev'], **updated_doc}
            else:
                print(f"Error updating session: {put_response.text}")
                raise Exception(f"Failed to update session: {put_response.text}")
        except Exception as e:
            print(f"Error updating session: {e}")
            raise
    
    def end_session(self, session_id: str) -> Dict:
        """End a session"""
        return self.update_session(session_id, {
            'status': 'completed',
            'ended_at': datetime.now().isoformat()
        })
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get active sessions"""
        db_name = self.databases['sessions']
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/sessions/_view/active_sessions",
                params={'include_docs': 'true'}
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                
                # Filter by user if specified
                if user_id:
                    documents = [doc for doc in documents if doc.get('user_id') == user_id]
                
                print(f"Retrieved {len(documents)} active sessions")
                return documents
            else:
                print(f"Error getting active sessions: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting active sessions: {e}")
            raise
    
    # Experiment operations
    def create_experiment(self, 
                         name: str, 
                         topology_id: str, 
                         parameters: Dict, 
                         description: str = '') -> Dict:
        """Create a new experiment"""
        db_name = self.databases['experiments']
        
        doc = {
            'name': name,
            'topology_id': topology_id,
            'parameters': parameters,
            'description': description,
            'status': 'running',
            'results': {},
            'metrics': [],
            'created_at': datetime.now().isoformat(),
            'started_at': datetime.now().isoformat()
        }
        
        try:
            response = self.session.post(f"{self.base_url}/{db_name}", json=doc)
            if response.status_code in [201, 202]:
                result = response.json()
                print(f"Created experiment: {result['id']}")
                return {'id': result['id'], 'rev': result['rev'], **doc}
            else:
                print(f"Error creating experiment: {response.text}")
                raise Exception(f"Failed to create experiment: {response.text}")
        except Exception as e:
            print(f"Error creating experiment: {e}")
            raise
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict) -> Dict:
        """Update experiment metrics"""
        db_name = self.databases['experiments']
        
        try:
            # Get current document
            get_response = self.session.get(f"{self.base_url}/{db_name}/{experiment_id}")
            if get_response.status_code != 200:
                raise Exception(f"Experiment not found: {experiment_id}")
            
            current_doc = get_response.json()
            
            # Add new metrics
            if 'metrics' not in current_doc:
                current_doc['metrics'] = []
            
            current_doc['metrics'].append({
                'timestamp': datetime.now().isoformat(),
                **metrics
            })
            current_doc['updated_at'] = datetime.now().isoformat()
            
            # Save updated document
            put_response = self.session.put(
                f"{self.base_url}/{db_name}/{experiment_id}",
                json=current_doc
            )
            
            if put_response.status_code in [201, 202]:
                result = put_response.json()
                print(f"Updated experiment metrics: {experiment_id}")
                return {'id': result['id'], 'rev': result['rev']}
            else:
                print(f"Error updating experiment metrics: {put_response.text}")
                raise Exception(f"Failed to update experiment metrics: {put_response.text}")
        except Exception as e:
            print(f"Error updating experiment metrics: {e}")
            raise
    
    def complete_experiment(self, experiment_id: str, results: Dict) -> Dict:
        """Complete an experiment"""
        db_name = self.databases['experiments']
        
        try:
            # Get current document
            get_response = self.session.get(f"{self.base_url}/{db_name}/{experiment_id}")
            if get_response.status_code != 200:
                raise Exception(f"Experiment not found: {experiment_id}")
            
            current_doc = get_response.json()
            
            # Update document
            current_doc.update({
                'status': 'completed',
                'results': results,
                'completed_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            
            # Save updated document
            put_response = self.session.put(
                f"{self.base_url}/{db_name}/{experiment_id}",
                json=current_doc
            )
            
            if put_response.status_code in [201, 202]:
                result = put_response.json()
                print(f"Completed experiment: {experiment_id}")
                return {'id': result['id'], 'rev': result['rev'], **current_doc}
            else:
                print(f"Error completing experiment: {put_response.text}")
                raise Exception(f"Failed to complete experiment: {put_response.text}")
        except Exception as e:
            print(f"Error completing experiment: {e}")
            raise
    
    def get_experiments_by_topology(self, topology_id: str) -> List[Dict]:
        """Get experiments for a topology"""
        db_name = self.databases['experiments']
        
        params = {
            'key': f'"{topology_id}"',
            'include_docs': 'true'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/{db_name}/_design/experiments/_view/by_topology",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = [row['doc'] for row in data['rows']]
                print(f"Retrieved {len(documents)} experiments for topology: {topology_id}")
                return documents
            else:
                print(f"Error getting experiments by topology: {response.text}")
                return []
        except Exception as e:
            print(f"Error getting experiments by topology: {e}")
            raise
    
    # Bulk operations
    def bulk_insert_context(self, contexts: List[Dict]) -> List[Dict]:
        """Bulk insert context documents"""
        db_name = self.databases['context']
        
        docs = []
        for context in contexts:
            doc = {
                **context,
                'timestamp': context.get('timestamp', datetime.now().isoformat()),
                'created_at': context.get('created_at', datetime.now().isoformat())
            }
            docs.append(doc)
        
        try:
            response = self.session.post(
                f"{self.base_url}/{db_name}/_bulk_docs",
                json={'docs': docs}
            )
            
            if response.status_code in [201, 202]:
                results = response.json()
                print(f"Bulk inserted {len(results)} context documents")
                return results
            else:
                print(f"Error bulk inserting context: {response.text}")
                raise Exception(f"Failed to bulk insert context: {response.text}")
        except Exception as e:
            print(f"Error bulk inserting context: {e}")
            raise
    
    # Utility methods
    def get_database_info(self, db_name: str) -> Dict:
        """Get information about a database"""
        try:
            response = self.session.get(f"{self.base_url}/{db_name}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting database info: {response.text}")
                return {}
        except Exception as e:
            print(f"Error getting database info: {e}")
            raise


def demonstrate_usage():
    """Demonstrate CouchDB integration"""
    print("=== CouchDB Integration Demo ===\n")
    
    try:
        context_store = TopologyContext()
        
        # Initialize databases
        context_store.initialize_databases()
        
        # Add various types of context
        note1 = context_store.add_context(
            'engineering_note',
            'Initial transformer architecture shows promising results with 95% accuracy on validation set. Consider increasing model depth for better performance.',
            'topology-123',
            {
                'priority': 'high',
                'category': 'performance',
                'author': 'ML Engineer',
                'validation_accuracy': 0.95
            }
        )
        print(f"Added engineering note: {note1['id']}\n")
        
        log1 = context_store.add_context(
            'training_log',
            'Epoch 5/10 completed. Loss decreased from 0.234 to 0.123. Memory usage stable at 4.2GB.',
            'topology-123',
            {
                'epoch': 5,
                'loss': 0.123,
                'accuracy': 0.952,
                'memory_usage_gb': 4.2,
                'gpu_utilization': 0.87
            }
        )
        print(f"Added training log: {log1['id']}\n")
        
        insight1 = context_store.add_context(
            'insight',
            'Attention heads 3 and 7 focus on different linguistic patterns. Head 3 captures syntactic relationships while head 7 handles semantic connections.',
            'topology-123',
            {
                'suggested_action': 'architecture_modification',
                'analysis_type': 'attention_visualization',
                'confidence': 0.89
            }
        )
        print(f"Added insight: {insight1['id']}\n")
        
        # Add system metrics
        metrics1 = context_store.add_context(
            'system_metrics',
            {'gpu_temp': 78, 'cpu_usage': 65, 'memory_free': 12.5, 'disk_usage': 89},
            'topology-123',
            {
                'monitoring_system': 'nvidia-smi',
                'alert_level': 'normal'
            }
        )
        print(f"Added system metrics: {metrics1['id']}\n")
        
        # Add annotations
        annotation1 = context_store.add_annotation(
            'topology-123',
            'topology',
            'This topology performs exceptionally well on classification tasks but shows degraded performance on sequence generation. Consider adding sequence-specific layers.',
            'user-456',
            ['performance', 'classification', 'generation', 'architecture']
        )
        print(f"Added annotation: {annotation1['id']}\n")
        
        annotation2 = context_store.add_annotation(
            'topology-123',
            'topology',
            'Memory efficiency is excellent thanks to QLoRA implementation. Can handle large datasets without OOM errors.',
            'user-456',
            ['memory', 'efficiency', 'qlora', 'scalability']
        )
        print(f"Added annotation: {annotation2['id']}\n")
        
        # Create and manage a session
        session = context_store.create_session(
            'user-456',
            'hyperparameter_experiment',
            {
                'experiment_type': 'grid_search',
                'parameters_to_test': ['learning_rate', 'batch_size', 'dropout'],
                'total_combinations': 24
            }
        )
        print(f"Created session: {session['id']}\n")
        
        # Update session progress
        context_store.update_session(session['id'], {
            'metadata': {
                **session['metadata'],
                'current_combination': 5,
                'progress_percentage': 20.8,
                'best_score_so_far': 0.94
            }
        })
        print("Updated session with progress information\n")
        
        # Create an experiment
        experiment = context_store.create_experiment(
            'Transformer Hyperparameter Optimization',
            'topology-123',
            {
                'learning_rate': [0.001, 0.0001, 0.00001],
                'batch_size': [16, 32, 64],
                'dropout': [0.1, 0.2, 0.3],
                'hidden_size': 512,
                'num_heads': 8,
                'num_layers': 6
            },
            'Comprehensive grid search to find optimal hyperparameters for transformer model performance on classification tasks.'
        )
        print(f"Created experiment: {experiment['id']}\n")
        
        # Update experiment with metrics over time
        metrics_updates = [
            {
                'combination': 1,
                'params': {'lr': 0.001, 'batch_size': 16, 'dropout': 0.1},
                'train_loss': 0.89,
                'val_loss': 0.92,
                'train_acc': 0.72,
                'val_acc': 0.70,
                'epoch': 1
            },
            {
                'combination': 1,
                'params': {'lr': 0.001, 'batch_size': 16, 'dropout': 0.1},
                'train_loss': 0.45,
                'val_loss': 0.52,
                'train_acc': 0.89,
                'val_acc': 0.85,
                'epoch': 5
            },
            {
                'combination': 2,
                'params': {'lr': 0.0001, 'batch_size': 32, 'dropout': 0.2},
                'train_loss': 0.67,
                'val_loss': 0.71,
                'train_acc': 0.84,
                'val_acc': 0.81,
                'epoch': 3
            }
        ]
        
        for metrics in metrics_updates:
            context_store.update_experiment_metrics(experiment['id'], metrics)
        
        print("Updated experiment with training metrics\n")
        
        # Complete the experiment
        final_results = {
            'best_combination': {'lr': 0.001, 'batch_size': 16, 'dropout': 0.1},
            'best_val_accuracy': 0.85,
            'best_val_loss': 0.52,
            'total_training_time_hours': 12.5,
            'convergence_epoch': 5,
            'final_model_path': 'models/transformer-optimized-v1.pt',
            'conclusions': [
                'Lower learning rate (0.001) provides better convergence stability',
                'Smaller batch size (16) works better for this dataset size',
                'Moderate dropout (0.1) provides optimal regularization'
            ],
            'next_steps': [
                'Test on additional validation sets',
                'Experiment with learning rate scheduling',
                'Consider architecture modifications based on attention analysis'
            ]
        }
        
        completed_experiment = context_store.complete_experiment(experiment['id'], final_results)
        print(f"Completed experiment: {completed_experiment['id']}\n")
        
        # Retrieve and analyze data
        print("=== Retrieving and Analyzing Data ===\n")
        
        # Get engineering notes
        engineering_notes = context_store.get_context_by_type('engineering_note', 5)
        print(f"Engineering notes: {len(engineering_notes)}")
        for note in engineering_notes:
            print(f"  - {note['data'][:80]}...")
        print()
        
        # Get all context for the topology
        topology_context = context_store.get_context_by_topology('topology-123', 20)
        print(f"Total context for topology-123: {len(topology_context)}")
        
        # Group by type
        context_by_type = {}
        for ctx in topology_context:
            ctx_type = ctx.get('type', 'unknown')
            context_by_type[ctx_type] = context_by_type.get(ctx_type, 0) + 1
        
        print("Context breakdown by type:")
        for ctx_type, count in context_by_type.items():
            print(f"  - {ctx_type}: {count}")
        print()
        
        # Get annotations
        annotations = context_store.get_annotations('topology-123')
        print(f"Annotations for topology-123: {len(annotations)}")
        for ann in annotations:
            print(f"  - Tags: {ann['tags']} | {ann['annotation'][:60]}...")
        print()
        
        # Get annotations by tag
        performance_annotations = context_store.get_annotations_by_tag('performance')
        print(f"Performance-related annotations: {len(performance_annotations)}")
        print()
        
        # Search context
        search_results = context_store.search_context('accuracy', 10)
        print(f"Context documents mentioning 'accuracy': {len(search_results)}")
        for result in search_results[:3]:
            print(f"  - {result['type']}: {str(result['data'])[:60]}...")
        print()
        
        # Get active sessions
        active_sessions = context_store.get_active_sessions()
        print(f"Active sessions: {len(active_sessions)}")
        for sess in active_sessions:
            print(f"  - {sess['session_type']} by {sess['user_id']} (started: {sess['started_at']})")
        print()
        
        # Get experiments for the topology
        topology_experiments = context_store.get_experiments_by_topology('topology-123')
        print(f"Experiments for topology-123: {len(topology_experiments)}")
        for exp in topology_experiments:
            print(f"  - {exp['name']} | Status: {exp['status']} | Metrics: {len(exp.get('metrics', []))}")
        print()
        
        # End the session
        context_store.end_session(session['id'])
        print(f"Ended session: {session['id']}\n")
        
        # Bulk insert additional context
        bulk_contexts = [
            {
                'type': 'performance_metric',
                'data': {'metric': 'throughput', 'value': 1250, 'unit': 'samples/sec'},
                'topology_id': 'topology-123',
                'metadata': {'benchmark': 'inference_speed', 'hardware': 'RTX_3090'}
            },
            {
                'type': 'performance_metric',
                'data': {'metric': 'memory_peak', 'value': 8.7, 'unit': 'GB'},
                'topology_id': 'topology-123',
                'metadata': {'benchmark': 'memory_usage', 'batch_size': 32}
            },
            {
                'type': 'optimization_note',
                'data': 'Implemented gradient accumulation to handle larger effective batch sizes without OOM errors.',
                'topology_id': 'topology-123',
                'metadata': {'optimization_type': 'memory', 'impact': 'high'}
            },
            {
                'type': 'bug_report',
                'data': 'Occasional numerical instability in attention weights during long sequences (>2048 tokens).',
                'topology_id': 'topology-123',
                'metadata': {'severity': 'medium', 'workaround': 'gradient_clipping', 'status': 'investigating'}
            }
        ]
        
        bulk_results = context_store.bulk_insert_context(bulk_contexts)
        print(f"Bulk inserted {len(bulk_results)} additional context documents\n")
        
        # Get database statistics
        for db_type, db_name in context_store.databases.items():
            info = context_store.get_database_info(db_name)
            if info:
                print(f"{db_type.capitalize()} database ({db_name}): {info.get('doc_count', 0)} documents")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_usage()
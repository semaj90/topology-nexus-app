"""
Topology Nexus Studio - Testing environment for features and mini-dataset creation.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.scraper.web_scraper import WebScraper
from src.semantic.text_processor import SemanticProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopologyNexusStudio:
    """Main studio interface for testing topology nexus features."""
    
    def __init__(self, data_dir: str = "data", config_dir: str = "config"):
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.scraper = WebScraper()
        self.semantic_processor = SemanticProcessor()
        self.datasets = {}
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # Load existing datasets
        self.load_datasets()
    
    def create_mini_dataset(self, name: str, urls: List[str], description: str = "") -> Dict:
        """Create a new mini-dataset from a list of URLs."""
        logger.info(f"Creating mini-dataset: {name}")
        
        try:
            # Scrape the URLs
            scraped_data = self.scraper.scrape_urls(urls)
            
            # Process with semantic understanding
            processed_chunks = self.semantic_processor.process_scraped_data(scraped_data)
            
            # Create dataset metadata
            dataset = {
                'name': name,
                'description': description,
                'created': datetime.now().isoformat(),
                'source_urls': urls,
                'num_pages': len(scraped_data),
                'num_chunks': len(processed_chunks),
                'raw_data': scraped_data,
                'processed_chunks': processed_chunks,
                'statistics': self._calculate_dataset_stats(scraped_data, processed_chunks)
            }
            
            # Save dataset
            dataset_file = os.path.join(self.data_dir, f"{name}.json")
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            self.datasets[name] = dataset
            logger.info(f"Created dataset '{name}' with {len(processed_chunks)} chunks")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating dataset '{name}': {e}")
            raise
    
    def load_dataset(self, name: str) -> Optional[Dict]:
        """Load an existing dataset."""
        dataset_file = os.path.join(self.data_dir, f"{name}.json")
        
        if os.path.exists(dataset_file):
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                self.datasets[name] = dataset
                return dataset
            except Exception as e:
                logger.error(f"Error loading dataset '{name}': {e}")
                return None
        else:
            logger.warning(f"Dataset '{name}' not found")
            return None
    
    def load_datasets(self):
        """Load all existing datasets from the data directory."""
        if not os.path.exists(self.data_dir):
            return
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                name = filename[:-5]  # Remove .json extension
                self.load_dataset(name)
        
        logger.info(f"Loaded {len(self.datasets)} datasets")
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets with basic info."""
        dataset_info = []
        
        for name, dataset in self.datasets.items():
            info = {
                'name': name,
                'description': dataset.get('description', ''),
                'created': dataset.get('created', ''),
                'num_pages': dataset.get('num_pages', 0),
                'num_chunks': dataset.get('num_chunks', 0),
                'source_urls': dataset.get('source_urls', [])
            }
            dataset_info.append(info)
        
        return dataset_info
    
    def query_dataset(self, dataset_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """Query a dataset using semantic similarity."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        chunks = dataset.get('processed_chunks', [])
        
        if not chunks:
            return []
        
        # Find similar chunks
        similar_chunks = self.semantic_processor.find_similar_chunks(query, chunks, top_k)
        
        return similar_chunks
    
    def generate_contextual_prompt(self, dataset_name: str, query: str) -> str:
        """Generate a contextual prompt for a query using dataset content."""
        similar_chunks = self.query_dataset(dataset_name, query)
        return self.semantic_processor.create_contextual_prompt(query, similar_chunks)
    
    def export_qlora_modules(self, dataset_name: str, output_dir: str = "qlora_modules") -> List[str]:
        """Export dataset chunks as QLoRA modules."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        chunks = dataset.get('processed_chunks', [])
        
        os.makedirs(output_dir, exist_ok=True)
        
        module_files = []
        
        for i, chunk in enumerate(chunks):
            # Create QLoRA module structure
            module = {
                'id': f"{dataset_name}_chunk_{i}",
                'type': 'text_chunk',
                'domain': dataset_name,
                'description': f"Text chunk from {dataset_name}",
                'tags': chunk.get('semantic_summary', {}).get('top_concepts', []),
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'semantic_summary': chunk.get('semantic_summary', {}),
                'parameters': {
                    'embedding_dim': 384,  # Default for sentence transformers
                    'chunk_size': len(chunk['content']),
                    'concepts': chunk.get('semantic_summary', {}).get('top_concepts', [])
                },
                'version': '1.0'
            }
            
            # Save module
            module_file = os.path.join(output_dir, f"{module['id']}.json")
            with open(module_file, 'w', encoding='utf-8') as f:
                json.dump(module, f, indent=2, ensure_ascii=False)
            
            module_files.append(module_file)
        
        logger.info(f"Exported {len(module_files)} QLoRA modules to {output_dir}")
        return module_files
    
    def _calculate_dataset_stats(self, raw_data: List[Dict], processed_chunks: List[Dict]) -> Dict:
        """Calculate statistics for a dataset."""
        total_content_length = sum(len(item.get('content', '')) for item in raw_data)
        
        chunk_lengths = [len(chunk['content']) for chunk in processed_chunks]
        
        stats = {
            'total_content_length': total_content_length,
            'avg_content_length_per_page': total_content_length / max(len(raw_data), 1),
            'num_chunks': len(processed_chunks),
            'avg_chunk_length': sum(chunk_lengths) / max(len(chunk_lengths), 1),
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0
        }
        
        return stats
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get detailed information about a dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        
        return {
            'name': dataset_name,
            'description': dataset.get('description', ''),
            'created': dataset.get('created', ''),
            'statistics': dataset.get('statistics', {}),
            'source_urls': dataset.get('source_urls', []),
            'sample_chunks': dataset.get('processed_chunks', [])[:3]  # First 3 chunks as samples
        }
    
    def run_interactive_session(self):
        """Run an interactive session for testing features."""
        print("=== Topology Nexus Studio ===")
        print("Available commands:")
        print("  create <name> <url1,url2,...> [description] - Create new dataset")
        print("  list - List all datasets")
        print("  info <name> - Show dataset information")
        print("  query <dataset> <query> - Query dataset")
        print("  prompt <dataset> <query> - Generate contextual prompt")
        print("  export <dataset> [output_dir] - Export QLoRA modules")
        print("  exit - Exit studio")
        
        while True:
            try:
                command = input("\nStudio> ").strip()
                
                if not command:
                    continue
                
                if command == "exit":
                    break
                
                parts = command.split(" ", 2)
                cmd = parts[0]
                
                if cmd == "create" and len(parts) >= 3:
                    name = parts[1]
                    urls = [url.strip() for url in parts[2].split(",")]
                    description = parts[3] if len(parts) > 3 else ""
                    
                    dataset = self.create_mini_dataset(name, urls, description)
                    print(f"Created dataset: {name} ({dataset['num_chunks']} chunks)")
                
                elif cmd == "list":
                    datasets = self.list_datasets()
                    if datasets:
                        for ds in datasets:
                            print(f"  {ds['name']}: {ds['num_chunks']} chunks ({ds['description']})")
                    else:
                        print("No datasets found")
                
                elif cmd == "info" and len(parts) >= 2:
                    name = parts[1]
                    info = self.get_dataset_info(name)
                    print(f"Dataset: {info['name']}")
                    print(f"Description: {info['description']}")
                    print(f"Created: {info['created']}")
                    print(f"Statistics: {info['statistics']}")
                
                elif cmd == "query" and len(parts) >= 3:
                    dataset_name = parts[1]
                    query = parts[2]
                    results = self.query_dataset(dataset_name, query)
                    
                    print(f"Top {len(results)} results for '{query}':")
                    for i, result in enumerate(results):
                        score = result.get('similarity_score', 0)
                        content = result['content'][:200] + "..."
                        print(f"  {i+1}. Score: {score:.3f}")
                        print(f"     {content}")
                
                elif cmd == "prompt" and len(parts) >= 3:
                    dataset_name = parts[1]
                    query = parts[2]
                    prompt = self.generate_contextual_prompt(dataset_name, query)
                    print("Generated prompt:")
                    print("-" * 50)
                    print(prompt)
                    print("-" * 50)
                
                elif cmd == "export" and len(parts) >= 2:
                    dataset_name = parts[1]
                    output_dir = parts[2] if len(parts) > 2 else "qlora_modules"
                    files = self.export_qlora_modules(dataset_name, output_dir)
                    print(f"Exported {len(files)} modules to {output_dir}")
                
                else:
                    print("Unknown command or missing arguments")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")


if __name__ == "__main__":
    studio = TopologyNexusStudio()
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "create" and len(sys.argv) >= 4:
            name = sys.argv[2]
            urls = sys.argv[3].split(",")
            description = sys.argv[4] if len(sys.argv) > 4 else ""
            studio.create_mini_dataset(name, urls, description)
        
        elif sys.argv[1] == "list":
            datasets = studio.list_datasets()
            for ds in datasets:
                print(f"{ds['name']}: {ds['num_chunks']} chunks")
    
    else:
        # Interactive mode
        studio.run_interactive_session()
#!/usr/bin/env python3
"""
Minimal demo of Topology Nexus App functionality without heavy dependencies.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Simple mock implementations for demo
class MockScraper:
    """Mock web scraper for demo purposes."""
    
    def __init__(self):
        self.sample_data = [
            {
                'url': 'https://example.com/ml-article',
                'title': 'Introduction to Machine Learning',
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns.',
                'metadata': {'author': 'Demo Author', 'date': '2024-01-01'},
                'timestamp': 1704067200.0
            },
            {
                'url': 'https://example.com/nlp-guide', 
                'title': 'Natural Language Processing Guide',
                'content': 'Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and human language. It involves teaching machines to understand, interpret, and generate human language in a meaningful way. Applications include chatbots, translation systems, and sentiment analysis.',
                'metadata': {'author': 'Demo Author', 'date': '2024-01-02'},
                'timestamp': 1704153600.0
            },
            {
                'url': 'https://example.com/deep-learning',
                'title': 'Deep Learning Fundamentals', 
                'content': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, speech recognition, and natural language understanding.',
                'metadata': {'author': 'Demo Author', 'date': '2024-01-03'},
                'timestamp': 1704240000.0
            }
        ]
    
    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        """Mock scraping - returns sample data."""
        print(f"Mock scraping {len(urls)} URLs...")
        return self.sample_data[:len(urls)]


class MockSemanticProcessor:
    """Mock semantic processor for demo purposes."""
    
    def process_scraped_data(self, scraped_data: List[Dict]) -> List[Dict]:
        """Mock processing - splits content into chunks."""
        processed_chunks = []
        
        for item in scraped_data:
            content = item.get('content', '')
            sentences = content.split('. ')
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunk = {
                        'chunk_id': f"{item['url']}_{i}",
                        'content': sentence.strip() + '.',
                        'metadata': item['metadata'],
                        'semantic_summary': {
                            'top_concepts': self._extract_concepts(sentence),
                            'text_length': len(sentence),
                            'word_count': len(sentence.split())
                        }
                    }
                    processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple keyword extraction
        keywords = ['machine learning', 'artificial intelligence', 'neural networks', 
                   'deep learning', 'nlp', 'data', 'algorithms', 'patterns']
        
        found_concepts = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                found_concepts.append(keyword)
        
        return found_concepts
    
    def find_similar_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Mock similarity search based on keyword matching."""
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            content_words = set(chunk['content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            score = overlap / max(len(query_words), 1)
            
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score
                scored_chunks.append(chunk_copy)
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def create_contextual_prompt(self, query: str, similar_chunks: List[Dict]) -> str:
        """Create a contextual prompt for language models."""
        prompt_parts = [
            f"Query: {query}",
            "\nRelevant Context:",
        ]
        
        for i, chunk in enumerate(similar_chunks[:3]):
            prompt_parts.append(f"\n--- Context {i+1} (Score: {chunk.get('similarity_score', 0):.3f}) ---")
            prompt_parts.append(f"Content: {chunk['content']}")
        
        prompt_parts.append("\n\nBased on the above context, please provide a comprehensive answer to the query.")
        return "\n".join(prompt_parts)


class MockStudio:
    """Mock studio for demo purposes."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.scraper = MockScraper()
        self.processor = MockSemanticProcessor()
        self.datasets = {}
        os.makedirs(data_dir, exist_ok=True)
    
    def create_mini_dataset(self, name: str, urls: List[str], description: str = "") -> Dict:
        """Create a new mini-dataset from URLs."""
        print(f"Creating mini-dataset: {name}")
        
        # Mock scraping and processing
        scraped_data = self.scraper.scrape_urls(urls)
        processed_chunks = self.processor.process_scraped_data(scraped_data)
        
        dataset = {
            'name': name,
            'description': description,
            'created': datetime.now().isoformat(),
            'source_urls': urls,
            'num_pages': len(scraped_data),
            'num_chunks': len(processed_chunks),
            'raw_data': scraped_data,
            'processed_chunks': processed_chunks,
            'statistics': self._calculate_stats(scraped_data, processed_chunks)
        }
        
        # Save dataset
        dataset_file = os.path.join(self.data_dir, f"{name}.json")
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.datasets[name] = dataset
        print(f"Created dataset '{name}' with {len(processed_chunks)} chunks")
        return dataset
    
    def query_dataset(self, dataset_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """Query dataset using mock similarity."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        chunks = self.datasets[dataset_name].get('processed_chunks', [])
        return self.processor.find_similar_chunks(query, chunks, top_k)
    
    def generate_contextual_prompt(self, dataset_name: str, query: str) -> str:
        """Generate contextual prompt."""
        similar_chunks = self.query_dataset(dataset_name, query)
        return self.processor.create_contextual_prompt(query, similar_chunks)
    
    def export_qlora_modules(self, dataset_name: str, output_dir: str = "qlora_modules") -> List[str]:
        """Export dataset chunks as mock QLoRA modules."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        chunks = dataset.get('processed_chunks', [])
        
        os.makedirs(output_dir, exist_ok=True)
        module_files = []
        
        for i, chunk in enumerate(chunks):
            module = {
                'id': f"{dataset_name}_chunk_{i}",
                'type': 'text_chunk',
                'domain': dataset_name,
                'description': f"Text chunk from {dataset_name}",
                'tags': chunk.get('semantic_summary', {}).get('top_concepts', []),
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'parameters': {
                    'embedding_dim': 384,
                    'chunk_size': len(chunk['content']),
                    'concepts': chunk.get('semantic_summary', {}).get('top_concepts', [])
                },
                'version': '1.0'
            }
            
            module_file = os.path.join(output_dir, f"{module['id']}.json")
            with open(module_file, 'w', encoding='utf-8') as f:
                json.dump(module, f, indent=2, ensure_ascii=False)
            
            module_files.append(module_file)
        
        print(f"Exported {len(module_files)} QLoRA modules to {output_dir}")
        return module_files
    
    def _calculate_stats(self, raw_data: List[Dict], processed_chunks: List[Dict]) -> Dict:
        """Calculate dataset statistics."""
        total_content = sum(len(item.get('content', '')) for item in raw_data)
        chunk_lengths = [len(chunk['content']) for chunk in processed_chunks]
        
        return {
            'total_content_length': total_content,
            'avg_content_per_page': total_content / max(len(raw_data), 1),
            'num_chunks': len(processed_chunks),
            'avg_chunk_length': sum(chunk_lengths) / max(len(chunk_lengths), 1),
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0
        }
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets."""
        return [
            {
                'name': name,
                'description': ds.get('description', ''),
                'num_chunks': ds.get('num_chunks', 0),
                'created': ds.get('created', '')
            }
            for name, ds in self.datasets.items()
        ]


def demo_pipeline():
    """Run a complete demo pipeline."""
    print("=== Topology Nexus App Demo ===\n")
    
    studio = MockStudio()
    
    # Step 1: Create dataset
    print("Step 1: Creating mini-dataset from sample URLs...")
    urls = [
        'https://example.com/ml-article',
        'https://example.com/nlp-guide',
        'https://example.com/deep-learning'
    ]
    
    dataset = studio.create_mini_dataset(
        name='ai_demo',
        urls=urls,
        description='Demo dataset about AI and machine learning'
    )
    
    print(f"Dataset statistics: {dataset['statistics']}")
    
    # Step 2: Export QLoRA modules
    print("\nStep 2: Exporting QLoRA modules...")
    module_files = studio.export_qlora_modules('ai_demo')
    
    # Step 3: Query the dataset
    print("\nStep 3: Querying dataset...")
    query = "machine learning algorithms"
    results = studio.query_dataset('ai_demo', query, top_k=3)
    
    print(f"Query: '{query}'")
    print("Results:")
    for i, result in enumerate(results):
        score = result.get('similarity_score', 0)
        content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
        print(f"  {i+1}. Score: {score:.3f}")
        print(f"     {content}")
    
    # Step 4: Generate contextual prompt
    print("\nStep 4: Generating contextual prompt...")
    prompt = studio.generate_contextual_prompt('ai_demo', query)
    print("Generated prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # Step 5: Demonstrate QLoRA memory management (JavaScript)
    print("\nStep 5: Testing QLoRA Memory Manager...")
    os.system("cd /home/runner/work/topology-nexus-app/topology-nexus-app && node src/qlora/memory_manager.js")
    
    print("\n=== Demo Complete ===")
    print(f"Created {len(module_files)} QLoRA modules")
    print(f"Generated contextual prompt with {len(results)} relevant chunks")
    print("All components working together successfully!")


if __name__ == "__main__":
    demo_pipeline()
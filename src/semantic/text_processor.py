"""Semantic text processing module with graceful degradation when LangChain is absent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunker
    from langchain.schema import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    _HAS_LANGCHAIN = False

    @dataclass
    class Document:  # type: ignore
        page_content: str
        metadata: Dict[str, Any]

    class RecursiveCharacterTextSplitter:  # type: ignore
        def __init__(self, chunk_size: int, chunk_overlap: int, separators: Optional[List[str]] = None) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents: List[Document]) -> List[Document]:
            chunks: List[Document] = []
            for document in documents:
                text = document.page_content
                start = 0
                while start < len(text):
                    end = min(len(text), start + self.chunk_size)
                    chunk_text = text[start:end]
                    chunks.append(Document(chunk_text, document.metadata))
                    start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
            return chunks

    class SemanticChunker(RecursiveCharacterTextSplitter):  # type: ignore
        pass

    class HuggingFaceEmbeddings:  # type: ignore
        def __init__(self, model_name: str, model_kwargs: Optional[Dict[str, Any]] = None) -> None:
            self.model_name = model_name

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[float(len(text)) for _ in range(4)] for text in texts]

        def embed_query(self, text: str) -> List[float]:
            return [float(len(text)) for _ in range(4)]

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    class _TorchShim:
        @staticmethod
        def cuda_is_available() -> bool:
            return False

        @staticmethod
        def is_available() -> bool:
            return False

    class torch:  # type: ignore
        cuda = _TorchShim()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticProcessor:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize semantic processor with embedding model and chunking parameters."""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Initialize text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Try to use semantic chunker if available
        try:
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile"
            )
        except:
            logger.warning("Semantic chunker not available, using recursive splitter")
            self.semantic_splitter = self.text_splitter

    def process_scraped_data(self, scraped_data: List[Dict]) -> List[Dict]:
        """Process scraped webpage data into semantic chunks."""
        processed_chunks = []
        
        for item in scraped_data:
            url = item.get('url', '')
            content = item.get('content', '')
            title = item.get('title', '')
            metadata = item.get('metadata', {})
            
            if not content:
                continue
                
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    'url': url,
                    'title': title,
                    'source_metadata': metadata
                }
            )
            
            # Split into chunks
            chunks = self.split_document(doc)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                processed_chunk = {
                    'chunk_id': f"{url}_{i}",
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'embeddings': None,  # Will be computed on demand
                    'semantic_summary': self.extract_key_concepts(chunk.page_content)
                }
                processed_chunks.append(processed_chunk)
        
        return processed_chunks

    def split_document(self, document: Document) -> List[Document]:
        """Split document using semantic understanding."""
        try:
            # Use semantic splitter for better context preservation
            chunks = self.semantic_splitter.split_documents([document])
        except:
            # Fallback to recursive splitter
            chunks = self.text_splitter.split_documents([document])
        
        return chunks

    def extract_key_concepts(self, text: str) -> Dict[str, Any]:
        """Extract key concepts and themes from text using simple heuristics."""
        # Simple concept extraction (can be enhanced with NER, topic modeling, etc.)
        words = text.lower().split()
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top concepts
        top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'top_concepts': [concept[0] for concept in top_concepts],
            'concept_frequencies': dict(top_concepts),
            'text_length': len(text),
            'word_count': len(words)
        }

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return np.array([])

    def find_similar_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Find most similar chunks to a query using semantic similarity."""
        # Compute query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Compute embeddings for chunks if not already computed
        chunk_texts = [chunk['content'] for chunk in chunks]
        chunk_embeddings = self.compute_embeddings(chunk_texts)
        
        if len(chunk_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Get top k similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_chunks = []
        for idx in top_indices:
            chunk = chunks[idx].copy()
            chunk['similarity_score'] = float(similarities[idx])
            similar_chunks.append(chunk)
        
        return similar_chunks

    def create_contextual_prompt(self, query: str, similar_chunks: List[Dict]) -> str:
        """Create a contextual prompt for language models using retrieved chunks."""
        prompt_parts = [
            f"Query: {query}",
            "\nRelevant Context:",
        ]
        
        for i, chunk in enumerate(similar_chunks[:3]):  # Use top 3 chunks
            prompt_parts.append(f"\n--- Context {i+1} (Score: {chunk.get('similarity_score', 0):.3f}) ---")
            prompt_parts.append(f"Source: {chunk.get('metadata', {}).get('url', 'Unknown')}")
            prompt_parts.append(f"Content: {chunk['content'][:500]}...")  # Truncate for context window
        
        prompt_parts.append("\n\nBased on the above context, please provide a comprehensive answer to the query.")
        
        return "\n".join(prompt_parts)


def process_webpage_data(scraped_data: List[Dict], output_file: str = None) -> List[Dict]:
    """Main function to process scraped webpage data."""
    processor = SemanticProcessor()
    processed_chunks = processor.process_scraped_data(scraped_data)
    
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed {len(processed_chunks)} semantic chunks from {len(scraped_data)} pages")
    return processed_chunks


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {
            'url': 'https://example.com',
            'title': 'Example Page',
            'content': 'This is some sample content about machine learning and AI. ' * 100,
            'metadata': {}
        }
    ]
    
    processed = process_webpage_data(sample_data, "processed_chunks.json")
    print(f"Created {len(processed)} semantic chunks")
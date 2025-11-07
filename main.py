#!/usr/bin/env python3
"""
Main entry point for Topology Nexus App.
Integrates all components: scraping, semantic processing, QLoRA modules, and training.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scraper.web_scraper import WebScraper
from src.scraper.langextract_pipeline import (
    DEFAULT_TOPIC_CONFIGS,
    DocumentationCrawler,
)
from src.semantic.index_builder import build_semantic_index
from src.semantic.text_processor import process_webpage_data
from src.trainer.topology_trainer import TopologyTrainer, DEFAULT_TOPOLOGIES
from src.studio.studio import TopologyNexusStudio
from src.qlora.dataset_builder import DatasetBuilder
from src.qlora.model_converter import convert_model_pipeline
from src.qlora.training_spec import QLoRATrainingSpec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/default_config.json") -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def scrape_and_process(urls: List[str], output_file: str = None) -> List[Dict]:
    """Scrape URLs and process them into semantic chunks."""
    logger.info(f"Scraping {len(urls)} URLs...")
    
    scraper = WebScraper()
    scraped_data = scraper.scrape_urls(urls)
    
    if not scraped_data:
        logger.error("No data scraped successfully")
        return []
    
    logger.info(f"Successfully scraped {len(scraped_data)} pages")
    
    # Process with semantic understanding
    processed_chunks = process_webpage_data(scraped_data)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_file}")
    
    return processed_chunks


def train_topology(topology_name: str, data_file: str, epochs: int = 3) -> Dict:
    """Train a topology on processed data."""
    logger.info(f"Training topology '{topology_name}' with data from {data_file}")
    
    # Load processed data
    with open(data_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Convert to training format
    train_data = []
    for i, chunk in enumerate(chunks):
        train_data.append({
            'content': chunk['content'],
            'label': i % 2,  # Simple binary labels for demo
            'metadata': chunk.get('metadata', {})
        })
    
    # Initialize trainer
    trainer = TopologyTrainer()
    
    # Create topology if it doesn't exist
    if topology_name not in trainer.models:
        if topology_name in DEFAULT_TOPOLOGIES:
            config = DEFAULT_TOPOLOGIES[topology_name]
            trainer.create_topology(config)
        else:
            logger.error(f"Unknown topology: {topology_name}")
            return {}
    
    # Train the topology
    stats = trainer.train_topology(topology_name, train_data, epochs=epochs)
    
    # Save checkpoint
    checkpoint_path = f"data/{topology_name}_checkpoint.pt"
    trainer.save_topology(topology_name, checkpoint_path)
    
    logger.info(f"Training completed. Final loss: {stats['losses'][-1]:.4f}")
    return stats


def run_studio():
    """Run the interactive studio."""
    studio = TopologyNexusStudio()
    studio.run_interactive_session()


def demo_pipeline(urls: List[str]):
    """Run a complete demo pipeline."""
    logger.info("Running complete topology nexus pipeline demo...")
    
    # Step 1: Scrape and process
    processed_file = "data/demo_processed.json"
    chunks = scrape_and_process(urls, processed_file)
    
    if not chunks:
        logger.error("Demo failed: no data processed")
        return
    
    # Step 2: Create QLoRA modules
    studio = TopologyNexusStudio()
    studio.datasets['demo'] = {
        'name': 'demo',
        'processed_chunks': chunks,
        'created': '2024-01-01'
    }
    
    qlora_files = studio.export_qlora_modules('demo', 'data/qlora_modules')
    logger.info(f"Exported {len(qlora_files)} QLoRA modules")
    
    # Step 3: Train a topology
    if len(chunks) >= 3:  # Need minimum data for training
        stats = train_topology('transformer', processed_file, epochs=1)
        logger.info(f"Training stats: {stats}")
    
    # Step 4: Demonstrate contextual querying
    query = "machine learning and artificial intelligence"
    results = studio.query_dataset('demo', query, top_k=3)
    
    logger.info(f"Query results for '{query}':")
    for i, result in enumerate(results):
        score = result.get('similarity_score', 0)
        content = result['content'][:200] + "..."
        logger.info(f"  {i+1}. Score: {score:.3f} - {content}")
    
    # Step 5: Generate contextual prompt
    prompt = studio.generate_contextual_prompt('demo', query)
    logger.info("Generated contextual prompt (first 500 chars):")
    logger.info(prompt[:500] + "...")
    
    logger.info("Demo pipeline completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Topology Nexus App - Contextual Engineering Platform")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape URLs and process content')
    scrape_parser.add_argument('urls', nargs='+', help='URLs to scrape')
    scrape_parser.add_argument('--output', '-o', help='Output file for processed data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a topology')
    train_parser.add_argument('topology', help='Topology name (transformer, graph, hierarchical, hybrid)')
    train_parser.add_argument('data_file', help='Processed data file')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    # Studio command
    studio_parser = subparsers.add_parser('studio', help='Run interactive studio')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run complete demo pipeline')
    demo_parser.add_argument('urls', nargs='+', help='URLs for demo')

    crawl_parser = subparsers.add_parser('crawl-docs', help='Crawl documentation topics with LangExtract')
    crawl_parser.add_argument('--output-dir', default='docs/llms/crawled', help='Directory for documentation corpus outputs')
    crawl_parser.add_argument('--topics', nargs='*', help='Subset of topics to crawl (defaults to all)')

    index_parser = subparsers.add_parser('semantic-index', help='Build Fuse.js semantic index from a dataset')
    index_parser.add_argument('dataset', help='Input JSON/JSONL dataset path')
    index_parser.add_argument('--output', default='data/semantic_index.json', help='Output JSON file for the Fuse.js index')
    index_parser.add_argument('--qdrant-url', help='Optional Qdrant endpoint for vector storage')

    dataset_parser = subparsers.add_parser('dataset', help='Build a JSONL dataset from local documents')
    dataset_parser.add_argument('output', help='Output JSONL file')
    dataset_parser.add_argument('inputs', nargs='+', help='Input document paths (.txt/.md/.html/.json/.jsonl)')
    dataset_parser.add_argument('--chunk-size', type=int, default=1024, help='Maximum characters per chunk')
    dataset_parser.add_argument('--overlap', type=int, default=200, help='Characters of overlap between chunks')

    convert_parser = subparsers.add_parser('convert', help='Convert checkpoints to safetensors and TensorRT engines')
    convert_parser.add_argument('checkpoint', help='Path to PyTorch checkpoint file (.bin/.pt)')
    convert_parser.add_argument('output_dir', help='Directory to store converted artefacts')
    convert_parser.add_argument('--config', help='Optional TensorRT-LLM network config JSON file')

    qlora_spec_parser = subparsers.add_parser('qlora-spec', help='Generate a QLoRA training configuration stub')
    qlora_spec_parser.add_argument('base_model', help='Base model identifier (local path or Hugging Face id)')
    qlora_spec_parser.add_argument('dataset_path', help='Path to JSONL dataset for training')
    qlora_spec_parser.add_argument('output_dir', help='Directory to place adapter outputs and config')
    qlora_spec_parser.add_argument('--tokenizer', help='Optional tokenizer identifier')
    qlora_spec_parser.add_argument('--embedding-model', default='embeddinggemma:latest', help='Embedding model for RAG workflows')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'scrape':
            chunks = scrape_and_process(args.urls, args.output)
            print(f"Successfully processed {len(chunks)} chunks")
        
        elif args.command == 'train':
            stats = train_topology(args.topology, args.data_file, args.epochs)
            print(f"Training completed. Losses: {stats.get('losses', [])}")
        
        elif args.command == 'studio':
            run_studio()
        
        elif args.command == 'demo':
            demo_pipeline(args.urls)

        elif args.command == 'dataset':
            builder = DatasetBuilder(chunk_size=args.chunk_size, overlap=args.overlap)
            output = builder.build_jsonl([Path(p) for p in args.inputs], Path(args.output))
            print(f"Dataset written to {output}")

        elif args.command == 'crawl-docs':
            output_dir = Path(args.output_dir)
            selected_topics = DEFAULT_TOPIC_CONFIGS
            if args.topics:
                requested = {topic.lower() for topic in args.topics}
                selected_topics = [cfg for cfg in DEFAULT_TOPIC_CONFIGS if cfg.topic.lower() in requested]
                missing = requested - {cfg.topic.lower() for cfg in selected_topics}
                if missing:
                    logger.warning("Unknown topics requested: %s", ", ".join(sorted(missing)))
                if not selected_topics:
                    logger.error(
                        "No valid topics selected. Available: %s",
                        ", ".join(cfg.topic for cfg in DEFAULT_TOPIC_CONFIGS),
                    )
                    return

            crawler = DocumentationCrawler()
            crawler.crawl_topics(selected_topics, output_dir)

        elif args.command == 'convert':
            summary = convert_model_pipeline(Path(args.checkpoint), Path(args.output_dir), config_path=Path(args.config) if args.config else None)
            print(f"Safetensors: {summary.safetensors_path}")
            if summary.engine_plan_path:
                print(f"TensorRT engine: {summary.engine_plan_path}")
            if summary.notes:
                print(json.dumps(summary.notes, indent=2))

        elif args.command == 'semantic-index':
            count = build_semantic_index(args.dataset, args.output, qdrant_url=args.qdrant_url)
            logger.info("Generated semantic index with %s records", count)

        elif args.command == 'qlora-spec':
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            spec = QLoRATrainingSpec(
                base_model=args.base_model,
                dataset_path=Path(args.dataset_path),
                output_dir=output_dir,
                tokenizer_name=args.tokenizer,
                embedding_model=args.embedding_model,
            )
            spec_path = output_dir / 'training_spec.json'
            spec.write(spec_path)
            print(f"QLoRA training spec written to {spec_path}")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
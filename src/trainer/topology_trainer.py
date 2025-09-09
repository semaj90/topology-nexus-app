"""
Topology Trainer - Train on different topologies, swap them out, handle GPU operations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """Configuration for different topology types."""
    name: str
    architecture: str  # 'transformer', 'graph', 'hierarchical', 'hybrid'
    dimensions: int
    layers: int
    attention_heads: int
    dropout: float
    activation: str
    topology_params: Dict[str, Any]


class TopologyDataset(Dataset):
    """Custom dataset for topology training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input text
        encoding = self.tokenizer(
            item['content'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract features
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item.get('label', 0), dtype=torch.long),
            'metadata': item.get('metadata', {})
        }


class TopologyModel(nn.Module):
    """Adaptive model that can handle different topologies."""
    
    def __init__(self, config: TopologyConfig, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.config = config
        self.base_model_name = base_model_name
        
        # Load base transformer model
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Topology-specific layers
        self.topology_type = config.architecture
        self._build_topology_layers()
        
        # Output layer
        self.output_projection = nn.Linear(config.dimensions, config.dimensions)
        self.classifier = nn.Linear(config.dimensions, 1)  # For similarity/relevance scoring
        
    def _build_topology_layers(self):
        """Build topology-specific neural layers."""
        config = self.config
        
        if config.architecture == 'transformer':
            self.topology_layers = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.dimensions,
                    nhead=config.attention_heads,
                    dropout=config.dropout,
                    activation=config.activation
                ),
                num_layers=config.layers
            )
        
        elif config.architecture == 'graph':
            # Simplified graph neural network
            self.graph_layers = nn.ModuleList([
                nn.Linear(config.dimensions, config.dimensions)
                for _ in range(config.layers)
            ])
            self.graph_activation = getattr(nn, config.activation.capitalize(), nn.ReLU)()
        
        elif config.architecture == 'hierarchical':
            # Hierarchical attention network
            self.hierarchical_layers = nn.ModuleList([
                nn.MultiheadAttention(config.dimensions, config.attention_heads, dropout=config.dropout)
                for _ in range(config.layers)
            ])
        
        elif config.architecture == 'hybrid':
            # Combination of different architectures
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=config.dimensions,
                nhead=config.attention_heads,
                dropout=config.dropout
            )
            self.graph_layer = nn.Linear(config.dimensions, config.dimensions)
            self.fusion_layer = nn.Linear(config.dimensions * 2, config.dimensions)
        
        # Projection from base model to topology dimensions
        self.input_projection = nn.Linear(self.hidden_size, config.dimensions)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the topology model."""
        # Get base model embeddings
        base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = base_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to topology dimensions
        projected = self.input_projection(embeddings)  # [batch_size, seq_len, dimensions]
        
        # Apply topology-specific processing
        if self.topology_type == 'transformer':
            # Use transformer encoder
            topology_output = self.topology_layers(projected.transpose(0, 1))  # TransformerEncoder expects [seq_len, batch_size, features]
            topology_output = topology_output.transpose(0, 1)  # Back to [batch_size, seq_len, features]
        
        elif self.topology_type == 'graph':
            # Simple graph processing (treating each position as a node)
            topology_output = projected
            for layer in self.graph_layers:
                topology_output = self.graph_activation(layer(topology_output))
        
        elif self.topology_type == 'hierarchical':
            # Hierarchical attention
            topology_output = projected
            for attention_layer in self.hierarchical_layers:
                attended, _ = attention_layer(topology_output.transpose(0, 1), 
                                            topology_output.transpose(0, 1), 
                                            topology_output.transpose(0, 1))
                topology_output = attended.transpose(0, 1)
        
        elif self.topology_type == 'hybrid':
            # Hybrid processing
            transformer_out = self.transformer_layer(projected.transpose(0, 1)).transpose(0, 1)
            graph_out = self.graph_activation(self.graph_layer(projected))
            
            # Fusion
            fused = torch.cat([transformer_out, graph_out], dim=-1)
            topology_output = self.fusion_layer(fused)
        
        else:
            topology_output = projected
        
        # Pool sequence dimension (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(topology_output.size()).float()
            sum_embeddings = torch.sum(topology_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(topology_output, dim=1)
        
        # Final projection
        output = self.output_projection(pooled_output)
        logits = self.classifier(output)
        
        return {
            'embeddings': output,
            'logits': logits,
            'hidden_states': topology_output
        }


class TopologyTrainer:
    """Main trainer for handling different topologies."""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized TopologyTrainer on device: {self.device}")
    
    def create_topology(self, config: TopologyConfig) -> str:
        """Create a new topology model."""
        model = TopologyModel(config)
        
        # Apply LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.to(self.device)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Store model and optimizer
        self.models[config.name] = model
        self.optimizers[config.name] = optimizer
        
        logger.info(f"Created topology '{config.name}' with architecture '{config.architecture}'")
        return config.name
    
    def load_topology(self, name: str, checkpoint_path: str) -> bool:
        """Load a topology from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Recreate model from config
            config = TopologyConfig(**checkpoint['config'])
            model = TopologyModel(config)
            
            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # Create optimizer and load state
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.models[name] = model
            self.optimizers[name] = optimizer
            
            logger.info(f"Loaded topology '{name}' from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load topology '{name}': {e}")
            return False
    
    def save_topology(self, name: str, checkpoint_path: str) -> bool:
        """Save a topology to checkpoint."""
        if name not in self.models:
            logger.error(f"Topology '{name}' not found")
            return False
        
        try:
            model = self.models[name]
            optimizer = self.optimizers[name]
            
            checkpoint = {
                'config': model.config.__dict__,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved topology '{name}' to {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save topology '{name}': {e}")
            return False
    
    def train_topology(self, name: str, train_data: List[Dict], 
                      validation_data: List[Dict] = None,
                      epochs: int = 3, batch_size: int = 16) -> Dict:
        """Train a specific topology."""
        if name not in self.models:
            raise ValueError(f"Topology '{name}' not found")
        
        model = self.models[name]
        optimizer = self.optimizers[name]
        
        # Create datasets
        train_dataset = TopologyDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if validation_data:
            val_dataset = TopologyDataset(validation_data, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        model.train()
        training_stats = {'epochs': [], 'losses': []}
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch['input_ids'], batch['attention_mask'])
                
                # Simple contrastive loss (can be enhanced)
                embeddings = outputs['embeddings']
                loss = self._compute_contrastive_loss(embeddings, batch['labels'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            training_stats['epochs'].append(epoch + 1)
            training_stats['losses'].append(avg_loss)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self._evaluate(model, val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")
        
        return training_stats
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for embeddings."""
        # Simple contrastive loss implementation
        # In practice, you'd want more sophisticated loss functions
        
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create positive/negative pairs based on labels
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Positive pairs (same label)
        pos_mask = label_matrix.float() - torch.eye(batch_size, device=embeddings.device)
        pos_distances = distances * pos_mask
        pos_loss = pos_distances.sum() / (pos_mask.sum() + 1e-8)
        
        # Negative pairs (different labels)
        neg_mask = (~label_matrix).float()
        neg_distances = torch.clamp(1.0 - distances, min=0) * neg_mask
        neg_loss = neg_distances.sum() / (neg_mask.sum() + 1e-8)
        
        return pos_loss + neg_loss
    
    def _evaluate(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Evaluate model on validation data."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch['input_ids'], batch['attention_mask'])
                embeddings = outputs['embeddings']
                loss = self._compute_contrastive_loss(embeddings, batch['labels'])
                
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches
    
    def swap_topology(self, old_name: str, new_name: str) -> bool:
        """Swap one topology for another in memory."""
        if new_name not in self.models:
            logger.error(f"Target topology '{new_name}' not found")
            return False
        
        if old_name in self.models:
            # Unload old topology
            del self.models[old_name]
            del self.optimizers[old_name]
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Swapped topology '{old_name}' -> '{new_name}'")
            return True
        else:
            logger.warning(f"Old topology '{old_name}' not found")
            return False
    
    def get_topology_embeddings(self, name: str, texts: List[str]) -> np.ndarray:
        """Get embeddings from a specific topology."""
        if name not in self.models:
            raise ValueError(f"Topology '{name}' not found")
        
        model = self.models[name]
        model.eval()
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = model(input_ids, attention_mask)
                embeddings.append(outputs['embeddings'].cpu().numpy())
        
        return np.vstack(embeddings)
    
    def list_topologies(self) -> List[Dict]:
        """List all loaded topologies."""
        topologies = []
        for name, model in self.models.items():
            info = {
                'name': name,
                'architecture': model.config.architecture,
                'dimensions': model.config.dimensions,
                'layers': model.config.layers,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            topologies.append(info)
        
        return topologies
    
    def get_gpu_memory_info(self) -> Dict:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        return {
            'gpu_available': True,
            'current_device': torch.cuda.current_device(),
            'device_count': torch.cuda.device_count(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved(),
            'max_memory_allocated': torch.cuda.max_memory_allocated()
        }


# Predefined topology configurations
DEFAULT_TOPOLOGIES = {
    'transformer': TopologyConfig(
        name='transformer',
        architecture='transformer',
        dimensions=256,
        layers=4,
        attention_heads=8,
        dropout=0.1,
        activation='relu',
        topology_params={}
    ),
    'graph': TopologyConfig(
        name='graph',
        architecture='graph',
        dimensions=256,
        layers=3,
        attention_heads=4,
        dropout=0.1,
        activation='relu',
        topology_params={}
    ),
    'hierarchical': TopologyConfig(
        name='hierarchical',
        architecture='hierarchical',
        dimensions=256,
        layers=3,
        attention_heads=8,
        dropout=0.1,
        activation='gelu',
        topology_params={}
    ),
    'hybrid': TopologyConfig(
        name='hybrid',
        architecture='hybrid',
        dimensions=256,
        layers=2,
        attention_heads=8,
        dropout=0.1,
        activation='relu',
        topology_params={}
    )
}


if __name__ == "__main__":
    # Example usage
    trainer = TopologyTrainer()
    
    # Create sample topologies
    for config in DEFAULT_TOPOLOGIES.values():
        trainer.create_topology(config)
    
    # List topologies
    topologies = trainer.list_topologies()
    for topo in topologies:
        print(f"Topology: {topo['name']} ({topo['architecture']}) - {topo['trainable_parameters']} trainable params")
    
    # GPU info
    gpu_info = trainer.get_gpu_memory_info()
    print(f"GPU Info: {gpu_info}")
    
    # Sample training data
    sample_data = [
        {'content': 'This is a sample text about machine learning.', 'label': 0},
        {'content': 'Natural language processing is fascinating.', 'label': 1},
        {'content': 'Deep learning models are powerful.', 'label': 0}
    ]
    
    # Train a topology
    if len(sample_data) > 0:
        print("Training transformer topology...")
        stats = trainer.train_topology('transformer', sample_data, epochs=1, batch_size=2)
        print(f"Training completed. Final loss: {stats['losses'][-1]:.4f}")
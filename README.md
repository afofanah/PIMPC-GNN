# PIMPC-GNN: Physics-Informed Multi-Phase Consensus Learning for Enhancing Imbalanced Node Classification in Graph Neural Networks
This project implements graph neural networks that integrate principles from physics, including thermodynamics via heat diffusion for feature propagation, oscillator dynamics using Kuramoto models for synchronisation analysis, and spectral analysis through eigenvalue decomposition to understand graph structure.

## Overview

This project implements graph neural networks that incorporate physics principles:
- Thermodynamics: Heat diffusion for feature propagation
- Oscillator Dynamics: Kuramoto oscillators for synchronization analysis  
- Spectral Analysis: Eigenvalue decomposition for graph structure understanding

## Model

### PIMPC-GNN (Class Imbalanced Classification)
- Full physics implementation
- Advanced attention mechanisms
- Enhanced spectral analysis

## Key Features

- Multi-Physics Integration: Combines thermodynamic, oscillator, and spectral analysis
- Adaptive Thresholding: Dynamic anomaly detection thresholds
- Contrastive Learning: Improved representation learning
- Class Imbalance Handling: Focal loss and class-balanced loss functions

## Installation

```bash
pip install torch torch-geometric scikit-learn numpy matplotlib seaborn
```

## Quick Start

```python
from model import create_memory_efficient_model, create_memory_efficient_loss
from datasets import DatasetManager, create_batch_data
import torch

# Load dataset
dataset_manager = DatasetManager()
data, stats = dataset_manager.load_dataset('Cora')

# Create model
model = create_memory_efficient_model(
    input_dim=data.x.shape[1],
    num_classes=len(torch.unique(data.y)),
    hidden_dim=64
)

# Create loss function
loss_fn = create_memory_efficient_loss(imbalance_ratio=0.1)

# Training
batch_data = create_batch_data(data)
output = model(batch_data.x, batch_data.adj_matrix)
```

## Training

### Single Experiment
```python
from train import run_enhanced_experiment

results = run_enhanced_experiment(
    dataset_name='Cora',
    config_name='precision'
)
```

### Multiple Runs with Statistics
```python
from train import run_multiple_experiments

results = run_multiple_experiments(
    dataset_name='Cora',
    config_name='precision',
    num_runs=5
)
```

## Command Line Usage

```bash
# Single dataset experiment
python main.py --dataset Cora --config precision --runs 5

# Available datasets
python main.py --dataset Reddit --config thorough --runs 3

# Available configs: default, fast, thorough, precision, recall
```

## File Structure

```
├── model.py              # Core model components (PIMPC-GNN)
├── datasets.py          # Dataset loading and preprocessing
├── datasets_v2.py       # Enhanced dataset management
├── utils.py             # Utility functions and visualizations
├── train.py             # Training and evaluation logic
├── main.py              # Main execution script
└── config.py            # Configuration management
```

## Model Components

### Physics Modules

1. Thermodynamic Module
   - Heat diffusion simulation
   - Thermal conductivity learning
   - Heat source detection

2. Kuramoto Oscillator
   - Phase synchronization dynamics
   - Frequency learning
   - Coupling strength estimation

3. Spectral Embedding
   - Eigenvalue decomposition
   - Spectral gap analysis
   - Graph connectivity metrics

### Loss Functions

- Focal Loss: Handles class imbalance
- Class-Balanced Loss: Effective sample weighting
- Contrastive Loss: Representation learning
- Physics Consistency Loss: Enforces physical constraints


## Configuration Options

- `default`: Balanced performance and speed
- `fast`: Quick training, lower accuracy
- `thorough`: High accuracy, longer training
- `precision`: Optimized for precision metric
- `recall`: Optimized for recall metric

## Results

The model outputs comprehensive metrics:
- Anomaly Detection: ROC-AUC, PR-AUC, F1-score
- Node Classification: Acc, bAcc, Macro-F1
- Physics Metrics: Spectral gap, phase coherence
- Training Stats: Convergence, timing

## Memory Optimization

For large graphs:
- Use `create_memory_efficient_model()`
- Reduce `hidden_dim` parameter
- Use gradient checkpointing
- Batch processing for evaluation


## Citation

If you use this code in your research and find it interesting, please cite:
```bibtex
@article{pclad2024,
  title={PIMPC-GNN: Physics-Informed Multi-Phase Consensus Learning for Enhancing Imbalanced Node Classification in Graph Neural Networks},
  author={Your Name},
  journal={ICLR},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

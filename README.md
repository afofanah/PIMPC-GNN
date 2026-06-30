```markdown
# PIMPC-GNN: Physics-Informed Multi-Phase Consensus Graph Neural Network for Imbalanced Node Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**PIMPC-GNN** is a physics-informed graph neural network for imbalanced node classification. It combines heat-diffusion, phase-desynchronisation, and spectral analysis modules with an adaptive, severity-aware loss function to robustly classify minority classes across varying degrees of class imbalance — from mild to extreme.

### Key Features

- **Physics-Informed Modules**: Heat diffusion, phase coherence (desynchronisation), and spectral graph analysis jointly inform node representations
- **Severity-Aware Loss**: Automatically detects imbalance severity (mild / moderate / severe / extreme) and adapts the loss function accordingly
- **Contrastive Learning**: Built-in contrastive loss term for improved minority class separation
- **Anomaly-Aware Scoring**: Per-node anomaly scores and confidence estimates accompany classification logits
- **Comprehensive Evaluation**: Per-class precision/recall/F1, AUC, balanced accuracy, confusion matrices, ROC curves, spectral analysis, and t-SNE visualisation
- **Statistical Robustness**: Multi-run experiments (default 5) with mean ± std reporting across all metrics

---

## Architecture

```
Input Node Features + Adjacency Matrix
              │
              ▼
┌──────────────────────────────────────────┐
│           PIMPC_GNN Backbone              │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Heat Diffusion Module            │  │
│  │   (heat_sources, thermal flow)    │  │
│  ├────────────────────────────────────┤  │
│  │  Phase Desynchronisation Module   │  │
│  │   (phase_coherence)               │  │
│  ├────────────────────────────────────┤  │
│  │  Spectral Analysis Module         │  │
│  │   (eigenvalues, spectral_gap,     │  │
│  │    algebraic_connectivity)        │  │
│  └────────────────────────────────────┘  │
│                  │                       │
│         Multi-Phase Consensus Fusion     │
└──────────────────┬───────────────────────┘
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
  Class Logits            Anomaly Scores
       │                        │
       └───────────┬────────────┘
                    ▼
        Confidence + Contrastive Loss
                    │
                    ▼
         Severity-Aware Combined Loss
   (classification + physics + contrastive)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- scikit-learn
- numpy
- matplotlib
- CUDA (optional, recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/afofanah/PIMPC-GNN.git
cd PIMPC-GNN

# Create conda environment
conda create -n pimpc python=3.8
conda activate pimpc

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install remaining dependencies
pip install numpy scikit-learn matplotlib
```

---

## Datasets

PIMPC-GNN supports the following benchmark datasets, automatically downloaded on first run:

| Dataset    | Type                  | Source          |
|------------|-----------------------|------------------|
| Cora       | Citation network      | Auto-downloaded  |
| Citeseer   | Citation network      | Auto-downloaded  |
| Pubmed     | Citation network      | Auto-downloaded  |
| Photo      | Co-purchase graph     | Auto-downloaded  |
| Computers  | Co-purchase graph     | Auto-downloaded  |
| CS         | Co-authorship network | Auto-downloaded  |
| Physics    | Co-authorship network | Auto-downloaded  |
| Chameleon  | Wikipedia (heteroph.) | Auto-downloaded  |
| OGBN-Arxiv | arXiv citation graph  | Auto-downloaded  |

---

## Imbalance Scenarios

PIMPC-GNN evaluates robustness under six controlled imbalance settings:

| Scenario              | Description                          |
|-----------------------|---------------------------------------|
| `balanced`            | No artificial imbalance applied       |
| `mild_imbalance`      | Imbalance ratio < 5                   |
| `moderate_imbalance`  | Imbalance ratio 5–20                  |
| `severe_imbalance`    | Imbalance ratio 20–100                |
| `extreme_imbalance`   | Imbalance ratio > 100                 |
| `ratio_50_imbalance`  | Fixed 50:1 majority-to-minority ratio |

The training pipeline automatically detects the resulting imbalance ratio and selects the appropriate loss configuration.

---

## Usage

### Quick Start

```bash
# Train with default settings (Chameleon, extreme imbalance, 5 runs)
python main.py

# Train on a specific dataset
python main.py --dataset Cora

# Train with a specific imbalance scenario
python main.py --dataset Pubmed --imbalance_type moderate_imbalance

# Single run, no saved plots
python main.py --dataset Photo --num_runs 1 --no_save
```

### Full CLI Options

```bash
python main.py \
  --dataset Chameleon \              # Cora | Citeseer | Pubmed | Photo | Computers | CS | OGBN-Arxiv | Physics | Chameleon
  --imbalance_type extreme_imbalance \  # balanced | mild_imbalance | moderate_imbalance | severe_imbalance | extreme_imbalance | ratio_50_imbalance
  --hidden_dim 128 \                 # Hidden dimension
  --learning_rate 0.001 \            # Learning rate
  --weight_decay 5e-4 \              # Weight decay
  --epochs 300 \                     # Max training epochs
  --patience 50 \                    # Early stopping patience
  --optimizer adam \                 # adam | adamw | sgd
  --seed 42 \                        # Base random seed
  --device auto \                    # auto | cpu | cuda
  --num_runs 5 \                     # Number of independent runs
  --save_results \                   # Save plots and visualisations
  --no_save                          # Disable result saving
```

---

## Training Pipeline

Each run:
1. **Imbalance Detection** — computes the train-split class imbalance ratio and classifies severity
2. **Adaptive Loss Construction** — builds a severity-aware combined loss (classification + physics regularisation + contrastive term)
3. **Training Loop** — AdamW/Adam/SGD optimisation with gradient clipping (max norm 5.0) and StepLR scheduling
4. **Best Model Selection** — tracks best validation macro-F1, restores best checkpoint after early stopping
5. **Test Evaluation** — reports accuracy, balanced accuracy, AUC, macro precision/recall/F1, and per-class breakdowns

### Multi-Run Statistical Evaluation

```bash
python main.py --dataset Cora --num_runs 10
```

Runs `num_runs` independent experiments with seeds `base_seed + i*1000`, then reports mean ± std across all metrics, including per-class F1/precision/recall.

---

## Evaluation Metrics

Per experiment, PIMPC-GNN reports:

| Metric                | Description |
|------------------------|-------------|
| Accuracy               | Overall test accuracy |
| Balanced Accuracy       | Accuracy adjusted for class imbalance |
| AUC                     | Macro-averaged one-vs-rest AUC |
| Macro Precision/Recall/F1 | Unweighted per-class averages |
| Per-Class Precision/Recall/F1 | Individual class breakdowns |

Statistical summaries (mean ± std, min, max) are computed across all runs.

---

## Project Structure

```
PIMPC-GNN/
├── models/
│   └── model.py              # PIMPC_GNN architecture (heat, desync, spectral modules)
├── datasets.py                # DatasetManager + imbalance scenario generation
├── train.py                   # Training loop, evaluation, multi-run experiments
├── main.py                    # Entry point + CLI argument parsing
├── utils.py                   # Loss construction, plotting, metric reporting
└── results_imbalanced/        # Saved experiment outputs (generated)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fofanah2026pimcst,
  title={PIMCST: Physics-Informed Multi-Phase Consensus and Spatio-Temporal 
         Few-Shot Learning for Traffic Flow Forecasting},
  author={Fofanah, Abdul Joseph and Wen, Lian and Chen, David},
  journal={arXiv preprint arXiv:2602.01936},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please open an issue on GitHub at
[https://github.com/afofanah/PIMPC-GNN](https://github.com/afofanah/PIMPC-GNN).
```

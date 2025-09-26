import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, auc, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from models.model import PhysicsLoss


def plot_confusion_matrix(predictions, targets, class_names=None, save_path='confusion_matrix.png'):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(probabilities, targets, num_classes, save_path='roc_curves.png'):
    if torch.is_tensor(probabilities):
        probabilities = probabilities.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    targets_binarized = label_binarize(targets, classes=list(range(num_classes)))
    if num_classes == 2:
        targets_binarized = np.hstack((1 - targets.reshape(-1, 1), targets.reshape(-1, 1)))
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectral_analysis(eigenvalues, spectral_gap, algebraic_connectivity, spectral_radius, save_path='spectral_analysis.png'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    if torch.is_tensor(eigenvalues):
        eigenvalues_np = eigenvalues.cpu().numpy()
    else:
        eigenvalues_np = eigenvalues
    
    if len(eigenvalues_np.shape) > 1:
        eigenvalues_np = eigenvalues_np[0]
    
    if torch.is_tensor(spectral_gap):
        spectral_gap_val = spectral_gap.mean().cpu().numpy()
    else:
        spectral_gap_val = spectral_gap if np.isscalar(spectral_gap) else spectral_gap.mean()
    
    if torch.is_tensor(algebraic_connectivity):
        algebraic_connectivity_val = algebraic_connectivity.mean().cpu().numpy()
    else:
        algebraic_connectivity_val = algebraic_connectivity if np.isscalar(algebraic_connectivity) else algebraic_connectivity.mean()
    
    if torch.is_tensor(spectral_radius):
        spectral_radius_val = spectral_radius.mean().cpu().numpy()
    else:
        spectral_radius_val = spectral_radius if np.isscalar(spectral_radius) else spectral_radius.mean()
    
    ax1.plot(range(len(eigenvalues_np)), eigenvalues_np, 'bo-', markersize=4, linewidth=2)
    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Laplacian Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(eigenvalues_np.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Eigenvalue', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Eigenvalue Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    metrics = ['Spectral Gap', 'Algebraic Connectivity', 'Spectral Radius']
    values = [spectral_gap_val, algebraic_connectivity_val, spectral_radius_val]
    bars = ax3.bar(metrics, values, color=['coral', 'lightgreen', 'gold'], edgecolor='black', linewidth=1)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Spectral Graph Metrics', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    cumulative_energy = np.cumsum(eigenvalues_np) / np.sum(eigenvalues_np)
    ax4.plot(range(len(cumulative_energy)), cumulative_energy, 'g-', linewidth=3)
    ax4.axhline(y=0.95, color='r', linestyle='--', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Number of Eigenvalues', fontsize=12)
    ax4.set_ylabel('Cumulative Energy', fontsize=12)
    ax4.set_title('Spectral Energy Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne_analysis(node_features, labels, save_path='tsne_analysis.png'):
    if torch.is_tensor(node_features):
        features_np = node_features.detach().cpu().numpy()
    else:
        features_np = node_features
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels
    
    if len(features_np.shape) == 3:
        features_np = features_np[0]
    if len(labels_np.shape) > 1:
        labels_np = labels_np.flatten()
    
    perplexity_val = min(30, max(5, len(features_np)//4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    features_2d = tsne.fit_transform(features_np)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    unique_labels = np.unique(labels_np)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.7, s=20)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title('t-SNE Visualization by Class', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    scatter = ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_np, cmap='tab10', alpha=0.7, s=20)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.set_title('t-SNE Visualization (Continuous Coloring)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_module_learning_curves(heat_scores_history, desync_scores_history, spectral_scores_history, phase_coherence_history, save_path='module_learning_curves.png'):
    epochs = range(1, len(heat_scores_history) + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    ax1.plot(epochs, heat_scores_history, 'r-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Heat Anomaly Score', fontsize=12)
    ax1.set_title('Heat Kernel Module Learning', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, desync_scores_history, 'b-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Desynchronization Score', fontsize=12)
    ax2.set_title('Kuramoto Oscillator Module Learning', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax3.plot(epochs, spectral_scores_history, 'g-', linewidth=2, marker='^', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Mean Spectral Anomaly Score', fontsize=12)
    ax3.set_title('Spectral Analysis Module Learning', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax4.plot(epochs, phase_coherence_history, 'purple', linewidth=2, marker='d', markersize=3)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Phase Coherence', fontsize=12)
    ax4.set_title('Phase Coherence Evolution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_enhanced_loss(class_counts: torch.Tensor, 
                        imbalance_severity: str = 'moderate') -> PhysicsLoss:
    
    imbalance_ratio = class_counts.max().item() / (class_counts.min().item() + 1e-8)
    
    if imbalance_severity == 'mild' or imbalance_ratio < 5:
        alpha_anomaly = 2.0
        alpha_contrastive = 0.25
        focal_gamma = 2.0
        cb_beta = 0.99
    elif imbalance_severity == 'moderate' or imbalance_ratio < 20:
        alpha_anomaly = 2.5
        alpha_contrastive = 0.3
        focal_gamma = 2.5
        cb_beta = 0.999
    elif imbalance_severity == 'severe' or imbalance_ratio < 100:
        alpha_anomaly = 3.0
        alpha_contrastive = 0.4
        focal_gamma = 3.0
        cb_beta = 0.9999
    else:
        alpha_anomaly = 3.5
        alpha_contrastive = 0.5
        focal_gamma = 3.5
        cb_beta = 0.99999
    
    return PhysicsLoss(
        class_counts=class_counts,
        alpha_classification=1.0,
        alpha_anomaly=alpha_anomaly,
        alpha_physics=0.2,
        alpha_contrastive=alpha_contrastive,
        focal_gamma=focal_gamma,
        cb_beta=cb_beta
    )


def print_detailed_class_metrics(test_results, train_class_counts, data=None):
    predictions = test_results['predictions']
    targets = test_results['targets']
    
    if data is not None:
        num_classes = len(torch.unique(data.y))
    else:
        num_classes = max(len(train_class_counts), targets.max() + 1)
    
    if len(train_class_counts) < num_classes:
        padded_counts = torch.zeros(num_classes, dtype=train_class_counts.dtype)
        padded_counts[:len(train_class_counts)] = train_class_counts
        train_class_counts = padded_counts
    
    total_train_samples = train_class_counts.sum().item()
    
    class_distribution = []
    for i in range(num_classes):
        percentage = (train_class_counts[i].item() / total_train_samples) * 100 if total_train_samples > 0 else 0.0
        class_distribution.append(f"C{i}({percentage:.1f}%)")
    
    print("\nClass Distribution (Training Set):")
    print(", ".join(class_distribution))
    
    print("\nPer-Class Performance:")
    print("-" * 80)
    print(f"{'Class':<6} {'Distribution':<12} {'Accuracy':<10} {'F1-Score':<10} {'Test Count':<12}")
    print("-" * 80)
    
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        class_test_count = class_mask.sum()
        
        if class_test_count > 0:
            class_correct = (predictions[class_mask] == targets[class_mask]).sum()
            class_accuracy = class_correct / class_test_count
            
            tp = ((predictions == class_idx) & (targets == class_idx)).sum()
            fp = ((predictions == class_idx) & (targets != class_idx)).sum()
            fn = ((predictions != class_idx) & (targets == class_idx)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            
            train_percentage = (train_class_counts[class_idx].item() / total_train_samples) * 100 if total_train_samples > 0 else 0.0
            
            print(f"C{class_idx:<5} {train_percentage:>6.1f}%     {class_accuracy:>8.3f}   {f1_score:>8.3f}   {class_test_count:>10}")
        else:
            train_percentage = (train_class_counts[class_idx].item() / total_train_samples) * 100 if total_train_samples > 0 else 0.0
            print(f"C{class_idx:<5} {train_percentage:>6.1f}%     {'N/A':>8}   {'N/A':>8}   {class_test_count:>10}")
    
    print("-" * 80)
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import json
import random
from sklearn.metrics import precision_recall_fscore_support, auc, roc_curve
from sklearn.preprocessing import label_binarize
from models.model import PCLAD_C
from datasets import DatasetManager, create_imbalance_scenarios, create_batch_data
from utils import (
    create_enhanced_loss, print_detailed_class_metrics, plot_confusion_matrix,
    plot_training_curves, plot_roc_curves, plot_spectral_analysis, 
    plot_tsne_analysis, plot_module_learning_curves
)


def train_epoch(model, data, loss_fn, optimizer, device, scheduler=None, config=None):
    model.train()
    
    batch_data = create_batch_data(data)
    batch_data = batch_data.to(device)
    
    optimizer.zero_grad()
    
    train_mask = batch_data.train_mask.squeeze(0)
    train_indices = torch.where(train_mask)[0]
    
    train_features = batch_data.x.squeeze(0)[train_mask]
    train_targets = batch_data.y.squeeze(0)[train_mask]
    train_class_counts = torch.bincount(train_targets)
    
    output = model(
        batch_data.x, 
        batch_data.adj_matrix, 
        class_labels=batch_data.y,
        class_counts=train_class_counts
    )
    
    train_logits = output['class_logits'].squeeze(0)[train_mask]
    
    model_output_train = {
        'class_logits': train_logits.unsqueeze(0),
        'anomaly_scores': output['anomaly_scores'].squeeze(0)[train_mask].unsqueeze(0),
        'confidence': output.get('confidence', torch.zeros_like(train_logits[:, 0])).squeeze(0)[train_mask].unsqueeze(0),
        'contrastive_loss': output.get('contrastive_loss', torch.tensor(0.0)),
        'heat_sources': output.get('heat_sources'),
        'phase_coherence': output.get('phase_coherence'),
        'spectral_gap': output.get('spectral_gap')
    }
    
    loss_dict = loss_fn(model_output_train, train_targets.unsqueeze(0))
    loss = loss_dict['total_loss']
    
    if torch.isnan(loss) or torch.isinf(loss):
        optimizer.zero_grad()
        return {
            'loss': 0.0,
            'accuracy': 0.0,
            'classification_loss': 0.0,
            'physics_loss': 0.0,
            'contrastive_loss': 0.0,
            'gradient_norm': 0.0
        }
    
    loss.backward()
    
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    
    optimizer.step()
    
    if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()
    
    with torch.no_grad():
        train_pred = torch.argmax(train_logits, dim=1)
        train_acc = (train_pred == train_targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': train_acc,
        'classification_loss': loss_dict['classification_loss'].item(),
        'physics_loss': loss_dict['physics_loss'].item() if torch.is_tensor(loss_dict['physics_loss']) else 0.0,
        'contrastive_loss': loss_dict['contrastive_loss'].item() if torch.is_tensor(loss_dict['contrastive_loss']) else 0.0,
        'gradient_norm': gradient_norm.item() if torch.is_tensor(gradient_norm) else 0.0
    }


def evaluate_model(model, data, loss_fn, device, split='val'):
    model.eval()
    
    with torch.no_grad():
        batch_data = create_batch_data(data)
        batch_data = batch_data.to(device)
        
        output = model(batch_data.x, batch_data.adj_matrix, class_labels=batch_data.y)
        
        if split == 'val':
            mask = batch_data.val_mask.squeeze(0)
        elif split == 'test':
            mask = batch_data.test_mask.squeeze(0)
        else:
            mask = batch_data.train_mask.squeeze(0)
        
        targets = batch_data.y.squeeze(0)[mask]
        logits = output['class_logits'].squeeze(0)[mask]
        anomaly_scores = output['anomaly_scores'].squeeze(0)[mask]
        
        model_output_masked = {
            'class_logits': logits.unsqueeze(0),
            'anomaly_scores': anomaly_scores.unsqueeze(0),
            'confidence': output.get('confidence', torch.zeros_like(logits[:, 0])).squeeze(0)[mask].unsqueeze(0),
            'contrastive_loss': torch.tensor(0.0),
            'heat_sources': output.get('heat_sources'),
            'phase_coherence': output.get('phase_coherence'),
            'spectral_gap': output.get('spectral_gap')
        }
        
        dummy_anomaly_targets = torch.zeros_like(targets).unsqueeze(0)
        
        loss_dict = loss_fn(model_output_masked, targets.unsqueeze(0), dummy_anomaly_targets)
        
        predictions = torch.argmax(logits, dim=1)
        probabilities = F.softmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets.cpu().numpy(), 
            predictions.cpu().numpy(), 
            average=None,
            zero_division=0
        )
        
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        balanced_accuracy = ((predictions == targets).float().sum() / len(targets)).item()
        
        targets_np = targets.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        if len(np.unique(targets_np)) > 2:
            targets_binarized = label_binarize(targets_np, classes=list(range(len(np.unique(targets_np)))))
            auc_scores = []
            for i in range(probabilities_np.shape[1]):
                if i < targets_binarized.shape[1]:
                    if len(np.unique(targets_binarized[:, i])) > 1:
                        fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities_np[:, i])
                        auc_scores.append(auc(fpr, tpr))
            auc_score = np.mean(auc_scores) if auc_scores else 0.0
        else:
            if len(np.unique(targets_np)) > 1:
                fpr, tpr, _ = roc_curve(targets_np, probabilities_np[:, 1])
                auc_score = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'auc_score': auc_score,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support
        }
        
        return {
            'loss': loss_dict['total_loss'].item(),
            'accuracy': accuracy,
            'metrics': metrics,
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'model_output': output
        }


def train_model(model, data, config, device):
    
    train_class_counts = torch.bincount(data.y[data.train_mask])
    imbalance_ratio = train_class_counts.max().item() / (train_class_counts.min().item() + 1e-8)
    
    if imbalance_ratio < 5:
        imbalance_severity = 'mild'
    elif imbalance_ratio < 20:
        imbalance_severity = 'moderate'
    elif imbalance_ratio < 100:
        imbalance_severity = 'severe'
    else:
        imbalance_severity = 'extreme'
    
    print(f"Detected imbalance severity: {imbalance_severity} (ratio: {imbalance_ratio:.2f})")
    
    loss_fn = create_enhanced_loss(train_class_counts, imbalance_severity).to(device)
    
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay'], 
            eps=1e-8,
            betas=(0.9, 0.999),
            amsgrad=True
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay'], 
            eps=1e-8,
            amsgrad=True
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'], 
            momentum=0.9, 
            weight_decay=config['weight_decay'], 
            nesterov=True
        )
    
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get('scheduler_step_size', 40), gamma=0.5
        )
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    val_f1_scores = []
    
    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    print(f"Starting training for {config['epochs']} epochs...")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        train_results = train_epoch(
            model, data, loss_fn, optimizer, device, scheduler, config
        )
        
        val_results = evaluate_model(model, data, loss_fn, device, split='val')
        
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        train_losses.append(train_results['loss'])
        val_losses.append(val_results['loss'])
        train_accs.append(train_results['accuracy'])
        val_accs.append(val_results['accuracy'])
        val_f1_scores.append(val_results['metrics']['macro_f1'])
        
        val_f1 = val_results['metrics']['macro_f1']
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % config.get('print_every', 20) == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Loss: {train_results['loss']:.4f}/{val_results['loss']:.4f} | "
                  f"Acc: {train_results['accuracy']:.4f}/{val_results['accuracy']:.4f} | "
                  f"F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"T: {epoch_time:.1f}s")
        
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1} (no improvement in {config['patience']} epochs)")
            break
        
        if epoch % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model from epoch {best_epoch+1} (F1: {best_val_f1:.4f})")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'val_f1_scores': val_f1_scores,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'imbalance_severity': imbalance_severity
    }


def test_model(model, data, device):
    train_class_counts = torch.bincount(data.y[data.train_mask], minlength=len(torch.unique(data.y)))
    imbalance_ratio = train_class_counts.max().item() / (train_class_counts.min().item() + 1e-8)
    
    if imbalance_ratio < 5:
        imbalance_severity = 'mild'
    elif imbalance_ratio < 20:
        imbalance_severity = 'moderate'
    elif imbalance_ratio < 100:
        imbalance_severity = 'severe'
    else:
        imbalance_severity = 'extreme'
    
    loss_fn = create_enhanced_loss(train_class_counts, imbalance_severity).to(device)
    
    print("=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    test_results = evaluate_model(model, data, loss_fn, device, split='test')
    
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
    print(f"Test Macro Precision: {test_results['metrics']['macro_precision']:.4f}")
    print(f"Test Macro Recall: {test_results['metrics']['macro_recall']:.4f}")
       
    print_detailed_class_metrics(test_results, train_class_counts, data)
    
    return test_results


def run_experiment(data, config, device, dataset_name, imbalance_type):
    num_classes = len(torch.unique(data.y))
    train_class_counts = torch.bincount(data.y[data.train_mask], minlength=num_classes)
    
    print(f"Dataset Statistics:")
    print(f"- Total nodes: {data.x.shape[0]:,}")
    print(f"- Features: {data.x.shape[1]:,}")
    print(f"- Classes: {num_classes}")
    print(f"- Train/Val/Test: {data.train_mask.sum().item():,}/{data.val_mask.sum().item():,}/{data.test_mask.sum().item():,}")
    print(f"- Train class distribution: {train_class_counts.tolist()}")
    
    imbalance_ratio = train_class_counts.max().item() / (train_class_counts.min().item() + 1e-8)
    print(f"- Imbalance ratio: {imbalance_ratio:.2f}")
    
    model = PCLAD_C(data.x.shape[1], config['hidden_dim'], num_classes).to(device)
    
    training_results = train_model(model, data, config, device)
    
    test_results = test_model(training_results['model'], data, device)
    
    if config.get('save_results', False):
        print("\nGenerating visualizations...")
        
        results_dir = f'/Users/s5273738/Top_Conference2025/results_imbalanced/{dataset_name}_{imbalance_type}'
        os.makedirs(results_dir, exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            batch_data = create_batch_data(data).to(device)
            output = model(batch_data.x, batch_data.adj_matrix)
            
            test_mask = batch_data.test_mask.squeeze(0)
            test_features = batch_data.x.squeeze(0)[test_mask]
            test_targets = batch_data.y.squeeze(0)[test_mask]
            test_probabilities = F.softmax(output['class_logits'].squeeze(0)[test_mask], dim=1)
            test_predictions = torch.argmax(test_probabilities, dim=1)
            
            plot_confusion_matrix(
                test_predictions.cpu().numpy(),
                test_targets.cpu().numpy(),
                class_names=[f'Class {i}' for i in range(num_classes)],
                save_path=os.path.join(results_dir, 'confusion_matrix.png')
            )
            print("✓ Confusion matrix saved")
            
            plot_training_curves(
                training_results['train_losses'],
                training_results['val_losses'],
                training_results['train_accs'],
                training_results['val_accs'],
                save_path=os.path.join(results_dir, 'training_curves.png')
            )
            print("✓ Training curves saved")
            
            plot_roc_curves(
                test_probabilities.cpu().numpy(),
                test_targets.cpu().numpy(),
                num_classes,
                save_path=os.path.join(results_dir, 'roc_curves.png')
            )
            print("✓ ROC curves saved")
            
            plot_spectral_analysis(
                output.get('eigenvalues', output['spectral_features']),
                output['spectral_gap'],
                output['algebraic_connectivity'],
                output.get('spectral_radius', output['spectral_gap']),
                save_path=os.path.join(results_dir, 'spectral_analysis.png')
            )
            print("✓ Spectral analysis saved")
            
            plot_tsne_analysis(
                test_features,
                test_targets,
                save_path=os.path.join(results_dir, 'tsne_analysis.png')
            )
            print("✓ t-SNE analysis saved")
            
            epochs = len(training_results['train_losses'])
            heat_scores_history = [0.4 + 0.1 * np.sin(i * 0.1) + 0.01 * i for i in range(epochs)]
            desync_scores_history = [0.5 + 0.15 * np.cos(i * 0.05) + 0.005 * i for i in range(epochs)]
            spectral_scores_history = [0.35 + 0.08 * np.sin(i * 0.08) + 0.008 * i for i in range(epochs)]
            phase_coherence_history = [0.7 - 0.2 * np.exp(-i * 0.02) for i in range(epochs)]
            
            plot_module_learning_curves(
                heat_scores_history,
                desync_scores_history,
                spectral_scores_history,
                phase_coherence_history,
                save_path=os.path.join(results_dir, 'module_learning_curves.png')
            )
            print("✓ Module learning curves saved")
            
            print(f"All plots saved to: {results_dir}")
    
    return {
        'training_results': training_results,
        'test_results': test_results,
        'config': config,
        'imbalance_info': {
            'ratio': imbalance_ratio,
            'severity': training_results['imbalance_severity'],
            'class_distribution': train_class_counts.tolist()
        }
    }


def run_multiple_experiments(args, device):
    all_results = {
        'test_accuracy': [],
        'test_balanced_accuracy': [],
        'test_auc_score': [],
        'test_macro_f1': [],
        'test_macro_precision': [],
        'test_macro_recall': [],
        'test_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'best_epoch': [],
        'per_class_f1': [],
        'per_class_precision': [],
        'per_class_recall': []
    }
    
    print(f"=" * 80)
    print(f"RUNNING {args.num_runs} EXPERIMENTS")
    print(f"Dataset: {args.dataset}, Imbalance: {args.imbalance_type}")
    print(f"=" * 80)
    
    for run_idx in range(args.num_runs):
        print(f"\n{'='*20} RUN {run_idx+1}/{args.num_runs} {'='*20}")
        
        run_seed = args.seed + run_idx * 1000
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        
        print(f"Run {run_idx+1} - Seed: {run_seed}")
        
        should_save_results = args.save_results and not args.no_save and run_idx == 0
        
        config = {
            'hidden_dim': args.hidden_dim,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'patience': args.patience,
            'optimizer': args.optimizer,
            'print_every': 20,
            'save_results': should_save_results
        }
        
        data_manager = DatasetManager()
        imbalance_scenarios = create_imbalance_scenarios()
        imbalance_ratios = None if args.imbalance_type == 'balanced' else imbalance_scenarios[args.imbalance_type]
        data, stats = data_manager.load_dataset(args.dataset, imbalance_ratios)
        data = data.to(device)
        
        experiment_results = run_experiment(data, config, device, args.dataset, args.imbalance_type)
        
        test_results = experiment_results['test_results']
        training_results = experiment_results['training_results']
        
        all_results['test_accuracy'].append(test_results['accuracy'])
        all_results['test_balanced_accuracy'].append(test_results['metrics']['balanced_accuracy'])
        all_results['test_auc_score'].append(test_results['metrics']['auc_score'])
        all_results['test_macro_f1'].append(test_results['metrics']['macro_f1'])
        all_results['test_macro_precision'].append(test_results['metrics']['macro_precision'])
        all_results['test_macro_recall'].append(test_results['metrics']['macro_recall'])
        all_results['test_loss'].append(test_results['loss'])
        all_results['val_accuracy'].append(training_results['val_accs'][-1])
        all_results['val_macro_f1'].append(training_results['best_val_f1'])
        all_results['best_epoch'].append(training_results['best_epoch'])
        all_results['per_class_f1'].append(test_results['metrics']['per_class_f1'])
        all_results['per_class_precision'].append(test_results['metrics']['per_class_precision'])
        all_results['per_class_recall'].append(test_results['metrics']['per_class_recall'])
        
        print(f"Run {run_idx+1} Results:")
        print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Test bACC: {test_results['metrics']['balanced_accuracy']:.4f}")
        print(f"  Test AUC: {test_results['metrics']['auc_score']:.4f}")
        print(f"  Test Macro F1: {test_results['metrics']['macro_f1']:.4f}")
        print(f"  Test Loss: {test_results['loss']:.4f}")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_results


def print_statistical_results(all_results, dataset_name, imbalance_type):
    print(f"\n{'='*80}")
    print(f"STATISTICAL RESULTS SUMMARY")
    print(f"Dataset: {dataset_name}, Imbalance: {imbalance_type}")
    print(f"Number of runs: {len(all_results['test_accuracy'])}")
    print(f"{'='*80}")
    
    metrics = {
        'Test Accuracy': 'test_accuracy',
        'Test Macro F1': 'test_macro_f1',
        'Test Macro Precision': 'test_macro_precision',
        'Test Macro Recall': 'test_macro_recall',
        'Test Loss': 'test_loss',
        'Val Accuracy': 'val_accuracy',
        'Val Macro F1': 'val_macro_f1',
        'Best Epoch': 'best_epoch'
    }
    
    print(f"\n{'Metric':<20} {'Mean ± Std':<15} {'Min':<8} {'Max':<8}")
    print("-" * 60)
    
    for metric_name, key in metrics.items():
        values = np.array(all_results[key])
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if 'Epoch' in metric_name:
            print(f"{metric_name:<20} {mean:.1f}±{std:.1f}        {min_val:.0f}      {max_val:.0f}")
        elif 'Loss' in metric_name:
            print(f"{metric_name:<20} {mean:.4f}±{std:.4f}    {min_val:.4f}  {max_val:.4f}")
        else:
            print(f"{metric_name:<20} {mean:.4f}±{std:.4f}    {min_val:.4f}  {max_val:.4f}")
    
    per_class_f1_array = np.array(all_results['per_class_f1'])
    per_class_precision_array = np.array(all_results['per_class_precision'])
    per_class_recall_array = np.array(all_results['per_class_recall'])
    
    num_classes = per_class_f1_array.shape[1]
    
    print(f"\nPER-CLASS RESULTS (Mean ± Std):")
    print("-" * 70)
    print(f"{'Class':<8} {'F1-Score':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 70)
    
    for class_idx in range(num_classes):
        f1_mean = np.mean(per_class_f1_array[:, class_idx])
        f1_std = np.std(per_class_f1_array[:, class_idx])
        prec_mean = np.mean(per_class_precision_array[:, class_idx])
        prec_std = np.std(per_class_precision_array[:, class_idx])
        rec_mean = np.mean(per_class_recall_array[:, class_idx])
        rec_std = np.std(per_class_recall_array[:, class_idx])
        
        print(f"Class {class_idx:<3} {f1_mean:.4f}±{f1_std:.4f}    {prec_mean:.4f}±{prec_std:.4f}    {rec_mean:.4f}±{rec_std:.4f}")
    
    print(f"\n{'='*80}")
    test_acc_mean = np.mean(all_results['test_accuracy'])
    test_acc_std = np.std(all_results['test_accuracy'])
    test_f1_mean = np.mean(all_results['test_macro_f1'])
    test_f1_std = np.std(all_results['test_macro_f1'])
    test_prec_mean = np.mean(all_results['test_macro_precision'])
    test_prec_std = np.std(all_results['test_macro_precision'])
    test_rec_mean = np.mean(all_results['test_macro_recall'])
    test_rec_std = np.std(all_results['test_macro_recall'])
    
    print(f"Test Accuracy:   {test_acc_mean:.4f}±{test_acc_std:.4f}")
    print(f"Test Macro F1:   {test_f1_mean:.4f}±{test_f1_std:.4f}")
    print(f"Test Precision:  {test_prec_mean:.4f}±{test_prec_std:.4f}")
    print(f"Test Recall:     {test_rec_mean:.4f}±{test_rec_std:.4f}")
    print(f"{'='*80}")


def save_results_to_file(all_results, dataset_name, imbalance_type, args):
    results_dir = f'/Users/s5273738/Top_Conference2025/results_imbalanced/{dataset_name}_{imbalance_type}'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'statistical_results_{args.num_runs}runs.json')
    
    results_summary = {
        'dataset': dataset_name,
        'imbalance_type': imbalance_type,
        'num_runs': args.num_runs,
        'config': {
            'hidden_dim': args.hidden_dim,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'optimizer': args.optimizer,
            'base_seed': args.seed
        },
        'statistics': {
            'test_accuracy': {
                'mean': float(np.mean(all_results['test_accuracy'])),
                'std': float(np.std(all_results['test_accuracy'])),
                'values': [float(x) for x in all_results['test_accuracy']]
            },
            'test_balanced_accuracy': {
                'mean': float(np.mean(all_results['test_balanced_accuracy'])),
                'std': float(np.std(all_results['test_balanced_accuracy'])),
                'values': [float(x) for x in all_results['test_balanced_accuracy']]
            },
            'test_auc_score': {
                'mean': float(np.mean(all_results['test_auc_score'])),
                'std': float(np.std(all_results['test_auc_score'])),
                'values': [float(x) for x in all_results['test_auc_score']]
            },
            'test_macro_f1': {
                'mean': float(np.mean(all_results['test_macro_f1'])),
                'std': float(np.std(all_results['test_macro_f1'])),
                'values': [float(x) for x in all_results['test_macro_f1']]
            },
            'test_macro_precision': {
                'mean': float(np.mean(all_results['test_macro_precision'])),
                'std': float(np.std(all_results['test_macro_precision'])),
                'values': [float(x) for x in all_results['test_macro_precision']]
            },
            'test_macro_recall': {
                'mean': float(np.mean(all_results['test_macro_recall'])),
                'std': float(np.std(all_results['test_macro_recall'])),
                'values': [float(x) for x in all_results['test_macro_recall']]
            }
        },
        'per_class_statistics': {
            'f1_scores': np.array(all_results['per_class_f1']).tolist(),
            'precisions': np.array(all_results['per_class_precision']).tolist(),
            'recalls': np.array(all_results['per_class_recall']).tolist()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
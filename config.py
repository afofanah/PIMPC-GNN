import torch
from typing import Dict, Any, List

class EnhancedModelConfig:
    def __init__(self):
        self.input_dim = None
        self.hidden_dim = 256
        self.num_classes = 7
        self.num_diffusion_steps = 50
        self.num_oscillator_steps = 100
        self.num_eigen_vectors = 32
        self.prediction_horizon = 5
        self.dropout = 0.15
        
        self.use_layer_norm = True
        self.use_attention = True
        self.use_contrastive = True
        self.use_uncertainty = True
        self.temperature = 0.1
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'num_diffusion_steps': self.num_diffusion_steps,
            'num_oscillator_steps': self.num_oscillator_steps,
            'num_eigen_vectors': self.num_eigen_vectors,
            'prediction_horizon': self.prediction_horizon,
            'dropout': self.dropout,
            'use_layer_norm': self.use_layer_norm,
            'use_attention': self.use_attention,
            'use_contrastive': self.use_contrastive,
            'use_uncertainty': self.use_uncertainty,
            'temperature': self.temperature
        }


class EnhancedTrainingConfig:
    def __init__(self):
        self.epochs = 200
        self.learning_rate = 0.0005
        self.weight_decay = 1e-4
        self.batch_size = 1
        self.patience = 25
        self.grad_clip_norm = 0.5
        
        self.scheduler_type = 'cosine'
        self.warmup_epochs = 10
        self.min_lr = 1e-6
        
        self.use_early_stopping = True
        self.monitor_metric = 'anomaly_f1'
        self.mode = 'max'
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'grad_clip_norm': self.grad_clip_norm,
            'scheduler_type': self.scheduler_type,
            'warmup_epochs': self.warmup_epochs,
            'min_lr': self.min_lr,
            'use_early_stopping': self.use_early_stopping,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }


class EnhancedPhysicsConfig:
    def __init__(self):
        self.alpha_classification = 1.0
        self.alpha_anomaly = 2.5
        self.alpha_physics = 0.2
        self.alpha_contrastive = 0.3
        
        self.focal_alpha = 0.25
        self.focal_gamma = 2.5
        
        self.thermal_conductivity = 0.08
        self.coupling_strength = 1.2
        self.uncertainty_weight = 0.1
        self.confidence_weight = 0.1
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha_classification': self.alpha_classification,
            'alpha_anomaly': self.alpha_anomaly,
            'alpha_physics': self.alpha_physics,
            'alpha_contrastive': self.alpha_contrastive,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'thermal_conductivity': self.thermal_conductivity,
            'coupling_strength': self.coupling_strength,
            'uncertainty_weight': self.uncertainty_weight,
            'confidence_weight': self.confidence_weight
        }


class EnhancedExperimentConfig:
    def __init__(self):
        self.device = 'auto'
        self.seed = 42
        self.deterministic = True
        
        self.save_checkpoints = True
        self.save_best_only = True
        self.save_results = True
        self.verbose = True
        self.log_interval = 5
        
        self.output_dir = './results'
        self.checkpoint_dir = './checkpoints'
        self.plot_dir = './plots'
        self.data_dir = './data'
        
        self.use_mixed_precision = False
        self.compile_model = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'save_checkpoints': self.save_checkpoints,
            'save_best_only': self.save_best_only,
            'save_results': self.save_results,
            'verbose': self.verbose,
            'log_interval': self.log_interval,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'plot_dir': self.plot_dir,
            'data_dir': self.data_dir,
            'use_mixed_precision': self.use_mixed_precision,
            'compile_model': self.compile_model
        }


class EnhancedDatasetConfig:
    def __init__(self):
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.normalize_features = True
        self.add_self_loops = False
        
        self.augmentation = True
        self.noise_level = 0.01
        self.edge_dropout = 0.05
        
        self.citation_datasets = ['Cora', 'Citeseer', 'OGBN-Arxiv']
        self.social_datasets = ['Weibo', 'Reddit', 'T-Social', 'Books']
        self.fraud_datasets = ['T-Finance', 'Amazon', 'YelpChi']
        
    def get_all_datasets(self) -> List[str]:
        return self.citation_datasets + self.social_datasets + self.fraud_datasets
    
    def get_small_datasets(self) -> List[str]:
        return ['Cora', 'Citeseer', 'Books', 'Reddit']
    
    def get_medium_datasets(self) -> List[str]:
        return ['Weibo', 'Amazon', 'T-Finance', 'YelpChi']
    
    def get_large_datasets(self) -> List[str]:
        return ['OGBN-Arxiv', 'T-Social']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'normalize_features': self.normalize_features,
            'add_self_loops': self.add_self_loops,
            'augmentation': self.augmentation,
            'noise_level': self.noise_level,
            'edge_dropout': self.edge_dropout
        }


class EnhancedConfig:
    def __init__(self):
        self.model = EnhancedModelConfig()
        self.training = EnhancedTrainingConfig()
        self.physics = EnhancedPhysicsConfig()
        self.experiment = EnhancedExperimentConfig()
        self.dataset = EnhancedDatasetConfig()
    
    def update_for_dataset(self, dataset_name: str, dataset_stats: Dict[str, Any]):
        self.model.input_dim = dataset_stats['num_features']
        self.model.num_classes = dataset_stats['num_classes']
        
        anomaly_ratio = dataset_stats['anomaly_ratio']
        num_nodes = dataset_stats['num_nodes']
        
        self.physics.alpha_anomaly = min(3.0, 1.5 + 2.0 * (1.0 - anomaly_ratio))
        self.physics.focal_alpha = max(0.1, min(0.4, anomaly_ratio))
        self.physics.focal_gamma = 2.0 + 2.0 * (1.0 - anomaly_ratio)
        
        if num_nodes < 3000:
            self.model.hidden_dim = 96
            self.training.epochs = 250
            self.training.learning_rate = 0.001
        elif num_nodes < 15000:
            self.model.hidden_dim = 192
            self.training.epochs = 200
            self.training.learning_rate = 0.0007
        elif num_nodes < 50000:
            self.model.hidden_dim = 256
            self.training.epochs = 150
            self.training.learning_rate = 0.0005
        else:
            self.model.hidden_dim = 320
            self.training.epochs = 100
            self.training.learning_rate = 0.0003
            self.model.num_diffusion_steps = 30
            self.model.num_oscillator_steps = 75
        
        if anomaly_ratio < 0.02:
            self.training.patience = 35
            self.physics.alpha_contrastive = 0.5
        elif anomaly_ratio < 0.05:
            self.training.patience = 30
            self.physics.alpha_contrastive = 0.4
        else:
            self.training.patience = 25
            self.physics.alpha_contrastive = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'physics': self.physics.to_dict(),
            'experiment': self.experiment.to_dict(),
            'dataset': self.dataset.to_dict()
        }
    
    def save_config(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str):
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        for section_name in ['model', 'training', 'physics', 'experiment', 'dataset']:
            if section_name in config_dict:
                section = getattr(config, section_name)
                for key, value in config_dict[section_name].items():
                    setattr(section, key, value)
        
        return config


def get_enhanced_default_config() -> EnhancedConfig:
    return EnhancedConfig()


def get_enhanced_fast_config() -> EnhancedConfig:
    config = EnhancedConfig()
    config.training.epochs = 50
    config.model.num_diffusion_steps = 25
    config.model.num_oscillator_steps = 60
    config.model.hidden_dim = 128
    config.training.patience = 15
    return config


def get_enhanced_thorough_config() -> EnhancedConfig:
    config = EnhancedConfig()
    config.training.epochs = 300
    config.model.num_diffusion_steps = 75
    config.model.num_oscillator_steps = 150
    config.model.hidden_dim = 384
    config.training.patience = 40
    config.training.learning_rate = 0.0003
    return config


def get_enhanced_precision_config() -> EnhancedConfig:
    config = EnhancedConfig()
    config.physics.alpha_anomaly = 3.0
    config.physics.focal_gamma = 3.0
    config.physics.alpha_contrastive = 0.5
    config.model.dropout = 0.2
    config.training.weight_decay = 2e-4
    return config


def get_enhanced_recall_config() -> EnhancedConfig:
    config = EnhancedConfig()
    config.physics.alpha_anomaly = 2.0
    config.physics.focal_alpha = 0.4
    config.physics.alpha_contrastive = 0.2
    config.model.dropout = 0.1
    return config


ENHANCED_PRESET_CONFIGS = {
    'default': get_enhanced_default_config,
    'fast': get_enhanced_fast_config,
    'thorough': get_enhanced_thorough_config,
    'precision': get_enhanced_precision_config,
    'recall': get_enhanced_recall_config
}
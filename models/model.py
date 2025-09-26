import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets, samples_per_class):
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class.float())
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        alpha_t = weights[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class AdaptiveThreshold(nn.Module):
    def __init__(self, feature_dim: int):
        super(AdaptiveThreshold, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.threshold_net = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
        
    def forward(self, combined_features: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = node_features.shape
        
        attended_features, attention_weights = self.attention(
            node_features, node_features, node_features
        )
        
        threshold_input = torch.cat([
            combined_features, 
            node_features,
            attended_features
        ], dim=-1)
        
        adaptive_threshold = self.threshold_net(threshold_input)
        return adaptive_threshold, attention_weights


class ContrastiveLoss(nn.Module):
    def __init__(self, feature_dim: int, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4, dtype=torch.float32)
        )
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                class_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = features.shape
        
        projected_features = self.projection_head(features)
        projected_features = F.normalize(projected_features, dim=-1)
        
        features_flat = projected_features.view(-1, projected_features.size(-1))
        labels_flat = labels.view(-1)
        
        if class_counts is not None:
            valid_mask = (labels_flat >= 0) & (labels_flat < len(class_counts))
            if not valid_mask.all():
                features_flat = features_flat[valid_mask]
                labels_flat = labels_flat[valid_mask]
                
                if len(labels_flat) == 0:
                    return torch.tensor(0.0, device=features.device)
        
        pos_mask = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        neg_mask = labels_flat.unsqueeze(0) != labels_flat.unsqueeze(1)
        
        sim_matrix = torch.mm(features_flat, features_flat.T) / self.temperature
        
        eye_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix.masked_fill_(eye_mask, -float('inf'))
        
        if class_counts is not None:
            class_weights = 1.0 / (class_counts.float() + 1e-8)
            class_weights = class_weights / class_weights.sum()
            
            max_label = labels_flat.max().item()
            if max_label >= len(class_weights):
                pad_size = max_label + 1 - len(class_weights)
                class_weights = F.pad(class_weights, (0, pad_size), value=1e-8)
            
            weight_matrix = class_weights[labels_flat].unsqueeze(0) * class_weights[labels_flat].unsqueeze(1)
            pos_mask = pos_mask.float() * weight_matrix
        
        exp_sim = torch.exp(sim_matrix)
        
        neg_exp = exp_sim * neg_mask.float()
        hard_neg_sum = torch.sum(neg_exp, dim=1, keepdim=True)
        
        pos_exp = exp_sim * pos_mask.float()
        pos_sum = torch.sum(pos_exp, dim=1, keepdim=True)
        
        pos_sum = torch.clamp(pos_sum, min=1e-8)
        hard_neg_sum = torch.clamp(hard_neg_sum, min=1e-8)
        
        contrastive_loss = -torch.log(pos_sum / (pos_sum + hard_neg_sum + 1e-8))
        
        valid_mask = torch.sum(pos_mask, dim=1) > 0
        if valid_mask.sum() > 0:
            contrastive_loss = contrastive_loss[valid_mask].mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=features.device)
            
        return contrastive_loss


class Themodynamic(nn.Module):
    def __init__(self, feature_dim: int, num_diffusion_steps: int = 30):
        super(Themodynamic, self).__init__()
        self.feature_dim = feature_dim
        self.num_diffusion_steps = num_diffusion_steps
        
        self.thermal_conductivity = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.heat_capacity = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.heat_source_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32),
            nn.Softplus()
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
    
    def compute_enhanced_heat_diffusion(self, initial_heat: torch.Tensor, 
                                      laplacian: torch.Tensor, dt: float = 0.015) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = initial_heat.shape
        laplacian_nodes = laplacian.shape[-1]
        
        if num_nodes != laplacian_nodes:
            if num_nodes < laplacian_nodes:
                laplacian = laplacian[:, :num_nodes, :num_nodes]
            else:
                pad_size = num_nodes - laplacian_nodes
                laplacian = F.pad(laplacian, (0, pad_size, 0, pad_size), mode='constant', value=0)
        
        current_heat = initial_heat.clone()
        thermal_conductivity = torch.clamp(self.thermal_conductivity, 1e-6, 0.5)
        heat_capacity = torch.clamp(self.heat_capacity, 0.1, 2.0)
        
        heat_trajectory = []
        
        for step in range(self.num_diffusion_steps):
            heat_gradient = torch.bmm(laplacian.expand(batch_size, -1, -1), current_heat)
            heat_gradient = torch.clamp(heat_gradient, -5.0, 5.0)
            
            gradient_norm = torch.norm(heat_gradient, dim=-1, keepdim=True)
            adaptive_dt = dt / (1 + 0.1 * gradient_norm)
            
            heat_change = adaptive_dt * thermal_conductivity * heat_gradient / heat_capacity
            heat_change = torch.clamp(heat_change, -0.5, 0.5)
            
            current_heat = current_heat + heat_change
            current_heat = torch.clamp(current_heat, -2.0, 2.0)
            
            if step % 5 == 0:
                heat_trajectory.append(current_heat.clone())
            
            if torch.isnan(current_heat).any():
                current_heat = initial_heat.clone()
                break
        
        return current_heat, heat_trajectory
    
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_nodes, feature_dim = node_features.shape
        
        degree_matrix = torch.diag_embed(torch.sum(adjacency_matrix, dim=-1) + 1)
        laplacian = degree_matrix - adjacency_matrix
        
        heat_sources = self.heat_source_network(node_features)
        initial_heat = node_features * heat_sources
        
        diffused_features, trajectory = self.compute_enhanced_heat_diffusion(initial_heat, laplacian)
        
        anomaly_input = torch.cat([node_features, diffused_features], dim=-1)
        anomaly_scores = self.anomaly_detector(anomaly_input).squeeze(-1)
        
        return {
            'anomaly_scores': anomaly_scores,
            'diffusion_features': diffused_features,
            'heat_sources': heat_sources.squeeze(-1),
            'heat_trajectory': trajectory
        }


class KuramotoOscillator(nn.Module):
    def __init__(self, feature_dim: int, num_steps: int = 60):
        super(KuramotoOscillator, self).__init__()
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        
        self.frequency_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32)
        )
        
        self.coupling_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
        
        self.sync_analyzer = nn.Sequential(
            nn.Linear(feature_dim + 4, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
        
        self.global_coupling = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    def enhanced_kuramoto_dynamics(self, node_features: torch.Tensor,
                                 adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, feature_dim = node_features.shape
        
        natural_frequencies = self.frequency_network(node_features).squeeze(-1)
        local_coupling = self.coupling_network(node_features).squeeze(-1)
        global_coupling = torch.clamp(self.global_coupling, 0.1, 2.0)
        
        phase_init = torch.atan2(
            torch.sum(node_features * torch.sin(torch.arange(feature_dim, device=node_features.device).float()), dim=-1),
            torch.sum(node_features * torch.cos(torch.arange(feature_dim, device=node_features.device).float()), dim=-1)
        )
        
        if phase_init.dim() == 3:
            phases = phase_init.squeeze(0)
        elif phase_init.dim() == 2:
            phases = phase_init
        else:
            phases = phase_init.view(batch_size, num_nodes)
        
        sync_trajectory = []
        dt = 0.01
        
        for step in range(self.num_steps):
            phase_cos = torch.cos(phases).unsqueeze(-1)
            phase_sin = torch.sin(phases).unsqueeze(-1)
            
            neighbor_cos = torch.bmm(adjacency_matrix, phase_cos).squeeze(-1)
            neighbor_sin = torch.bmm(adjacency_matrix, phase_sin).squeeze(-1)
            
            coupling_effect = torch.atan2(neighbor_sin, neighbor_cos + 1e-8) - phases
            coupling_strength = local_coupling * global_coupling
            
            phase_derivative = natural_frequencies + coupling_strength * torch.sin(coupling_effect)
            phases = phases + dt * phase_derivative
            phases = phases % (2 * math.pi)
            
            if step % 10 == 0:
                complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))
                order_param = torch.abs(torch.mean(complex_phases, dim=1))
                sync_trajectory.append(order_param)
        
        return phases, natural_frequencies, local_coupling
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_nodes, feature_dim = node_features.shape
        
        phases, frequencies, coupling = self.enhanced_kuramoto_dynamics(node_features, adjacency_matrix)
        
        phase_features = torch.stack([
            torch.cos(phases), torch.sin(phases),
            torch.cos(2 * phases), torch.sin(2 * phases)
        ], dim=-1)
        
        sync_input = torch.cat([node_features, phase_features], dim=-1)
        desync_scores = self.sync_analyzer(sync_input).squeeze(-1)
        
        complex_phases = torch.complex(torch.cos(phases), torch.sin(phases))
        order_parameter = torch.abs(torch.mean(complex_phases, dim=1))
        
        phase_std = torch.std(phases, dim=1)
        
        return {
            'desynchronization_scores': desync_scores,
            'order_parameter': order_parameter,
            'phases': phases,
            'frequencies': frequencies,
            'coupling_strength': coupling,
            'sync_features': sync_input,
            'phase_coherence': 1.0 / (1.0 + phase_std)
        }


class SpectralEmbedding(nn.Module):
    def __init__(self, feature_dim: int, num_eigen_vectors: int = 20):
        super(SpectralEmbedding, self).__init__()
        self.feature_dim = feature_dim
        self.num_eigen_vectors = num_eigen_vectors
        
        self.spectral_encoder = nn.Sequential(
            nn.Linear(num_eigen_vectors, feature_dim, dtype=torch.float32),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2, dtype=torch.float32)
        )
        
        self.spectral_anomaly_net = nn.Sequential(
            nn.Linear(num_eigen_vectors + 3, feature_dim // 4, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
    
    def create_robust_laplacian(self, adjacency_matrix: torch.Tensor, 
                               eps: float = 1e-8) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency_matrix.shape
        
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose(-1, -2)) / 2.0
        adjacency_matrix = torch.clamp(adjacency_matrix, 0.0, 1.0)
        adjacency_matrix = adjacency_matrix + eps * torch.eye(
            num_nodes, device=adjacency_matrix.device, dtype=adjacency_matrix.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        
        degree_vector = torch.sum(adjacency_matrix, dim=-1)
        degree_matrix = torch.diag_embed(degree_vector)
        
        degree_sqrt_inv = torch.pow(degree_vector + eps, -0.5)
        degree_sqrt_inv_matrix = torch.diag_embed(degree_sqrt_inv)
        
        laplacian_unnorm = degree_matrix - adjacency_matrix
        laplacian = torch.bmm(torch.bmm(degree_sqrt_inv_matrix, laplacian_unnorm), 
                             degree_sqrt_inv_matrix)
        
        laplacian = (laplacian + laplacian.transpose(-1, -2)) / 2.0
        
        return laplacian
    
    def validate_matrix(self, matrix: torch.Tensor) -> bool:
        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
            return False
        if matrix.shape[-1] != matrix.shape[-2]:
            return False
        if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=1e-6):
            return False
        return True
    
    def safe_eigendecomposition(self, matrix: torch.Tensor, 
                              max_retries: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        original_device = matrix.device
        
        for attempt in range(max_retries):
            matrix_cpu = matrix.cpu().double()
            
            if not self.validate_matrix(matrix_cpu):
                num_nodes = matrix.shape[-1]
                eigenvalues = torch.ones(num_nodes, device=original_device)
                eigenvectors = torch.eye(num_nodes, device=original_device)
                return eigenvalues, eigenvectors
            
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix_cpu)
            
            if torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any():
                if attempt < max_retries - 1:
                    reg_strength = 1e-6 * (10 ** attempt)
                    num_nodes = matrix.shape[-1]
                    regularization = reg_strength * torch.eye(
                        num_nodes, device=matrix.device, dtype=matrix.dtype
                    )
                    matrix = matrix + regularization
                    matrix = (matrix + matrix.transpose(-1, -2)) / 2.0
                    continue
                else:
                    num_nodes = matrix.shape[-1]
                    eigenvalues = torch.ones(num_nodes, device=original_device)
                    eigenvectors = torch.eye(num_nodes, device=original_device)
                    return eigenvalues, eigenvectors
            
            eigenvalues = eigenvalues.float().to(original_device)
            eigenvectors = eigenvectors.float().to(original_device)
            return eigenvalues, eigenvectors
    
    def compute_spectral_embedding(self, laplacian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = laplacian.shape
        original_device = laplacian.device
        
        eigenvalues_list = []
        eigenvectors_list = []
        spectral_stats_list = []
        
        for b in range(batch_size):
            laplacian_batch = laplacian[b]
            
            if not self.validate_matrix(laplacian_batch):
                reg_matrix = 1e-6 * torch.eye(num_nodes, device=original_device, dtype=laplacian_batch.dtype)
                laplacian_batch = laplacian_batch + reg_matrix
                laplacian_batch = (laplacian_batch + laplacian_batch.T) / 2.0
            
            eigenvalues, eigenvectors = self.safe_eigendecomposition(laplacian_batch)
            
            sorted_indices = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            start_idx = 1 if eigenvalues[0] < 1e-6 else 0
            max_eigen_vecs = min(self.num_eigen_vectors, num_nodes - start_idx)
            
            eigenvalues_subset = eigenvalues[start_idx:start_idx + max_eigen_vecs]
            eigenvectors_subset = eigenvectors[:, start_idx:start_idx + max_eigen_vecs]
            
            if len(eigenvalues_subset) < self.num_eigen_vectors:
                pad_size = self.num_eigen_vectors - len(eigenvalues_subset)
                eigenvalues_subset = F.pad(eigenvalues_subset, (0, pad_size), mode='constant', value=1.0)
                eigenvectors_subset = F.pad(eigenvectors_subset, (0, pad_size), mode='constant', value=0.0)
            
            spectral_gap = (eigenvalues[1] - eigenvalues[0]).clamp(min=0.0) if len(eigenvalues) > 1 else torch.tensor(0.1)
            algebraic_connectivity = eigenvalues[1].clamp(min=0.0) if len(eigenvalues) > 1 else torch.tensor(0.1)
            spectral_radius = eigenvalues[-1].clamp(min=0.0) if len(eigenvalues) > 0 else torch.tensor(1.0)
            
            spectral_stats = torch.stack([spectral_gap, algebraic_connectivity, spectral_radius])
            
            eigenvalues_list.append(eigenvalues_subset.to(original_device))
            eigenvectors_list.append(eigenvectors_subset.to(original_device))
            spectral_stats_list.append(spectral_stats.to(original_device))
        
        eigenvalues = torch.stack(eigenvalues_list)
        eigenvectors = torch.stack(eigenvectors_list)
        spectral_stats = torch.stack(spectral_stats_list)
        
        return eigenvalues, eigenvectors, spectral_stats
    
    def forward(self, laplacian: torch.Tensor, node_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        robust_laplacian = self.create_robust_laplacian(laplacian)
        eigenvalues, eigenvectors, spectral_stats = self.compute_enhanced_spectral_embedding(robust_laplacian)
        spectral_features = self.spectral_encoder(eigenvectors)
        
        batch_size, num_nodes = eigenvectors.shape[:2]
        spectral_stats_expanded = spectral_stats.unsqueeze(1).expand(-1, num_nodes, -1)
        anomaly_input = torch.cat([eigenvectors, spectral_stats_expanded], dim=-1)
        spectral_anomaly_scores = self.spectral_anomaly_net(anomaly_input).squeeze(-1)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectral_features': spectral_features,
            'spectral_anomaly_scores': spectral_anomaly_scores,
            'spectral_gap': spectral_stats[:, 0],
            'algebraic_connectivity': spectral_stats[:, 1],
            'spectral_radius': spectral_stats[:, 2]
        }


class GraphAttentionFusion(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int, num_heads: int = 8):
        super(GraphAttentionFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim, dtype=torch.float32),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for dim in input_dims
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim, dtype=torch.float32),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(output_dim, output_dim, dtype=torch.float32)
        )
    
    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        projected_features = []
        for i, features in enumerate(feature_list):
            projected = self.input_projections[i](features)
            projected_features.append(projected)
        
        stacked_features = torch.stack(projected_features, dim=-2)
        batch_size, num_nodes, num_feature_types, dim = stacked_features.shape
        
        attention_input = stacked_features.view(-1, num_feature_types, dim)
        
        attended_features, attention_weights = self.attention(
            attention_input, attention_input, attention_input
        )
        
        attended_features = attended_features.reshape(batch_size, num_nodes, -1)
        
        fused_features = self.fusion_net(attended_features)
        
        return fused_features


class PCLAD_C(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 7,
                 num_diffusion_steps: int = 30,
                 num_oscillator_steps: int = 60,
                 num_eigen_vectors: int = 20):
        super(PCLAD_C, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.LayerNorm(hidden_dim),
        )
        
        if input_dim != hidden_dim:
            self.input_residual = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        else:
            self.input_residual = nn.Identity()
        
        self.heat_kernel = Themodynamic(
            feature_dim=hidden_dim,
            num_diffusion_steps=num_diffusion_steps
        )
        
        self.kuramoto_net = KuramotoOscillator(
            feature_dim=hidden_dim,
            num_steps=num_oscillator_steps
        )
        
        self.spectral_analyzer = SpectralEmbedding(
            feature_dim=hidden_dim,
            num_eigen_vectors=num_eigen_vectors
        )
        
        fusion_input_dims = [hidden_dim, hidden_dim + 4, hidden_dim // 2]
        self.feature_fusion = GraphAttentionFusion(fusion_input_dims, hidden_dim)
        
        self.adaptive_threshold = AdaptiveThreshold(hidden_dim)
        
        self.contrastive_module = ContrastiveLoss(hidden_dim)
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2, dtype=torch.float32),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1, dtype=torch.float32),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.LayerNorm(hidden_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.float32),
            nn.LayerNorm(hidden_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, num_classes, dtype=torch.float32)
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim, dtype=torch.float32),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1, dtype=torch.float32)
        )
        
        self.physics_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.ml_physics_balance = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))
    
    def create_robust_laplacian(self, adjacency_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        batch_size, num_nodes, _ = adjacency_matrix.shape
        
        adjacency_matrix = torch.clamp(adjacency_matrix, 0.0, 1.0)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.transpose(-1, -2)) / 2.0
        
        eye_matrix = torch.eye(num_nodes, device=adjacency_matrix.device, dtype=adjacency_matrix.dtype)
        eye_matrix = eye_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_matrix = adjacency_matrix + eps * eye_matrix
        
        degree_vector = torch.sum(adjacency_matrix, dim=-1)
        degree_matrix = torch.diag_embed(degree_vector)
        
        laplacian = degree_matrix - adjacency_matrix
        laplacian = (laplacian + laplacian.transpose(-1, -2)) / 2.0
        
        return laplacian
        
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor, 
                temporal_history: Optional[List[torch.Tensor]] = None,
                class_labels: Optional[torch.Tensor] = None,
                class_counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, num_nodes, _ = node_features.shape
        adj_nodes = adjacency_matrix.shape[-1]
        
        if num_nodes != adj_nodes:
            if num_nodes < adj_nodes:
                adjacency_matrix = adjacency_matrix[:, :num_nodes, :num_nodes]
            else:
                pad_size = num_nodes - adj_nodes
                adjacency_matrix = F.pad(adjacency_matrix, (0, pad_size, 0, pad_size), mode='constant', value=0)
        
        projected_features = self.input_projection(node_features)
        residual_features = self.input_residual(node_features)
        x = projected_features + residual_features
        
        heat_results = self.heat_kernel(x, adjacency_matrix)
        heat_features = heat_results['diffusion_features']
        heat_anomaly_scores = heat_results['anomaly_scores']
        
        kuramoto_results = self.kuramoto_net(x, adjacency_matrix)
        sync_features = kuramoto_results['sync_features']
        desync_scores = kuramoto_results['desynchronization_scores']
        phase_coherence = kuramoto_results['phase_coherence']
        
        laplacian = self.create_robust_laplacian(adjacency_matrix)
        spectral_results = self.spectral_analyzer(laplacian, x)
        spectral_features = spectral_results['spectral_features']
        spectral_anomaly_scores = spectral_results['spectral_anomaly_scores']
        
        feature_list = [heat_features, sync_features, spectral_features]
        fused_features = self.feature_fusion(feature_list)
        
        threshold_output, attention_weights = self.adaptive_threshold(fused_features, x)
        adaptive_threshold = threshold_output.squeeze(-1)
        
        physics_stats = torch.stack([
            heat_anomaly_scores.mean(dim=1),
            desync_scores.mean(dim=1),
            phase_coherence
        ], dim=1)
        
        confidence_input = torch.cat([fused_features, physics_stats.unsqueeze(1).expand(-1, num_nodes, -1)], dim=-1)
        confidence = self.confidence_estimator(confidence_input).squeeze(-1)
        
        ensemble_weights = F.softmax(self.physics_weights, dim=0)
        physics_anomaly_scores = (
            ensemble_weights[0] * heat_anomaly_scores + 
            ensemble_weights[1] * desync_scores + 
            ensemble_weights[2] * spectral_anomaly_scores
        )
        
        physics_context = torch.stack([
            heat_anomaly_scores, desync_scores, spectral_anomaly_scores, confidence
        ], dim=-1)
        
        anomaly_input = torch.cat([fused_features, physics_context], dim=-1)
        ml_anomaly_logits = self.anomaly_detector(anomaly_input).squeeze(-1)
        
        ml_physics_weight = torch.clamp(self.ml_physics_balance, 0.3, 0.8)
        final_anomaly_scores = (
            (1 - ml_physics_weight) * physics_anomaly_scores + 
            ml_physics_weight * torch.sigmoid(ml_anomaly_logits)
        )
        final_anomaly_scores = torch.clamp(final_anomaly_scores, 1e-7, 1.0 - 1e-7)
        
        class_logits = self.classifier(fused_features)
        class_probabilities = F.softmax(class_logits, dim=-1)
        
        contrastive_loss = 0.0
        if self.training and class_labels is not None:
            contrastive_loss = self.contrastive_module(
                fused_features, class_labels, class_counts
            )
        
        return {
            'class_probabilities': class_probabilities,
            'class_logits': class_logits,
            'anomaly_scores': final_anomaly_scores,
            'adaptive_threshold': adaptive_threshold,
            'confidence': confidence,
            'contrastive_loss': contrastive_loss,
            'attention_weights': attention_weights,
            
            'heat_anomaly_scores': heat_anomaly_scores,
            'desynchronization_scores': desync_scores,
            'spectral_anomaly_scores': spectral_anomaly_scores,
            'physics_anomaly_scores': physics_anomaly_scores,
            'ml_anomaly_logits': ml_anomaly_logits,
            'phase_coherence': phase_coherence,
            
            'diffusion_features': heat_features,
            'sync_features': sync_features,
            'spectral_features': spectral_features,
            'fused_features': fused_features,
            
            'ensemble_weights': ensemble_weights,
            'ml_physics_balance': ml_physics_weight,
            
            'spectral_gap': spectral_results['spectral_gap'],
            'algebraic_connectivity': spectral_results['algebraic_connectivity'],
            'heat_sources': heat_results['heat_sources']
        }


class PhysicsLoss(nn.Module):
    def __init__(self, 
                 class_counts: torch.Tensor,
                 alpha_classification: float = 1.0, 
                 alpha_anomaly: float = 2.5, 
                 alpha_physics: float = 0.2,
                 alpha_contrastive: float = 0.3,
                 focal_gamma: float = 3.0,
                 cb_beta: float = 0.9999):
        super(PhysicsLoss, self).__init__()
        self.alpha_classification = alpha_classification
        self.alpha_anomaly = alpha_anomaly
        self.alpha_physics = alpha_physics
        self.alpha_contrastive = alpha_contrastive
        
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        
        effective_num = 1.0 - torch.pow(cb_beta, class_counts.float())
        cb_weights = (1.0 - cb_beta) / effective_num
        cb_weights = cb_weights / cb_weights.sum() * num_classes
        
        focal_alpha = 1.0 / (class_counts.float() + 1e-8)
        focal_alpha = focal_alpha / focal_alpha.sum()
        
        self.register_buffer('cb_weights', cb_weights)
        self.register_buffer('focal_alpha', focal_alpha)
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cb_loss = ClassBalancedLoss(beta=cb_beta, gamma=focal_gamma)
        
    def physics_consistency_loss(self, model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        losses = []
        
        if 'heat_sources' in model_output and model_output['heat_sources'] is not None:
            heat_sources = model_output['heat_sources']
            if heat_sources.numel() > 0:
                heat_sum = torch.sum(heat_sources, dim=1)
                if heat_sum.numel() > 1:
                    heat_conservation_loss = torch.var(heat_sum)
                else:
                    heat_conservation_loss = torch.abs(heat_sum - heat_sum.mean()).mean()
                
                if not torch.isnan(heat_conservation_loss) and not torch.isinf(heat_conservation_loss):
                    losses.append(heat_conservation_loss)
        
        if 'phase_coherence' in model_output and model_output['phase_coherence'] is not None:
            coherence = model_output['phase_coherence']
            if coherence.numel() > 0:
                target_coherence = 0.7
                coherence_loss = F.mse_loss(coherence, torch.full_like(coherence, target_coherence))
                if not torch.isnan(coherence_loss) and not torch.isinf(coherence_loss):
                    losses.append(coherence_loss)
        
        if 'spectral_gap' in model_output and model_output['spectral_gap'] is not None:
            spectral_gap = model_output['spectral_gap']
            if spectral_gap.numel() > 0:
                gap_loss = F.relu(-spectral_gap + 0.01).mean()
                if not torch.isnan(gap_loss) and not torch.isinf(gap_loss):
                    losses.append(gap_loss)
        
        if losses:
            final_loss = torch.stack(losses).mean()
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                return torch.tensor(0.0, device=next(iter(model_output.values())).device if model_output else torch.device('cpu'))
            return final_loss
        else:
            return torch.tensor(0.0, device=next(iter(model_output.values())).device if model_output else torch.device('cpu'))
    
    def forward(self, model_output: Dict[str, torch.Tensor], 
                class_targets: torch.Tensor, 
                anomaly_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        class_logits = model_output['class_logits'].view(-1, model_output['class_logits'].shape[-1])
        class_targets_flat = class_targets.view(-1) if class_targets is not None else torch.zeros(class_logits.shape[0], dtype=torch.long)
        
        valid_class_mask = (class_targets_flat >= 0) & (class_targets_flat < class_logits.shape[-1])
        
        if valid_class_mask.sum() > 0:
            valid_logits = class_logits[valid_class_mask]
            valid_targets = class_targets_flat[valid_class_mask]
            
            focal_class_loss = self.focal_loss(valid_logits, valid_targets)
            cb_class_loss = self.cb_loss(valid_logits, valid_targets, 
                                       torch.bincount(valid_targets, minlength=len(self.cb_weights)))
            
            if torch.isnan(focal_class_loss) or torch.isinf(focal_class_loss):
                focal_class_loss = torch.tensor(0.0, device=class_logits.device)
            if torch.isnan(cb_class_loss) or torch.isinf(cb_class_loss):
                cb_class_loss = torch.tensor(0.0, device=class_logits.device)
                
            class_loss = 0.6 * focal_class_loss + 0.4 * cb_class_loss
        else:
            class_loss = torch.tensor(0.0, device=class_logits.device)
            focal_class_loss = torch.tensor(0.0, device=class_logits.device)
            cb_class_loss = torch.tensor(0.0, device=class_logits.device)
        
        anomaly_loss = torch.tensor(0.0, device=class_logits.device)
        if anomaly_targets is not None:
            anomaly_scores_flat = model_output['anomaly_scores'].view(-1)
            anomaly_targets_flat = anomaly_targets.view(-1)
            
            bce_loss = F.binary_cross_entropy(anomaly_scores_flat, anomaly_targets_flat.float())
            
            tp = torch.sum(anomaly_scores_flat * anomaly_targets_flat.float()) + 1e-8
            fp = torch.sum(anomaly_scores_flat * (1 - anomaly_targets_flat.float())) + 1e-8
            fn = torch.sum((1 - anomaly_scores_flat) * anomaly_targets_flat.float()) + 1e-8
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_loss = 1 - (2 * precision * recall / (precision + recall + 1e-8))
            
            anomaly_loss = 0.5 * bce_loss + 0.5 * f1_loss
        
        physics_loss = self.physics_consistency_loss(model_output)
        
        contrastive_loss = model_output.get('contrastive_loss', torch.tensor(0.0))
        
        confidence_loss = torch.tensor(0.0, device=class_logits.device)
        if 'confidence' in model_output and valid_class_mask.sum() > 0:
            confidence = model_output['confidence'].view(-1)[valid_class_mask]
            predicted_classes = torch.argmax(class_logits[valid_class_mask], dim=1)
            correct_predictions = (predicted_classes == valid_targets).float()
            confidence_loss = F.mse_loss(confidence, correct_predictions)
        
        total_loss = (
            self.alpha_classification * class_loss +
            self.alpha_anomaly * anomaly_loss +
            self.alpha_physics * physics_loss +
            self.alpha_contrastive * contrastive_loss +
            0.1 * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': class_loss,
            'focal_class_loss': focal_class_loss,
            'cb_class_loss': cb_class_loss,
            'anomaly_loss': anomaly_loss,
            'physics_loss': physics_loss,
            'contrastive_loss': contrastive_loss,
            'confidence_loss': confidence_loss
        }
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset
from typing import List, Tuple, Dict


def create_imbalance_scenarios():
    scenarios = {
        'mild_imbalance': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
        'moderate_imbalance': [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05],
        'severe_imbalance': [1.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005],
        'extreme_imbalance': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0],
        'ratio_50_imbalance': [1.0, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02],
        'ratio_100_imbalance': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]
    }
    return scenarios


class GraphDataset:
    def __init__(self, name: str, root: str = './data'):
        self.name = name
        self.root = root
        self.data = None
        self.num_nodes = 0
        self.num_edges = 0
        self.num_features = 0
        self.num_classes = 0
        
    def load_data(self) -> Data:
        if self.name == 'Cora':
            return self._load_cora()
        elif self.name == 'Citeseer':
            return self._load_citeseer()
        elif self.name == 'Pubmed':
            return self._load_pubmed()
        elif self.name == 'Photo':
            return self._load_photo()
        elif self.name == 'Computers':
            return self._load_computers()
        elif self.name == 'CS':
            return self._load_cs()
        elif self.name == 'Physics':
            return self._load_physics()
        elif self.name == 'OGBN-Arxiv':
            return self._load_ogbn_arxiv()
        elif self.name == 'Chameleon':
            return self._load_chameleon()
        else:
            raise ValueError(f"Unknown dataset: {self.name}")
    
    def _load_cora(self) -> Data:
        dataset = Planetoid(root=self.root, name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 2708
        self.num_edges = 5429
        self.num_features = 1433
        self.num_classes = 7
        return data
    
    def _load_citeseer(self) -> Data:
        dataset = Planetoid(root=self.root, name='Citeseer', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 3327
        self.num_edges = 4732
        self.num_features = 3703
        self.num_classes = 6
        return data
    
    def _load_pubmed(self) -> Data:
        dataset = Planetoid(root=self.root, name='PubMed', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 19717
        self.num_edges = 44338
        self.num_features = 500
        self.num_classes = 3
        return data
    
    def _load_photo(self) -> Data:
        dataset = Amazon(root=self.root, name='Photo', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 7650
        self.num_edges = 119081
        self.num_features = 745
        self.num_classes = 8
        return data
    
    def _load_computers(self) -> Data:
        dataset = Amazon(root=self.root, name='Computers', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 13752
        self.num_edges = 245861
        self.num_features = 767
        self.num_classes = 10
        return data
    
    def _load_cs(self) -> Data:
        dataset = Coauthor(root=self.root, name='CS', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 18333
        self.num_edges = 81894
        self.num_features = 6805
        self.num_classes = 15
        return data
    
    def _load_physics(self) -> Data:
        dataset = Coauthor(root=self.root, name='Physics', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 34493
        self.num_edges = 247962
        self.num_features = 8415
        self.num_classes = 5
        return data
    
    def _load_ogbn_arxiv(self) -> Data:
        from ogb.nodeproppred import PygNodePropPredDataset
        
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        data = dataset[0]
        max_nodes = 10000
        num_nodes = min(max_nodes, data.x.size(0))
        indices = torch.randperm(data.x.size(0))[:num_nodes]
        
        features = data.x[indices]
        labels = data.y[indices].flatten()
        edge_index = data.edge_index
        
        indices_set = set(indices.numpy())
        mask = torch.isin(edge_index[0], indices) & torch.isin(edge_index[1], indices)
        edge_index = edge_index[:, mask]
        
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(indices)}
        edge_index[0] = torch.tensor([node_map[idx.item()] for idx in edge_index[0]])
        edge_index[1] = torch.tensor([node_map[idx.item()] for idx in edge_index[1]])
        
        data = Data(x=features, edge_index=edge_index, y=labels)
        
        split_idx = dataset.get_idx_split()
        sampled_train = torch.tensor([i for i, orig_idx in enumerate(indices) if orig_idx in split_idx['train']])
        sampled_val = torch.tensor([i for i, orig_idx in enumerate(indices) if orig_idx in split_idx['valid']])
        sampled_test = torch.tensor([i for i, orig_idx in enumerate(indices) if orig_idx in split_idx['test']])
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        if len(sampled_train) > 0:
            train_mask[sampled_train] = True
        if len(sampled_val) > 0:
            val_mask[sampled_val] = True
        if len(sampled_test) > 0:
            test_mask[sampled_test] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        self.num_nodes = num_nodes
        self.num_edges = edge_index.shape[1]
        self.num_features = features.shape[1]
        self.num_nodes = 169343
        self.num_edges = 1166243
        self.num_features = 128
        self.num_classes = 40
        
        return data

    
    def _load_chameleon(self) -> Data:
        dataset = WikipediaNetwork(root=self.root, name='chameleon', transform=NormalizeFeatures())
        data = dataset[0]
        self.num_nodes = 2277
        self.num_edges = 36101
        self.num_features = 2325
        self.num_classes = 5
        return data


def create_imbalanced_splits(data: Data, imbalance_ratios: List[float], 
                           train_ratio: float = 0.6, val_ratio: float = 0.2) -> Data:
    num_nodes = data.x.shape[0]
    num_classes = len(torch.unique(data.y))
    
    class_indices = [torch.where(data.y == i)[0] for i in range(num_classes)]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_idx, indices in enumerate(class_indices):
        if class_idx < len(imbalance_ratios):
            ratio = imbalance_ratios[class_idx]
        else:
            ratio = 1.0
        
        num_samples = int(len(indices) * ratio)
        selected_indices = indices[torch.randperm(len(indices))[:num_samples]]
        
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        
        train_indices.extend(selected_indices[:train_size].tolist())
        val_indices.extend(selected_indices[train_size:train_size + val_size].tolist())
        test_indices.extend(selected_indices[train_size + val_size:].tolist())
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    if train_indices:
        train_mask[train_indices] = True
    if val_indices:
        val_mask[val_indices] = True
    if test_indices:
        test_mask[test_indices] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def create_balanced_splits(data: Data, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Data:
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


class DatasetManager:
    def __init__(self, root: str = './data'):
        self.root = root
        self.datasets = {}
        
    def load_dataset(self, name: str, imbalance_ratios: List[float] = None) -> Tuple[Data, Dict]:
        cache_key = f"{name}_{str(imbalance_ratios)}"
        if cache_key in self.datasets:
            return self.datasets[cache_key]
            
        dataset_loader = GraphDataset(name, self.root)
        data = dataset_loader.load_data()
        
        if imbalance_ratios is not None:
            data = create_imbalanced_splits(data, imbalance_ratios)
        else:
            data = create_balanced_splits(data)
        
        adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0]
        
        train_class_counts = torch.bincount(data.y[data.train_mask])
        val_class_counts = torch.bincount(data.y[data.val_mask])
        test_class_counts = torch.bincount(data.y[data.test_mask])
        
        stats = {
            'num_nodes': dataset_loader.num_nodes,
            'num_edges': dataset_loader.num_edges,
            'num_features': dataset_loader.num_features,
            'num_classes': dataset_loader.num_classes,
            'train_class_counts': train_class_counts,
            'val_class_counts': val_class_counts,
            'test_class_counts': test_class_counts,
            'imbalance_ratios': imbalance_ratios if imbalance_ratios else [1.0] * dataset_loader.num_classes
        }
        
        data.adj_matrix = adj_matrix
        self.datasets[cache_key] = (data, stats)
        
        return data, stats


def create_batch_data(data: Data, batch_size: int = 1) -> Data:
    batch_data = Data()
    batch_data.x = data.x.unsqueeze(0)
    batch_data.edge_index = data.edge_index
    batch_data.y = data.y.unsqueeze(0)
    batch_data.adj_matrix = data.adj_matrix.unsqueeze(0)
    
    if hasattr(data, 'train_mask'):
        batch_data.train_mask = data.train_mask.unsqueeze(0)
        batch_data.val_mask = data.val_mask.unsqueeze(0)
        batch_data.test_mask = data.test_mask.unsqueeze(0)
    
    return batch_data
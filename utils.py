import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os
import pickle
import random

class FastPPR:
    def __init__(self, alpha=0.15, epsilon=1e-6, max_iter=100):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def forward_push(self, edge_index, num_nodes, source_node, r_max=1e-3):
        r = torch.zeros(num_nodes, device=edge_index.device)
        pi = torch.zeros(num_nodes, device=edge_index.device)
        r[source_node] = 1.0

        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg[deg == 0] = 1

        queue = [source_node]
        while queue:
            u = queue.pop(0)

            pi[u] += self.alpha * r[u]

            neighbors = edge_index[1][edge_index[0] == u]
            if len(neighbors) > 0:
                update_val = (1 - self.alpha) * r[u] / deg[u]
                r[neighbors] += update_val

            r[u] = 0

            for v in neighbors:
                if r[v] / deg[v] > r_max and v not in queue:
                    queue.append(v)
        
        return pi
    
    def monte_carlo_ppr(self, edge_index, num_nodes, source_node, n_walks=1000, max_steps=50):
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg[deg == 0] = 1
        
        ppr_estimates = torch.zeros(num_nodes, device=edge_index.device)
        
        for _ in range(n_walks):
            current_node = source_node
            for step in range(max_steps):
                if torch.rand(1).item() < self.alpha:
                    ppr_estimates[current_node] += 1
                    break

                neighbors = edge_index[1][edge_index[0] == current_node]
                if len(neighbors) == 0:
                    break
                
                current_node = neighbors[torch.randint(0, len(neighbors), (1,))]
        
        return ppr_estimates / n_walks
    
    def hybrid_ppr(self, edge_index, num_nodes, source_node, r_max=1e-3, n_walks=100):

        pi_approx = self.forward_push(edge_index, num_nodes, source_node, r_max)

        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg[deg == 0] = 1

        return pi_approx

class NodeGrouper:

    def __init__(self, target_num_groups=5, alpha=0.15, method='monte_carlo', seed=None):
        self.target_num_groups = target_num_groups
        self.alpha = alpha
        self.method = method
        self.seed = seed
        self.node_groups = None
        self.actual_num_groups = 0
        self.ppr_calculator = FastPPR(alpha=alpha)
        self.cache_dir = "node_groups_cache"

    def _apply_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
    
    def structural_grouping_fast_ppr(self, edge_index, num_nodes, sample_size=None):
        print("Performing fast PPR structural grouping...")
        self._apply_seed()
        
        if sample_size is None:

            if num_nodes > 10000:
                sample_size = 500
            elif num_nodes > 5000:
                sample_size = 1000
            else:
                sample_size = min(2000, num_nodes)

        if num_nodes > sample_size:
            sampled_nodes = np.random.choice(num_nodes, size=sample_size, replace=False)
        else:
            sampled_nodes = np.arange(num_nodes)
        
        print(f"Computing PPR for {len(sampled_nodes)} sampled nodes...")
        
        ppr_vectors = []
        for i, node_idx in enumerate(sampled_nodes):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(sampled_nodes)}")
            
            if self.method == 'monte_carlo':
                ppr = self.ppr_calculator.monte_carlo_ppr(
                    edge_index, num_nodes, node_idx, n_walks=100, max_steps=30
                )
            elif self.method == 'forward_push':
                ppr = self.ppr_calculator.forward_push(
                    edge_index, num_nodes, node_idx, r_max=1e-4
                )
            else: 
                ppr = self.ppr_calculator.hybrid_ppr(
                    edge_index, num_nodes, node_idx, r_max=1e-4, n_walks=50
                )
            
            ppr_vectors.append(ppr.cpu().numpy())
        
        ppr_vectors = np.array(ppr_vectors)

        n_components = min(50, ppr_vectors.shape[1], len(ppr_vectors)-1)
        pca = PCA(n_components=n_components, random_state=42)
        ppr_reduced = pca.fit_transform(ppr_vectors)
        
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        kmeans = KMeans(n_clusters=self.target_num_groups, random_state=42, n_init=10)
        sampled_labels = kmeans.fit_predict(ppr_reduced)

        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
        deg_features = deg.unsqueeze(1).cpu().numpy()

        sampled_deg_features = deg_features[sampled_nodes]
        
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(sampled_deg_features)
        
        distances, indices = knn.kneighbors(deg_features)
        structural_labels = sampled_labels[indices.flatten()]
        
        return torch.tensor(structural_labels, device=edge_index.device)
    
    def precompute_groups(self, x, edge_index):
        print("Starting fast node grouping...")

        print("1. Fast attribute grouping...")
        attribute_groups = self.attribute_grouping(x)

        num_nodes = x.size(0)
        print(f"2. Fast PPR structural grouping for {num_nodes} nodes...")
        
        structural_groups = self.structural_grouping_fast_ppr(edge_index, num_nodes)
        print("3. Computing intersection groups...")
        final_groups = []
        
        for attr_group in range(self.target_num_groups):
            for struct_group in range(self.target_num_groups):
                mask = (attribute_groups == attr_group) & (structural_groups == struct_group)
                if mask.sum() > 0:
                    final_groups.append(torch.where(mask)[0])
        
        self.node_groups = final_groups
        self.actual_num_groups = len(final_groups)
        
        print(f"Fast grouping completed: {self.actual_num_groups} groups")
        return final_groups
    
    def attribute_grouping(self, x):
        if x.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            x_reduced = pca.fit_transform(x.cpu().detach().numpy())
        else:
            x_reduced = x.cpu().detach().numpy()
            
        kmeans = KMeans(n_clusters=self.target_num_groups, random_state=42, n_init=10)
        attribute_labels = kmeans.fit_predict(x_reduced)
        return torch.tensor(attribute_labels, device=x.device)
    
    def get_groups(self):
        if self.node_groups is None:
            raise ValueError("Node groups not computed. Call precompute_groups first.")
        return self.node_groups
    
    def get_actual_num_groups(self):
        if self.node_groups is None:
            raise ValueError("Node groups not computed. Call precompute_groups first.")
        return self.actual_num_groups
    
    def _get_cache_filename(self, dataset_name, num_nodes, num_features):
        os.makedirs(self.cache_dir, exist_ok=True)
        filename = f"{dataset_name}_{num_nodes}_{num_features}_{self.target_num_groups}_{self.alpha}_{self.method}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def save_groups(self, dataset_name, num_nodes, num_features):
        if self.node_groups is None:
            raise ValueError("No node groups to save. Call precompute_groups first.")
        
        cache_file = self._get_cache_filename(dataset_name, num_nodes, num_features)

        groups_data = {
            'node_groups': [group.cpu().numpy() for group in self.node_groups],
            'actual_num_groups': self.actual_num_groups,
            'target_num_groups': self.target_num_groups,
            'alpha': self.alpha,
            'method': self.method
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(groups_data, f)
        
        print(f"Node groups saved to: {cache_file}")
        return cache_file
    
    def load_groups(self, dataset_name, num_nodes, num_features):
        cache_file = self._get_cache_filename(dataset_name, num_nodes, num_features)
        
        if not os.path.exists(cache_file):
            print(f"Cache file not found: {cache_file}")
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                groups_data = pickle.load(f)

            if (groups_data['target_num_groups'] != self.target_num_groups or
                groups_data['alpha'] != self.alpha or
                groups_data['method'] != self.method):
                print(f"Parameter mismatch in cache file: {cache_file}")
                return False

            self.node_groups = [torch.from_numpy(group) for group in groups_data['node_groups']]
            self.actual_num_groups = groups_data['actual_num_groups']
            
            return True
            
        except Exception as e:
            print(f"Error loading cache file {cache_file}: {e}")
            return False
    
    def precompute_groups_with_cache(self, x, edge_index, dataset_name):
        num_nodes = x.size(0)
        num_features = x.size(1)

        if self.load_groups(dataset_name, num_nodes, num_features):
            return self.node_groups

        print("Cache not found or loading failed, computing node groups...")
        return self.precompute_groups(x, edge_index)


def preprocess_node_groups(x, edge_index, dataset_name, target_num_groups=5, 
                          alpha=0.15, method='monte_carlo', use_cache=True, 
                          force_recompute=False, seed=None):
    grouper = NodeGrouper(target_num_groups=target_num_groups, alpha=alpha, method=method, seed=seed)
    
    if force_recompute:
        print("Force recomputing node groups...")
        node_groups = grouper.precompute_groups(x, edge_index)
    elif use_cache:
        node_groups = grouper.precompute_groups_with_cache(x, edge_index, dataset_name)
    else:
        node_groups = grouper.precompute_groups(x, edge_index)
    
    actual_num_groups = grouper.get_actual_num_groups()

    cache_file = None
    if use_cache or force_recompute:
        cache_file = grouper.save_groups(dataset_name, x.size(0), x.size(1))
    
    return node_groups, actual_num_groups, cache_file


def load_dataset(name, root='./data', seed=None, verbose=True, fold_idx=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    if name.lower() == 'photo':
        dataset = Amazon(root=root, name='Photo')
        data = dataset[0]
        data = add_masks(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    elif name.lower() == 'computer':
        dataset = Amazon(root=root, name='Computers')
        data = dataset[0]
        data = add_masks(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    elif name.lower() in ['cs']:
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=root, name='CS', transform=NormalizeFeatures())
        data = dataset[0]
        data = add_masks(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    elif name.lower() == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        data = dataset[0]
        data.y = data.y.squeeze(1)   
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index)      
        data = T.ToSparseTensor(remove_edge_index=False)(data)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data = process_ogb(data, train_idx, valid_idx, test_idx)
    elif name.lower() in ['wikics', 'wiki-cs']:
        from torch_geometric.datasets import WikiCS
        dataset = WikiCS(root='./data/wikics')
        data = dataset[0]
        if fold_idx is None:
            fold_idx = 0  
        else:
            num_folds = data.train_mask.shape[1] if hasattr(data, 'train_mask') and data.train_mask.dim() == 2 else 20
            fold_idx = fold_idx % num_folds
        
        if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
            data.train_mask = data.train_mask[:, fold_idx]
        if hasattr(data, 'val_mask') and data.val_mask.dim() == 2:
            data.val_mask = data.val_mask[:, fold_idx]
    elif name.lower() == 'flickr':
        from torch_geometric.datasets import Flickr
        dataset = Flickr(root='./data/flickr')
        data = dataset[0]
    else:
        raise ValueError(f"Unknown dataset: {name}.")

    if verbose:
        print_dataset_info(name, dataset, data)
    
    return dataset, data

def process_ogb(data, train_idx, valid_idx, test_idx):
    n = data.num_nodes
    train_mask = create_mask(n, train_idx)
    val_mask = create_mask(n, valid_idx)
    test_mask = create_mask(n, test_idx)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def create_class_split(label, label_num_per_class=20, valid_num=500, test_num=1000):
        train_idx, non_train_idx = [], []
        idx = torch.arange(label.shape[0])
        class_list = label.unique()
        
        for i in range(class_list.shape[0]):
            c_i = class_list[i]
            idx_i = idx[label == c_i]
            n_i = idx_i.shape[0]
            if n_i <= label_num_per_class:
                train_idx += idx_i.tolist()
                print(f"Warning: Class {c_i} has only {n_i} samples, using all as training samples")
            else:
                rand_idx = idx_i[torch.randperm(n_i)]
                train_idx += rand_idx[:label_num_per_class].tolist()
                non_train_idx += rand_idx[label_num_per_class:].tolist()
        
        train_idx = torch.as_tensor(train_idx)
        non_train_idx = torch.as_tensor(non_train_idx)

        non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]

        if non_train_idx.shape[0] < valid_num + test_num:
            total_remaining = non_train_idx.shape[0]
            valid_num_adj = int(total_remaining * valid_num / (valid_num + test_num))
            valid_idx = non_train_idx[:valid_num_adj]
            test_idx = non_train_idx[valid_num_adj:]
            print(f"Warning: Insufficient samples. Using valid:{valid_idx.shape[0]}, test:{test_idx.shape[0]}")
        else:
            valid_idx = non_train_idx[:valid_num]
            test_idx = non_train_idx[valid_num:valid_num + test_num]
        
        print(f"Data split - train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
        
        return train_idx, valid_idx, test_idx
    

def add_masks(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

    num_nodes = data.num_nodes

    indices = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data

def create_mask(n, pos):
    res = torch.zeros(n, dtype=torch.bool)
    res[pos] = True
    return res

def print_dataset_info(name, dataset, data):
    print(f"\n{' Dataset Info ':-^50}")
    print(f"Name: {name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")
    print("-" * 50 + "\n")

def parse_expert_config(config_str):
    experts = []
    for expert_str in config_str.split(';'):
        if not expert_str:
            continue
        parts = expert_str.split(',')
        expert_type = parts[0]
        params = {}
        for param in parts[1:]:
            if '=' in param:
                key, value = param.split('=')
                try:
                    params[key] = eval(value)
                except:
                    params[key] = value
        experts.append({'expert_type': expert_type, **params})
    return experts
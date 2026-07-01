import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv


class Expert(nn.Module):
    def __init__(self, in_channels, out_channels, expert_type='gcn', 
                 dropout=0.5, use_bn=False, activation='relu', norm: str = 'ln'):
        super(Expert, self).__init__()
        self.expert_type = expert_type
        self.use_bn = use_bn
        
        if expert_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif expert_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif expert_type == 'gat':
            self.conv = GATConv(in_channels, out_channels)
        elif expert_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),  
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),  
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),  
                nn.ReLU(),
            )
            self.conv = GINConv(mlp)
        else:
            raise ValueError(f"Unsupported expert_type: {expert_type}")

        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'none' or norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm: {norm}. Choose from ['ln', 'bn', 'none'].")

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class GateNetwork(nn.Module):

    def __init__(self, in_channels, num_experts, top_k=2, dropout=0.3):
        super(GateNetwork, self).__init__()
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.gate_norm = nn.LayerNorm(in_channels)
        self.W_g = nn.Linear(in_channels, num_experts)
        self.W_n = nn.Linear(in_channels, num_experts)
        
    def forward(self, h):
        h = self.dropout(h)
        main_scores = self.W_g(h)
        noise_scores = self.W_n(h)
        epsilon = torch.randn_like(noise_scores)
        noisy_scores = noise_scores * F.softplus(epsilon)
        
        q_i = main_scores + noisy_scores
        topk_scores, topk_indices = torch.topk(q_i, self.top_k, dim=1)
        
        g_sparse = torch.zeros_like(q_i)
        g_sparse.scatter_(1, topk_indices, F.softmax(topk_scores, dim=1))
        
        return g_sparse

class ConfidencePredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(ConfidencePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):

        return self.predictor(x)


class ConfidenceFusion(nn.Module):
    def __init__(self, input_dim, num_experts, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.confidence_predictor = ConfidencePredictor(input_dim, dropout=dropout)
        self.eps = 1e-12   

    def safe_log(self, x):
        return torch.log(torch.clamp(x, min=self.eps))

    def forward(self, expert_outputs, shared_output):
        batch_size, num_experts, output_dim = expert_outputs.shape

        single_confidences = []
        for m in range(num_experts):
            expert_out = expert_outputs[:, m, :]
            confidence = torch.sigmoid(self.confidence_predictor(expert_out))  # [B,1]
            single_confidences.append(confidence.squeeze(-1))

        single_confidences = torch.stack(single_confidences, dim=1)  # [B, M]

        single_confidences = torch.clamp(single_confidences, min=1e-4, max=1-1e-4)

        multi_confidences = []
        safe_log_single = self.safe_log(single_confidences)  

        for m in range(num_experts):
            mask = [j for j in range(num_experts) if j != m]

            log_other = torch.sum(safe_log_single[:, mask], dim=1)       # [B]
            log_all = torch.sum(safe_log_single, dim=1)                  # [B]

            log_all = torch.clamp(log_all, min=-50, max=50)

            holo_confidence = log_other / (log_all + self.eps)
            multi_confidences.append(holo_confidence)

        multi_confidences = torch.stack(multi_confidences, dim=1)  # [B, M]
        overall_confidences = single_confidences + multi_confidences
        overall_confidences = torch.nan_to_num(
            overall_confidences,
            nan=-50.0,
            posinf=50.0,
            neginf=-50.0,
        )

        overall_confidences = torch.clamp(overall_confidences, min=-50, max=50)

        fusion_weights = F.softmax(overall_confidences, dim=1)

        weighted_experts = torch.sum(
            expert_outputs * fusion_weights.unsqueeze(-1),
            dim=1
        )

        fused_output = shared_output + weighted_experts

        return fused_output, fusion_weights, single_confidences, multi_confidences, overall_confidences


class ConfidenceAlignmentLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ConfidenceAlignmentLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, expert_outputs, single_confidences, labels, label_mask=None):
        
        batch_size, num_experts, num_classes = expert_outputs.shape
        
        masked_expert_outputs = expert_outputs[label_mask]  # [masked_size, num_experts, num_classes]
        masked_single_confidences = single_confidences[label_mask]  # [masked_size, num_experts]
        masked_labels = labels[label_mask]  # [masked_size]
        
        masked_size = masked_expert_outputs.shape[0]
        

        expert_probs = F.softmax(masked_expert_outputs, dim=-1)  # [masked_size, num_experts, num_classes]
        correct_class_probs = expert_probs[torch.arange(masked_size), :, masked_labels]  # [masked_size, num_experts]

        alignment_loss = F.l1_loss(masked_single_confidences, correct_class_probs, reduction=self.reduction)
        
        return alignment_loss


class RoutingContrastiveLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, gate_weights, node_groups, tau=0.1):

        prototypes = []
        for group in node_groups:
            if len(group) > 0:
                prototypes.append(gate_weights[group].mean(dim=0))
        
        if len(prototypes) <= 1:
            return torch.tensor(0.0, device=gate_weights.device)
        
        prototypes = torch.stack(prototypes) # [C, M]
        
        total_loss = 0
        node_count = 0

        for c_idx, group in enumerate(node_groups):
            if len(group) == 0: continue

            logits = F.cosine_similarity(
                gate_weights[group].unsqueeze(1), 
                prototypes.unsqueeze(0), 
                dim=-1
            ) / tau

            labels = torch.full((len(group),), c_idx, dtype=torch.long, device=gate_weights.device)
            total_loss += F.cross_entropy(logits, labels, reduction='sum')
            node_count += len(group)
            
        return total_loss / node_count


class MoEGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 num_shared_experts=1,
                 num_specialized_experts=3,
                 top_k=2, 
                 expert_type='gcn',
                 gate_dropout=0.3,
                 expert_dropout=0.5,
                 use_bn=False,
                 use_residual=False,
                 norm: str = 'ln'): 
        super(MoEGNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_shared_experts = num_shared_experts
        self.num_specialized_experts = num_specialized_experts
        self.top_k = top_k
        self.total_experts = num_shared_experts + num_specialized_experts
        self.use_residual = use_residual

        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
        
        self.shared_experts = nn.ModuleList()
        for i in range(num_shared_experts):
            shared_expert_type = 'gcn' if expert_type == 'mix' else expert_type
            expert = Expert(
                in_channels,
                out_channels,
                expert_type=shared_expert_type,
                dropout=expert_dropout,
                use_bn=use_bn,
                norm=norm,
            )
            self.shared_experts.append(expert)
        
        self.specialized_experts = nn.ModuleList()
        if expert_type == 'mix':
            mix_types = ['gcn', 'sage', 'gin']
            for i in range(num_specialized_experts):
                specialized_expert_type = mix_types[i % len(mix_types)]
                expert = Expert(
                    in_channels,
                    out_channels,
                    expert_type=specialized_expert_type,
                    dropout=expert_dropout, 
                    use_bn=use_bn,
                    norm=norm,
                )
                self.specialized_experts.append(expert)
        else:
            for i in range(num_specialized_experts):
                expert = Expert(
                    in_channels,
                    out_channels,
                    expert_type=expert_type,
                    dropout=expert_dropout, 
                    use_bn=use_bn,
                    norm=norm,
                )
                self.specialized_experts.append(expert)
    
        self.gate_network = GateNetwork(in_channels, num_specialized_experts, top_k, dropout=gate_dropout)
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        gate_weights = self.gate_network(x)
        h_shared_list = []
        for expert in self.shared_experts:
            h_expert = expert(x, edge_index)
            h_shared_list.append(h_expert)
        
        h_shared = torch.stack(h_shared_list).mean(dim=0)

        h_specialized_list = []
        for expert in self.specialized_experts:
            h_expert = expert(x, edge_index)
            h_specialized_list.append(h_expert)
        
        h_specialized = torch.stack(h_specialized_list, dim=1)

        h_specialized_weighted = torch.sum(
            h_specialized * gate_weights.unsqueeze(-1), dim=1
        )

        h_prime = h_shared + h_specialized_weighted

        if self.use_residual:
            if self.residual_proj is not None:
                h_prime = h_prime + self.residual_proj(x)
            else:
                h_prime = h_prime + x
        
        h_prime = self.activation(h_prime)
        
        return h_prime, gate_weights

class MoEGNN(nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_channels, 
                 num_classes, 
                 num_layers=3,
                 num_shared_experts=1,
                 num_specialized_experts=3,
                 top_k=2, 
                 expert_type='gcn',
                 gate_dropout=0.3,
                 expert_dropout=0.5,
                 confidence_dropout=0.1,
                 norm: str = 'ln',
                 routing_temperature=0.1,
                 use_confidence_fusion=True,
                 use_residual=False):
        super(MoEGNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_specialized_experts = num_specialized_experts
        self.top_k = top_k
        self.use_confidence_fusion = use_confidence_fusion

        if num_features > 8000:
            self.dim_reduction = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Dropout(expert_dropout)
            )
            self.use_dim_reduction = True
            actual_input_dim = hidden_channels
            print(f"⚠️  Input dimension {num_features} > 5000, adding dimension reduction to {hidden_channels}")
        else:
            self.dim_reduction = None
            self.use_dim_reduction = False
            actual_input_dim = num_features

        self.moe_layers = nn.ModuleList()
        for i in range(num_layers):

            if i == 0:
                in_channels = actual_input_dim
                out_channels = hidden_channels
            elif i == num_layers - 1:
                in_channels = hidden_channels
                out_channels = num_classes
            else:
                in_channels = hidden_channels
                out_channels = hidden_channels
            
            moe_layer = MoEGNNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                num_shared_experts=num_shared_experts,
                num_specialized_experts=num_specialized_experts,
                top_k=top_k,
                expert_type=expert_type,
                gate_dropout=gate_dropout,
                expert_dropout=expert_dropout,
                use_residual=use_residual,
                norm=norm
            )
            self.moe_layers.append(moe_layer)

        self.routing_losses = None
        self.node_groups = None

        if use_confidence_fusion:
            self.confidence_fusion = ConfidenceFusion(
                num_classes, num_specialized_experts, dropout=confidence_dropout
            )
            self.confidence_alignment_loss = ConfidenceAlignmentLoss()

        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (GCNConv, SAGEConv, GATConv, GINConv)):
                module.reset_parameters()
    
    def set_node_groups(self, node_groups):
        self.node_groups = node_groups
        self.actual_num_groups = len(node_groups)
        if self.actual_num_groups > 1:
            self.routing_losses = nn.ModuleList([
                RoutingContrastiveLoss() for _ in range(self.num_layers)
            ])
        else:
            self.routing_losses = None
    
    def forward(self, x, edge_index, edge_attr=None, return_routing_loss=False, 
                return_confidence_loss=False, labels=None, label_mask=None):
        if self.use_dim_reduction:
            h = self.dim_reduction(x)
        else:
            h = x
        
        all_gate_weights = []
        routing_losses = []
        for layer_idx, layer in enumerate(self.moe_layers[:-1]):
            h, gate_weights = layer(h, edge_index)
            all_gate_weights.append(gate_weights)

            if return_routing_loss and self.routing_losses is not None and self.node_groups is not None:
                layer_routing_loss = self.routing_losses[layer_idx](gate_weights, self.node_groups)
                routing_losses.append(layer_routing_loss)

        last_layer = self.moe_layers[-1]
        confidence_loss = None
        
        if not self.use_confidence_fusion:
            h, gate_weights = last_layer(h, edge_index)
            all_gate_weights.append(gate_weights)
            
            if return_routing_loss and self.routing_losses is not None and self.node_groups is not None:
                layer_routing_loss = self.routing_losses[-1](gate_weights, self.node_groups)
                routing_losses.append(layer_routing_loss)
        else:
            h_shared_list = []
            for expert in last_layer.shared_experts:
                h_expert = expert(h, edge_index)
                h_shared_list.append(h_expert)
            h_shared = torch.stack(h_shared_list).mean(dim=0)

            h_specialized_list = []
            for expert in last_layer.specialized_experts:
                h_expert = expert(h, edge_index)
                h_specialized_list.append(h_expert)
            h_specialized = torch.stack(h_specialized_list, dim=1)  # [num_nodes, num_experts, num_classes]

            h, fusion_weights, single_confidences, multi_confidences, overall_confidences = self.confidence_fusion(
                h_specialized, h_shared
            )

            all_gate_weights.append(None)
            
            if return_confidence_loss and labels is not None:
                expert_logits = h_specialized  # [num_nodes, num_experts, num_classes]
                confidence_loss = self.confidence_alignment_loss(
                    expert_logits, single_confidences, labels, label_mask
                )
            
            if return_routing_loss and self.routing_losses is not None and self.node_groups is not None:
                routing_losses.append(torch.tensor(0.0, device=h.device))

        logits = h
        results = [F.log_softmax(logits, dim=-1), all_gate_weights]
        
        if return_routing_loss and self.routing_losses is not None and len(routing_losses) > 0:
            total_routing_loss = torch.stack(routing_losses).mean()
            results.append(total_routing_loss)
        elif return_routing_loss:
            results.append(torch.tensor(0.0, device=logits.device))
        
        if return_confidence_loss and confidence_loss is not None:
            results.append(confidence_loss)
        elif return_confidence_loss:
            results.append(torch.tensor(0.0, device=logits.device))
        
        return tuple(results)
    
    def get_expert_usage(self, gate_weights):
        usage = []
        for g_i in gate_weights:
            layer_usage = (g_i > 0).float().mean(dim=0)
            usage.append(layer_usage)
        return usage
    
    def get_expert_statistics(self):
        stats = {
            'total_layers': self.num_layers,
            'total_shared_experts': sum(layer.num_shared_experts for layer in self.moe_layers),
            'total_specialized_experts': sum(layer.num_specialized_experts for layer in self.moe_layers),
            'total_experts': sum(layer.total_experts for layer in self.moe_layers),
            'actual_num_groups': self.actual_num_groups,
            'layer_configs': []
        }
        
        for i, layer in enumerate(self.moe_layers):
            stats['layer_configs'].append({
                'layer': i,
                'shared_experts': layer.num_shared_experts,
                'specialized_experts': layer.num_specialized_experts,
                'total_experts': layer.total_experts,
                'top_k': layer.top_k
            })
        
        return stats
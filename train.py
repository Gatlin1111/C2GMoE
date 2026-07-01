import argparse
import torch
import torch.nn.functional as F
import time
import numpy as np
from model import MoEGNN
from utils import load_dataset, preprocess_node_groups
from sklearn.metrics import f1_score, accuracy_score
import json
import os
import random

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, data, optimizer, epoch, total_epochs, lambda1=0.1, lambda2=0.1): 
    model.train()
    optimizer.zero_grad()

    forward_kwargs = {'return_routing_loss': model.routing_losses is not None}
    if hasattr(model, 'use_confidence_fusion') and model.use_confidence_fusion:
        forward_kwargs['return_confidence_loss'] = True
        forward_kwargs['labels'] = data.y  #
        forward_kwargs['label_mask'] = data.train_mask  

    model_output = model(data.x, data.edge_index, **forward_kwargs)

    out = model_output[0]
    all_gate_weights = model_output[1] if len(model_output) > 1 else []
    
    if len(model_output) >= 3:
        routing_loss = model_output[2]
    else:
        routing_loss = torch.tensor(0.0, device=out.device)
    
    if len(model_output) >= 4:
        confidence_loss = model_output[3]
    else:
        confidence_loss = torch.tensor(0.0, device=out.device)


    task_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    warmup_epochs = 100
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        l1 = lambda1 * progress
        l2 = lambda2 * progress
    else:
        l1 = lambda1
        l2 = lambda2

    loss = (task_loss + 
            l1 * routing_loss +
            l2 * confidence_loss)
        
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.confidence_fusion.parameters(), max_norm=2.0)

    optimizer.step()
    
    return loss.item(), task_loss.item(), routing_loss.item(), confidence_loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    model_output = model(data.x, data.edge_index)
    

    out = model_output[0]
    all_gate_weights = model_output[1] if len(model_output) > 1 else []
    
    pred = out.argmax(dim=1)

    y_train = data.y[data.train_mask].cpu().numpy()
    y_val = data.y[data.val_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()
    
    pred_train = pred[data.train_mask].cpu().numpy()
    pred_val = pred[data.val_mask].cpu().numpy()
    pred_test = pred[data.test_mask].cpu().numpy()

    train_acc = accuracy_score(y_train, pred_train)
    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(y_test, pred_test)

    train_macro_f1 = f1_score(y_train, pred_train, average='macro')
    val_macro_f1 = f1_score(y_val, pred_val, average='macro')
    test_macro_f1 = f1_score(y_test, pred_test, average='macro')

    train_micro_f1 = f1_score(y_train, pred_train, average='micro')
    val_micro_f1 = f1_score(y_val, pred_val, average='micro')
    test_micro_f1 = f1_score(y_test, pred_test, average='micro')
    
    expert_usage_per_layer = []
    for layer_idx, gate_weights in enumerate(all_gate_weights):
        if gate_weights is not None:
            usage = (gate_weights > 0).float().mean(dim=0)
            expert_usage_per_layer.append({
                'layer': layer_idx,
                'usage': usage.cpu().numpy()
            })
    
    return (train_acc, val_acc, test_acc, 
            train_macro_f1, val_macro_f1, test_macro_f1,
            train_micro_f1, val_micro_f1, test_micro_f1,
            expert_usage_per_layer)

def run_experiment(run_id, seed, args):
    print(f"\n{'='*60}")
    print(f"\nRun {run_id}")
    print(f"{'='*60}")
    fold_idx = (run_id - 1) % 20 if args.dataset.lower() in ['wikics', 'wiki-cs', 'chameleon','squirrel', 'roman-empire', 'amazon-ratings'] else None
    dataset, data = load_dataset(args.dataset, seed=seed, verbose=False, fold_idx=fold_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    def count_mask_nodes(mask):
        if isinstance(mask, torch.Tensor):
            return mask.cpu().numpy().sum()  
        elif isinstance(mask, np.ndarray):
            return mask.sum()
        else:
            return 0

    train_num = count_mask_nodes(data.train_mask)
    val_num = count_mask_nodes(data.val_mask)
    test_num = count_mask_nodes(data.test_mask)
    total_num = data.num_nodes 

    print(f"Train node: {train_num} ({train_num/total_num*100:.1f}%)")
    print(f"Valid Node: {val_num} ({val_num/total_num*100:.1f}%)")
    print(f"Test Node: {test_num} ({test_num/total_num*100:.1f}%)")

    print("Preprocessing node groups...")
    node_groups, actual_num_groups, cache_file = preprocess_node_groups(
        x=data.x,
        edge_index=data.edge_index,
        dataset_name=args.dataset,
        target_num_groups=args.target_num_groups,
        use_cache=True,
        force_recompute=False,
        seed=seed
    )
    
    print(f"Node groups: {actual_num_groups}")

    model = MoEGNN(
        num_features=dataset.num_features,
        hidden_channels=args.hidden_channels,
        num_classes=dataset.num_classes,
        num_layers=args.num_layers,
        num_shared_experts=args.num_shared_experts,
        num_specialized_experts=args.num_specialized_experts,
        top_k=args.top_k,
        expert_type=args.expert_type,
        gate_dropout=args.gate_dropout,
        expert_dropout=args.expert_dropout,
        confidence_dropout=args.confidence_dropout,
        norm=args.norm,
        use_confidence_fusion=True,
        use_residual=args.use_residual
    ).to(device)

    model.set_node_groups(node_groups)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    best_val_acc = 0
    best_test_acc = 0
    best_test_macro_f1 = 0
    best_test_micro_f1 = 0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss, task_loss, routing_loss, confidence_loss = train(
            model, data, optimizer, epoch, args.epochs, args.lambda1, args.lambda2
        )
        (train_acc, val_acc, test_acc, 
         train_macro_f1, val_macro_f1, test_macro_f1,
         train_micro_f1, val_micro_f1, test_micro_f1,
         expert_usage) = test(model, data)
        
        if epoch % 10 == 0 and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_test_macro_f1 = test_macro_f1
            best_test_micro_f1 = test_micro_f1
            best_epoch = epoch

        if epoch % 20 == 0:
            print(f'Run {run_id} - Epoch {epoch:03d}: '
                  f'Loss {loss:.4f}, Val Acc {val_acc:.4f}, TestAcc {test_acc:.4f}, Best Test Acc {best_test_acc:.4f}')
    
    result = {
        'seed': seed,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_test_macro_f1': best_test_macro_f1,
        'best_test_micro_f1': best_test_micro_f1,
        'best_epoch': best_epoch
    }
    
    print(f"Seed {seed} completed: Test Acc {best_test_acc:.4f}, Test Macro-F1 {best_test_macro_f1:.4f}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='MoEGNN on Graph Datasets')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='Dataset name')

    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    parser.add_argument('--lambda1', type=float, default=0.1,
                       help='Weight for routing contrastive loss (λ1)')
    parser.add_argument('--lambda2', type=float, default=0.1,
                       help='Weight for confidence predictor loss (λ2)')
    
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    parser.add_argument('--expert_type', type=str, default='gcn', choices=['gcn', 'sage', 'gin', 'mix'])
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of MoE layers')
    parser.add_argument('--num_shared_experts', type=int, default=1,
                       help='Number of shared experts per layer')
    parser.add_argument('--num_specialized_experts', type=int, default=8,
                       help='Number of specialized experts per layer')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of specialized experts to select per node')
    parser.add_argument('--target_num_groups', type=int, default=20,
                       help='Target number of node groups for preprocessing')

    parser.add_argument('--gate_dropout', type=float, default=0.1,
                       help='Dropout rate for gate networks')
    parser.add_argument('--expert_dropout', type=float, default=0.1,
                       help='Dropout rate for experts')
    parser.add_argument('--confidence_dropout', type=float, default=0.1,
                       help='Dropout rate for confidence predictor')

    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn', 'none'],
                       help='Normalization type used in experts/gates (ln: LayerNorm, bn: BatchNorm1d, none: Identity)')
    
    parser.add_argument('--use_residual', action='store_true',
                       help='Use residual connections in MoEGNN layers')
    
    args = parser.parse_args()

    seed = 2025
    set_all_seeds(seed)
    
    all_results = []
    
    for idx in range(1, 11):
        result = run_experiment(idx, seed, args)
        all_results.append(result)

    test_accs = [r['best_test_acc'] for r in all_results]
    test_macro_f1s = [r['best_test_macro_f1'] for r in all_results]
    test_micro_f1s = [r['best_test_micro_f1'] for r in all_results]
    val_accs = [r['best_val_acc'] for r in all_results]

    stats = {
        'test_acc': {
            'mean': np.mean(test_accs),
            'std': np.std(test_accs),
            'min': np.min(test_accs),
            'max': np.max(test_accs)
        },
        'test_macro_f1': {
            'mean': np.mean(test_macro_f1s),
            'std': np.std(test_macro_f1s),
            'min': np.min(test_macro_f1s),
            'max': np.max(test_macro_f1s)
        },
        'test_micro_f1': {
            'mean': np.mean(test_micro_f1s),
            'std': np.std(test_micro_f1s),
            'min': np.min(test_micro_f1s),
            'max': np.max(test_micro_f1s)
        },
        'val_acc': {
            'mean': np.mean(val_accs),
            'std': np.std(val_accs),
            'min': np.min(val_accs),
            'max': np.max(val_accs)
        }
    }

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS ")
    print(f"{'='*80}")
    
    print(f"\n📊 Individual Results:")
    print(f"{'Seed':>6} {'Val Acc':>8} {'Test Acc':>8} {'Test Macro':>10} {'Test Micro':>10}")
    print(f"{'-'*50}")
    for result in all_results:
        print(f"{result['seed']:6d} {result['best_val_acc']:8.4f} {result['best_test_acc']:8.4f} "
              f"{result['best_test_macro_f1']:10.4f} {result['best_test_micro_f1']:10.4f}")
    
    print(f"\n🎯 Key Results:")
    print(f"Test Accuracy:      {stats['test_acc']['mean']:.4f} ± {stats['test_acc']['std']:.4f}")
    print(f"Test Macro-F1:      {stats['test_macro_f1']['mean']:.4f} ± {stats['test_macro_f1']['std']:.4f}")
    print(f"Test Micro-F1:      {stats['test_micro_f1']['mean']:.4f} ± {stats['test_micro_f1']['std']:.4f}")
    print(f"Range:              [{stats['test_acc']['min']:.4f} - {stats['test_acc']['max']:.4f}]")

    results_dict = {
        'config': vars(args),
        'individual_results': all_results,
        'statistics': stats
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"moe_gnn_results_{args.dataset}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
import torch
import argparse
import warnings
from train import run_multiple_experiments, print_statistical_results, save_results_to_file

torch.set_default_dtype(torch.float32)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='Chameleon', 
                       choices=['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'CS', 'OGBN-Arxiv','Physics', 'Chameleon'])
    
    parser.add_argument('--imbalance_type', type=str, default='extreme_imbalance',
                       choices=['balanced', 'mild_imbalance', 'moderate_imbalance', 
                               'severe_imbalance', 'extreme_imbalance', 'ratio_50_imbalance'])
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save comprehensive results')
    parser.add_argument('--no_save', action='store_true', help='Disable result saving')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for statistical results')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Running {args.num_runs} experiments with base seed {args.seed}")
    
    all_results = run_multiple_experiments(args, device)
    
    print_statistical_results(all_results, args.dataset, args.imbalance_type)
    
    if args.save_results and not args.no_save:
        save_results_to_file(all_results, args.dataset, args.imbalance_type, args)

if __name__ == '__main__':
    main()
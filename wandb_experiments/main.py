import argparse
import torch
import wandb
import os

from config import SWEEP_CONFIG, DEV_SWEEP_CONFIG
from train import load_data, train_model
from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall


def train_and_evaluate():
    """Main training and evaluation function for a single wandb run"""
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize wandb run
    run = wandb.init()
    config = wandb.config
    
    # Load data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {config.num_training_examples} training examples...")
    
    positions, valid_moves = load_data(num_training_examples=config.num_training_examples)
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    
    # Train model
    print("Starting training...")
    results_dict, model = train_model(positions, valid_moves, config)
    
    # Evaluate model
    print("Evaluating model vs random opponent...")
    avg_moves, score, all_moves = evaluate_model_vs_random(model, num_games=100)
    
    # Calculate precision and recall on training data
    print("Calculating precision and recall...")
    model.eval()
    with torch.no_grad():
        predicted_moves = model(positions)
    precision = calculate_precision(predicted_moves, valid_moves)
    recall = calculate_recall(predicted_moves, valid_moves)
    
    # Create move length histogram
    all_moves_hist = [0] * 10
    for m in all_moves:
        all_moves_hist[m // 10] += 1
    
    # Log all results to wandb
    final_results = {
        'final_train_loss': results_dict['final_train_loss'],
        'final_test_loss': results_dict['final_test_loss'],
        'best_test_loss': results_dict['best_test_loss'],
        'avg_moves': avg_moves,
        'score': score,
        'precision': precision,
        'recall': recall,
        'win_rate': score / 100.0,  # Convert to percentage
    }
    
    # Log move histogram
    for i, count in enumerate(all_moves_hist):
        final_results[f'moves_hist_{i*10}-{i*10+9}'] = count
    
    wandb.log(final_results)
    
    # Save model artifact
    model_path = f"model_{run.name}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Log model as artifact
    artifact = wandb.Artifact(f"chess_model_{run.name}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # Clean up local model file
    os.remove(model_path)
    
    print(f"Run completed! Score: {score}, Win rate: {score/100:.2%}")
    
    wandb.finish()
    
    # Clear GPU memory at end
    del positions, valid_moves, model, predicted_moves
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description='Chess-SL WandB Experiments')
    parser.add_argument('--dev_mode', action='store_true', 
                       help='Run with development configuration (smaller parameter space)')
    parser.add_argument('--project', type=str, default='chess-sl',
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='WandB entity (username or team name)')
    parser.add_argument('--single_run', action='store_true',
                       help='Run a single experiment instead of a sweep')
    args = parser.parse_args()

    # Set up wandb project
    wandb.login()  # Make sure user is logged in
    
    if args.single_run:
        # Run single experiment with default configuration
        config = DEV_SWEEP_CONFIG if args.dev_mode else SWEEP_CONFIG
        default_config = {}
        for param, param_config in config['parameters'].items():
            if 'values' in param_config:
                default_config[param] = param_config['values'][0]
            else:
                default_config[param] = param_config['value']
        
        wandb.init(project=args.project, entity=args.entity, config=default_config)
        train_and_evaluate()
    else:
        # Initialize sweep
        sweep_config = DEV_SWEEP_CONFIG if args.dev_mode else SWEEP_CONFIG
        
        print(f"Creating {'development' if args.dev_mode else 'full'} hyperparameter sweep...")
        print(f"Project: {args.project}")
        if args.entity:
            print(f"Entity: {args.entity}")
        
        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.project,
            entity=args.entity
        )
        
        print(f"Sweep created with ID: {sweep_id}")
        print(f"Starting sweep agent...")
        print(f"You can monitor progress at: https://wandb.ai/{args.entity or 'your-username'}/{args.project}/sweeps/{sweep_id}")
        
        # Run sweep agent
        wandb.agent(sweep_id, train_and_evaluate)


if __name__ == "__main__":
    main()
import torch
import wandb

from train import load_data, train_model
from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall


def main():
    """Main training and evaluation function that expects wandb.config to be set"""
    # Initialize wandb run
    wandb.init()
    
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get config from wandb
    config = dict(wandb.config)
    
    # Load data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {config['num_training_examples']} training examples...")
    
    positions, valid_moves = load_data(num_training_examples=config['num_training_examples'])
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    
    # Train model
    print("Starting training...")
    results_dict, model = train_model(positions, valid_moves, config)
    
    # Evaluate model
    print("Evaluating model vs random opponent...")
    avg_moves, score, all_moves = evaluate_model_vs_random(model, num_games=100)

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
        # 'precision': precision,
        # 'recall': recall,
        'win_rate': score / 100.0,  # Convert to percentage
    }
    
    # Log move histogram
    for i, count in enumerate(all_moves_hist):
        final_results[f'moves_hist_{i*10}-{i*10+9}'] = count
    
    wandb.log(final_results)
    
    print(f"Run completed! Score: {score}, Win rate: {score/100:.2%}")

if __name__ == "__main__":
    main()
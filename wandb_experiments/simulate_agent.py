#!/usr/bin/env python3

import torch
import wandb
import gc
import sys

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB, Free: {free:.3f} GB")

def train_and_evaluate():
    """This is the EXACT function from main.py that wandb calls"""
    # This is copied directly from main.py to simulate the exact wandb agent behavior
    from train import load_data, train_model
    from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall
    
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize wandb run (this is called by wandb agent)
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
    
    # THIS IS THE PROBLEMATIC PART - try to skip it for now
    try:
        print("Calculating precision and recall...")
        model.eval()
        with torch.no_grad():
            predicted_moves = model(positions)  # This causes OOM
        precision = calculate_precision(predicted_moves, valid_moves)
        recall = calculate_recall(predicted_moves, valid_moves)
    except torch.cuda.OutOfMemoryError:
        print("Skipping precision/recall due to OOM")
        precision = 0.0
        recall = 0.0
        predicted_moves = torch.zeros(1)  # Dummy tensor
    
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
        'win_rate': score / 100.0,
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
    import os
    os.remove(model_path)
    
    print(f"Run completed! Score: {score}, Win rate: {score/100:.2%}")
    
    wandb.finish()
    
    # Clear GPU memory at end
    del positions, valid_moves, model, predicted_moves
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def simulate_wandb_agent():
    """Simulate the wandb agent calling train_and_evaluate multiple times"""
    
    # Simulate multiple sweep configurations
    configs = [
        {'num_training_examples': 30000, 'n_blocks': 0, 'n_channels': 4, 'n_hidden': 1024, 'batch_size': 64, 'num_epochs': 1, 'optimizer': 'SGD', 'learning_rate_multiplier': 1},
        {'num_training_examples': 30000, 'n_blocks': 0, 'n_channels': 4, 'n_hidden': 1024, 'batch_size': 64, 'num_epochs': 1, 'optimizer': 'ADAM', 'learning_rate_multiplier': 1},
        {'num_training_examples': 30000, 'n_blocks': 0, 'n_channels': 4, 'n_hidden': 2048, 'batch_size': 64, 'num_epochs': 1, 'optimizer': 'SGD', 'learning_rate_multiplier': 1},
    ]
    
    print("=== Simulating WandB Agent ===")
    print_memory_info("Agent start")
    
    for i, config in enumerate(configs):
        print(f"\n--- Agent Run {i+1} ---")
        print_memory_info(f"Before run {i+1}")
        
        # This is what wandb.agent does - it calls the function with config
        try:
            # Monkey patch wandb to use our config
            class MockRun:
                def __init__(self, config):
                    self.config = type('Config', (), config)()
                    self.name = f"mock-run-{i+1}"
                    
            wandb.init = lambda: MockRun(config)
            
            train_and_evaluate()
            
        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            
        print_memory_info(f"After run {i+1}")
        
        # Force cleanup like agent might do
        gc.collect()
        print_memory_info(f"After cleanup {i+1}")

if __name__ == "__main__":
    simulate_wandb_agent()
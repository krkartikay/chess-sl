#!/usr/bin/env python3

import torch
import wandb
import gc
import os

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")

def simulate_actual_wandb_run():
    """Simulate the exact sequence that happens in the real wandb run"""
    from train import load_data, train_model
    from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall
    
    print_memory_info("Run start")
    
    # This mimics exactly what main.py does
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize wandb run (this might be the culprit!)
    run = wandb.init(project="debug-memory", mode="disabled")  # disabled to prevent actual logging
    config = wandb.config
    
    # Set some config values manually since we're not using a real sweep
    config.update({
        'num_training_examples': 30000,
        'n_blocks': 0,
        'n_channels': 4,
        'n_hidden': 2048,
        'batch_size': 64,
        'num_epochs': 2,  # Reduced for debugging
        'learning_rate_multiplier': 1,
        'optimizer': 'ADAM'
    })
    
    print_memory_info("After wandb.init")
    
    # Load data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positions, valid_moves = load_data(num_training_examples=config.num_training_examples)
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    
    print_memory_info("After data loading")
    
    # Train model (this calls train_model which might have issues)
    results_dict, model = train_model(positions, valid_moves, config)
    
    print_memory_info("After training")
    
    # Evaluate model
    avg_moves, score, all_moves = evaluate_model_vs_random(model, num_games=10)  # Reduced games
    
    print_memory_info("After evaluation vs random")
    
    # Calculate precision and recall (this is where the OOM happens!)
    model.eval()
    with torch.no_grad():
        predicted_moves = model(positions)  # This line causes OOM in real runs
    precision = calculate_precision(predicted_moves, valid_moves)
    recall = calculate_recall(predicted_moves, valid_moves)
    
    print_memory_info("After precision/recall")
    
    # Log results (wandb might keep references!)
    final_results = {
        'final_train_loss': results_dict['final_train_loss'],
        'final_test_loss': results_dict['final_test_loss'],
        'best_test_loss': results_dict['best_test_loss'],
        'avg_moves': avg_moves,
        'score': score,
        'precision': precision,
        'recall': recall,
    }
    
    wandb.log(final_results)
    print_memory_info("After wandb.log")
    
    # Save model artifact (might keep references!)
    model_path = f"debug_model.pt"
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact(f"debug_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print_memory_info("After artifact logging")
    
    # Clean up
    os.remove(model_path)
    
    # Manual cleanup
    del positions, valid_moves, model, predicted_moves
    
    print_memory_info("After manual deletion")
    
    # wandb.finish() - does this clean up properly?
    wandb.finish()
    
    print_memory_info("After wandb.finish")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print_memory_info("After final cleanup")

def main():
    print("=== WandB Memory Leak Debug ===")
    print_memory_info("Initial")
    
    for i in range(3):
        print(f"\n--- WandB Run {i+1} ---")
        simulate_actual_wandb_run()
        print_memory_info(f"End of wandb run {i+1}")
        gc.collect()
        print_memory_info(f"After gc.collect run {i+1}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import torch
import wandb
import gc
import subprocess
import time

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB, Free: {free:.3f} GB")

def run_single_wandb_experiment():
    """Run a single experiment with REAL wandb logging (not disabled)"""
    from train import load_data, train_model
    from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall
    
    print_memory_info("Real wandb run start")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize REAL wandb run (not disabled!)
    run = wandb.init(
        project="debug-memory-leak", 
        config={
            'num_training_examples': 30000,  # Smaller to avoid precision/recall OOM
            'n_blocks': 0,
            'n_channels': 4,
            'n_hidden': 1024,
            'batch_size': 64,
            'num_epochs': 1,
            'learning_rate_multiplier': 1,
            'optimizer': 'SGD'
        }
    )
    
    print_memory_info("After real wandb.init")
    
    # Load and process data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positions, valid_moves = load_data(num_training_examples=run.config.num_training_examples)
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    
    print_memory_info("After data loading")
    
    # Train model
    results_dict, model = train_model(positions, valid_moves, run.config)
    print_memory_info("After training")
    
    # Evaluate (skip precision/recall to avoid OOM)
    avg_moves, score, all_moves = evaluate_model_vs_random(model, num_games=10)
    print_memory_info("After evaluation")
    
    # Log results to wandb (this might hold references)
    wandb.log({
        'final_train_loss': results_dict['final_train_loss'],
        'final_test_loss': results_dict['final_test_loss'],
        'score': score
    })
    print_memory_info("After wandb.log")
    
    # Save and log artifact (this might hold references)
    model_path = f"debug_model_{run.id}.pt"
    torch.save(model.state_dict(), model_path)
    
    artifact = wandb.Artifact(f"debug_model_{run.id}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print_memory_info("After artifact logging")
    
    # Cleanup
    import os
    os.remove(model_path)
    del positions, valid_moves, model
    
    print_memory_info("After manual cleanup")
    
    # Finish wandb
    wandb.finish()
    print_memory_info("After wandb.finish")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    print_memory_info("After final cleanup")

def test_real_wandb_accumulation():
    """Test if real wandb logging causes accumulation"""
    print("=== Testing Real WandB Memory Accumulation ===")
    
    for i in range(3):
        print(f"\n--- Real WandB Run {i+1} ---")
        try:
            run_single_wandb_experiment()
            time.sleep(2)  # Small delay between runs
        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            break

if __name__ == "__main__":
    test_real_wandb_accumulation()
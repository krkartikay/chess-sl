#!/usr/bin/env python3

import torch
import wandb
import gc
import os

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB, Free: {free:.3f} GB")

def simulate_failing_run():
    """Try to reproduce the exact conditions from the failing logs"""
    from train import load_data, train_model
    from evaluate import evaluate_model_vs_random, calculate_precision, calculate_recall
    
    # Use the EXACT same parameters as the failing runs
    config = {
        'num_training_examples': 100000,  # This is what was failing
        'n_blocks': 0,
        'n_channels': 4, 
        'n_hidden': 1024,  # The logs show this was the config for failing runs
        'batch_size': 64,
        'num_epochs': 1,  # Full epochs like real runs
        'learning_rate_multiplier': 1,
        'optimizer': 'SGD'
    }
    
    print(f"Simulating run with {config['num_training_examples']} examples, {config['n_hidden']} hidden units")
    print_memory_info("Run start")
    
    # Clear memory like main.py does
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Initialize wandb in disabled mode to avoid actual logging
    run = wandb.init(project="debug-oom", mode="disabled")
    wandb.config.update(config)
    
    print_memory_info("After wandb.init")
    
    # Load data exactly like main.py
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {config['num_training_examples']} training examples...")
    
    positions, valid_moves = load_data(num_training_examples=config['num_training_examples'])
    print_memory_info("After load_data")
    
    # Move to GPU like main.py does
    positions = positions.to(device)
    print_memory_info("After positions.to(device)")
    
    valid_moves = valid_moves.to(device)
    print_memory_info("After valid_moves.to(device)")
    
    # Train model with full epochs
    print("Starting training...")
    try:
        results_dict, model = train_model(positions, valid_moves, wandb.config)
        print_memory_info("After train_model")
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during training: {e}")
        print_memory_info("After training OOM")
        return
    
    # Evaluate model
    print("Evaluating model vs random opponent...")
    try:
        avg_moves, score, all_moves = evaluate_model_vs_random(model, num_games=100)
        print_memory_info("After evaluate_model_vs_random")
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during evaluation: {e}")
        print_memory_info("After evaluation OOM")
        return
    
    # This is where it fails in the logs - precision/recall calculation
    print("Calculating precision and recall...")
    try:
        model.eval()
        with torch.no_grad():
            predicted_moves = model(positions)  # This line causes OOM in logs
        print_memory_info("After model(positions)")
        
        precision = calculate_precision(predicted_moves, valid_moves)
        recall = calculate_recall(predicted_moves, valid_moves)
        print_memory_info("After precision/recall calculation")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during precision/recall: {e}")
        print_memory_info("After precision/recall OOM")
        return
    
    print("Run completed successfully")
    wandb.finish()

def reproduce_accumulation():
    """Try to reproduce the memory accumulation across multiple runs"""
    print("=== Attempting to Reproduce OOM ===")
    print_memory_info("Initial state")

    for i in range(10):  # Try multiple runs
        print(f"\n--- Run {i+1} ---")
        try:
            simulate_failing_run()
            print_memory_info(f"End of run {i+1}")
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_memory_info(f"After cleanup run {i+1}")
            
        except Exception as e:
            print(f"Run {i+1} failed with: {e}")
            print_memory_info(f"After run {i+1} failure")
            break

if __name__ == "__main__":
    reproduce_accumulation()
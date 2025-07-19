#!/usr/bin/env python3

import torch
import os
from train import load_data

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:30s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")

def simulate_run():
    """Simulate a single wandb run to track memory usage"""
    print_memory_info("Start of run")
    
    # Load data (like main.py does)
    positions, valid_moves = load_data(num_training_examples=10000)
    print_memory_info("After load_data")
    
    # Move to GPU (like main.py does)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    print_memory_info("After moving to GPU in main")
    
    # This is what train_model does - ANOTHER move to GPU!
    positions = positions.to(device)  # Redundant!
    valid_moves = valid_moves.to(device)  # Redundant!
    print_memory_info("After redundant GPU move in train")
    
    # Cleanup
    del positions, valid_moves
    print_memory_info("After del")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_info("After empty_cache")

if __name__ == "__main__":
    print("=== Memory Debug Test ===")
    print_memory_info("Initial")
    
    # Run multiple simulated runs to see accumulation
    for i in range(3):
        print(f"\n--- Simulated Run {i+1} ---")
        simulate_run()
#!/usr/bin/env python3

import torch
import gc
import sys

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")

def print_tensor_count():
    """Count how many tensors are in memory"""
    tensor_count = 0
    total_size = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            tensor_count += 1
            if obj.is_cuda:
                total_size += obj.element_size() * obj.nelement()
    print(f"CUDA tensors in memory: {tensor_count}, Total size: {total_size / 1024**3:.3f} GB")

def simulate_wandb_run():
    """Simulate what happens in a wandb run"""
    from train import load_data
    from model import ChessModel
    
    print_memory_info("Run start")
    print_tensor_count()
    
    # What main.py does
    positions, valid_moves = load_data(num_training_examples=30000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    
    print_memory_info("After data loading")
    print_tensor_count()
    
    # Create model and do some training simulation
    model = ChessModel(n_blocks=0, n_channels=4, n_hidden=2048, filter_size=3).to(device)
    
    print_memory_info("After model creation")
    print_tensor_count()
    
    # Do some forward passes (simulating training)
    model.eval()
    with torch.no_grad():
        _ = model(positions[:1000])  # Small batch
    
    print_memory_info("After forward pass")
    print_tensor_count()
    
    # This is what main.py tries to do at the end
    del positions, valid_moves, model
    
    print_memory_info("After manual deletion")
    print_tensor_count()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_memory_info("After empty_cache")
    print_tensor_count()
    
    # Force garbage collection
    gc.collect()
    
    print_memory_info("After gc.collect")
    print_tensor_count()

def main():
    print("=== Memory Persistence Debug ===")
    print_memory_info("Initial state")
    print_tensor_count()
    
    # Simulate multiple runs like wandb sweep does
    for i in range(3):
        print(f"\n--- Simulated Run {i+1} ---")
        simulate_wandb_run()
        print_memory_info(f"End of run {i+1}")
        print_tensor_count()

if __name__ == "__main__":
    main()
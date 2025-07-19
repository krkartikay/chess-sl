#!/usr/bin/env python3

import torch
from train import load_data
from model import ChessModel

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")

def debug_evaluation_memory():
    """Debug the memory issue in evaluation step"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_memory_info("Start")
    
    # Load large dataset like failing runs
    positions, valid_moves = load_data(num_training_examples=100000)
    print_memory_info("After loading 100k examples")
    
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    print_memory_info("After moving 100k to GPU")
    
    # Create model
    model = ChessModel(n_blocks=0, n_channels=4, n_hidden=2048, filter_size=3).to(device)
    print_memory_info("After creating model")
    
    model.eval()
    
    # This is what main.py does - process ALL positions at once!
    print("About to process all positions at once (this will likely OOM)...")
    try:
        with torch.no_grad():
            predicted_moves = model(positions)  # This line causes OOM!
        print_memory_info("After full evaluation (if it worked)")
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error: {e}")
        print_memory_info("After OOM error")
    
    # Show how it should be done - in batches
    print("\nNow trying batch-wise evaluation...")
    torch.cuda.empty_cache()
    print_memory_info("After clearing cache")
    
    batch_size = 1000
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i:i+batch_size]
            batch_predictions = model(batch_positions)
            all_predictions.append(batch_predictions.cpu())  # Move to CPU to save GPU memory
            print_memory_info(f"After batch {i//batch_size + 1}")
    
    predicted_moves = torch.cat(all_predictions, dim=0).to(device)
    print_memory_info("After concatenating all predictions")

if __name__ == "__main__":
    debug_evaluation_memory()
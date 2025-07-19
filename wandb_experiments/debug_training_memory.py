#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from train import load_data
from model import ChessModel
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

def print_memory_info(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:40s} - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")

def debug_training_loop():
    """Debug memory during actual training to find the leak"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_memory_info("Start")
    
    # Load data
    positions, valid_moves = load_data(num_training_examples=30000)  # Use 30k like the failing runs
    print_memory_info("After loading data")
    
    # Move to GPU
    positions = positions.to(device)
    valid_moves = valid_moves.to(device)
    print_memory_info("After moving data to GPU")
    
    # Create model
    model = ChessModel(n_blocks=0, n_channels=4, n_hidden=2048, filter_size=3).to(device)
    print_memory_info("After creating model")
    
    # Create data loaders (this might be causing issues)
    train_size = int(0.8 * len(positions))
    test_size = len(positions) - train_size
    
    train_dataset, test_dataset = random_split(
        TensorDataset(positions, valid_moves), [train_size, test_size])
    print_memory_info("After creating datasets")
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    print_memory_info("After creating dataloaders")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Training loop (just a few batches to see memory pattern)
    model.train()
    for batch_num, (batch_positions, batch_moves) in enumerate(train_dataloader):
        if batch_num >= 5:  # Only do a few batches
            break
            
        print_memory_info(f"Batch {batch_num} start")
        
        optimizer.zero_grad()
        move_probs = model(batch_positions)
        print_memory_info(f"Batch {batch_num} after forward")
        
        loss = F.binary_cross_entropy(move_probs, batch_moves)
        print_memory_info(f"Batch {batch_num} after loss")
        
        loss.backward()
        print_memory_info(f"Batch {batch_num} after backward")
        
        optimizer.step()
        print_memory_info(f"Batch {batch_num} after step")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        for batch_num, (test_positions, test_moves) in enumerate(test_dataloader):
            if batch_num >= 3:
                break
            print_memory_info(f"Test batch {batch_num} start")
            test_probs = model(test_positions)
            print_memory_info(f"Test batch {batch_num} after forward")
    
    print_memory_info("End of training debug")

if __name__ == "__main__":
    debug_training_loop()
import torch
import torch.nn.functional as F
import wandb
import os
import argparse

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from typing import Dict, List, Tuple

from model import ChessModel


def load_data(filename="../games.pth", num_training_examples=10000) -> Tuple[torch.Tensor, torch.Tensor]:
    print("Loading data...")
    
    # Try to load from parent directory first, then current directory
    data_paths = [filename, "games.pth"]
    data = None
    
    for path in data_paths:
        if os.path.exists(path):
            with open(path, "rb") as datafile:
                data = torch.load(datafile)
            break
    
    if data is None:
        raise FileNotFoundError(f"Could not find games.pth in any of these locations: {data_paths}")
    
    positions: torch.Tensor = data["positions"][:num_training_examples]
    valid_moves: torch.Tensor = data["moves"][:num_training_examples].float()

    print("Loaded data. Shape: ")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")

    return positions, valid_moves


def train_model(positions: torch.Tensor, valid_moves: torch.Tensor, config: dict, save_model: bool = False) -> Tuple[Dict, ChessModel]:
    # Transfer data to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    positions = positions.to(device)
    valid_moves = valid_moves.to(device)

    # Create the neural net with wandb config
    chess_model = ChessModel(
        n_blocks=config['n_blocks'],
        n_channels=config['n_channels'], 
        n_hidden=config['n_hidden'],
        filter_size=3  # Fixed value
    ).to(device)
    
    # Log model parameters
    wandb.log({
        "model_parameters": sum(p.numel() for p in chess_model.parameters()),
        "model_trainable_parameters": sum(p.numel() for p in chess_model.parameters() if p.requires_grad)
    })

    # Splitting the dataset into training and testing
    train_size = int(0.8 * len(positions))  # 80% for training
    test_size = len(positions) - train_size

    train_dataset, test_dataset = random_split(
        TensorDataset(positions, valid_moves), [train_size, test_size])

    dataloader_params = {'batch_size': config['batch_size'], 'shuffle': True}

    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    optimizer_class = {'SGD': torch.optim.SGD, 'ADAM': torch.optim.Adam}[config['optimizer']]
    
    # Convert learning rate multiplier to actual learning rate
    lr = config['learning_rate_multiplier']
    if config['optimizer'] == 'ADAM':
        lr = lr * 3e-4
    if config['optimizer'] == 'SGD':
        lr = lr * 1e2
    
    optimizer = optimizer_class(chess_model.parameters(), lr=lr)
    
    # Log optimizer config
    wandb.log({"actual_learning_rate": lr})

    best_test_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Training mode
        chess_model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_num, (train_positions, train_valid_moves) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # train_positions: [batch_size, 7, 8, 8] float32
            # train_valid_moves: [batch_size, 64*64] float32
            move_probs = chess_model(train_positions)  # -> [batch_size, 64*64] float32
            loss = F.binary_cross_entropy(move_probs, train_valid_moves)  # -> scalar

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_num % 100 == 0:
                print(f"{epoch+1}/{batch_num+1:3d}, Loss: {loss.item():.4f}")
                # Log batch-level metrics
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": epoch * len(train_dataloader) + batch_num
                })

        # Test Evaluation
        chess_model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for test_positions, test_valid_moves in test_dataloader:
                # test_positions: [batch_size, 7, 8, 8] float32
                # test_valid_moves: [batch_size, 64*64] float32
                test_move_probs = chess_model(test_positions)  # -> [batch_size, 64*64] float32
                test_loss = F.binary_cross_entropy(test_move_probs, test_valid_moves)  # -> scalar
                total_test_loss += test_loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        average_test_loss = total_test_loss / len(test_dataloader)
        
        # Track best model
        if average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "test_loss": average_test_loss,
            "best_test_loss": best_test_loss
        })
        
        print(f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}")

    results = {
        'final_train_loss': average_train_loss, 
        'final_test_loss': average_test_loss,
        'best_test_loss': best_test_loss
    }

    # Save model if requested
    if save_model:
        torch.save(chess_model.state_dict(), 'model.pth')
        print("Model saved to model.pth")

    return results, chess_model


def main():
    parser = argparse.ArgumentParser(description='Train a chess model')
    parser.add_argument('--save-model', action='store_true', 
                       help='Save the trained model to model.pth')
    parser.add_argument('--data-file', default='../games.pth',
                       help='Path to the training data file (default: ../games.pth)')
    parser.add_argument('--num-examples', type=int, default=10000,
                       help='Number of training examples to use (default: 10000)')
    
    args = parser.parse_args()
    
    # Default configuration for standalone training
    default_config = {
        'n_blocks': 8,
        'n_channels': 128,
        'n_hidden': 4096,
        'batch_size': 256,
        'num_epochs': 20,
        'optimizer': 'ADAM',
        'learning_rate_multiplier': 3.0
    }
    
    # Initialize wandb for tracking
    wandb.init(project="chess-sl", config=default_config)
    
    # Load data
    positions, valid_moves = load_data(args.data_file, args.num_examples)
    
    # Train model
    results, model = train_model(positions, valid_moves, default_config, save_model=args.save_model)
    
    print(f"Training completed. Final test loss: {results['final_test_loss']:.4f}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
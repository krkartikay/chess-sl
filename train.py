import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from typing import Dict, List, Tuple

from model import ChessModel
from observer import Observer

from config import *

def load_data(filename="games.pth") -> Tuple[torch.Tensor, torch.Tensor]:
    print("Loading data...")
    n_examples = NUM_TRAINING_EXAMPLES.get()

    with open("games.pth", "rb") as datafile:
        data = torch.load(datafile)
        positions: torch.Tensor = data["positions"][:n_examples]
        valid_moves: torch.Tensor = data["moves"][:n_examples].float()

    print("Loaded data. Shape: ")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")

    return positions, valid_moves

def train_model(positions: torch.Tensor, valid_moves: torch.Tensor) -> Tuple[Dict, ChessModel]:
    # Transfer data to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clear GPU cache before loading data
    if device == 'cuda':
        torch.cuda.empty_cache()

    positions = positions.to(device)
    valid_moves = valid_moves.to(device)

    # Create the neural net
    chess_model = ChessModel().to(device)

    # Splitting the dataset into training and testing
    train_size = int(0.8 * len(positions))  # 80% for training
    test_size = len(positions) - train_size

    train_dataset, test_dataset = random_split(
        TensorDataset(positions, valid_moves), [train_size, test_size])

    dataloader_params = {'batch_size': BATCH_SIZE.get(), 'shuffle': True}

    train_dataloader = DataLoader(train_dataset, **dataloader_params)
    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    optimizer = {'SGD': torch.optim.SGD,
                 'ADAM': torch.optim.Adam}[OPTIMIZER.get()]
    lr = LEARNING_RATE.get()
    if OPTIMIZER.get() == 'ADAM':
        lr = LEARNING_RATE.get() * 3e-4
    if OPTIMIZER.get() == 'SGD':
        lr = LEARNING_RATE.get() * 1e2
    sgd_optimizer = optimizer(chess_model.parameters(), lr=lr)

    loss_observer = Observer('loss', path="results/",
                             labels=['train_loss', 'test_loss'])

    for epoch in range(NUM_EPOCHS.get()):
        # Training mode
        chess_model.train()
        total_train_loss = 0
        for batch_num, (train_positions, train_valid_moves) in enumerate(train_dataloader):
            sgd_optimizer.zero_grad()

            move_probs = chess_model(train_positions)
            loss = F.binary_cross_entropy(move_probs, train_valid_moves)

            loss.backward()
            sgd_optimizer.step()

            total_train_loss += loss.item()
            if batch_num % 10 == 0:
                print(f"{epoch+1}/{batch_num+1:3d}, Loss: {loss.item():.4f}")

        # Test Evaluation
        chess_model.eval()

        total_test_loss = 0
        with torch.no_grad():
            for test_positions, test_valid_moves in test_dataloader:
                test_move_probs = chess_model(test_positions)
                test_loss = F.binary_cross_entropy(
                    test_move_probs, test_valid_moves)
                total_test_loss += test_loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        average_test_loss = total_test_loss / len(test_dataloader)
        loss_observer.record([average_train_loss, average_test_loss])
        print(f"Epoch {epoch+1}, Average Test Loss: {average_test_loss:.4f}")
        
        # Clear GPU cache after each epoch
        if device == 'cuda':
            torch.cuda.empty_cache()

    results = {'final_train_loss': average_train_loss, 'final_test_loss': average_test_loss}

    return results, chess_model

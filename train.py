import torch
import torch.nn.functional as F
import observation

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.utils.data.dataset import random_split

from model import ChessModel

BATCH_SIZE = 256
LEARNING_RATE = 0.1
NUM_EPOCHS = 10

print("Loading data...")

with open("games.pth", "rb") as datafile:
    data = torch.load(datafile)
    positions: torch.Tensor = data["positions"]
    valid_moves: torch.Tensor = data["moves"].float()

print("Loaded data. Shape: ")
print(f"positions : {positions.size()}")
print(f"moves     : {valid_moves.size()}")
print()

# Transfer data to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

positions = positions.to(device)
valid_moves = valid_moves.to(device)

# Create the neural net
chess_model = ChessModel().to(device)

# Splitting the dataset into training and testing
train_size = int(0.8 * len(positions))  # 80% for training
test_size = len(positions) - train_size

train_dataset, test_dataset = random_split(
    TensorDataset(positions, valid_moves), [train_size, test_size])

dataloader_params = {'batch_size': BATCH_SIZE, 'shuffle': True}

train_dataloader = DataLoader(train_dataset, **dataloader_params)
test_dataloader = DataLoader(test_dataset, **dataloader_params)

sgd_optimizer = SGD(chess_model.parameters(), lr=LEARNING_RATE)

t_loss_obs = observation.ContinuousObserver("train_loss", "results")
v_loss_obs = observation.ContinuousObserver("test_loss", "results")

for epoch in range(NUM_EPOCHS):
    for train_positions, train_valid_moves in train_dataloader:
        sgd_optimizer.zero_grad()

        move_probs = chess_model(train_positions)
        loss = F.binary_cross_entropy(move_probs, train_valid_moves)

        loss.backward()
        sgd_optimizer.step()

        t_loss_obs.record(loss.item())

    # Test Evaluation
    chess_model.eval()

    total_test_loss = 0
    with torch.no_grad():
        for test_positions, test_valid_moves in test_dataloader:
            test_move_probs = chess_model(test_positions)
            test_loss = F.binary_cross_entropy(
                test_move_probs, test_valid_moves)
            total_test_loss += test_loss.item()

    average_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch+1}, Average Test Loss: {average_test_loss:.4f}")
    v_loss_obs.record(average_test_loss)

t_loss_obs.plot()
v_loss_obs.plot()
v_loss_obs.write_csv()

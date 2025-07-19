import chess
import random
import torch
import os

from chess_utils import board_to_tensor, moves_to_tensor

from typing import List, Tuple

def generate_games(num_games: int, output_file: str = "games.pth"):
    games = []
    print(f"Generating {num_games} games.")
    for i in range(num_games):
        if i % 100 == 0:
            print(f"Generated {i}/{num_games} games...")
        game = generate_random_game()
        games.append(game)
    print(f"Done! Generated {num_games} games!")

    print("Converting data to tensors.")
    positions, valid_moves = convert_to_tensors(games)

    print(f"Saving to output file. Shape:")
    print(f"positions : {positions.size()}")
    print(f"moves     : {valid_moves.size()}")
    print()

    save_to_file(positions, valid_moves, output_file)


def generate_random_game() -> List[Tuple[chess.Board, List[chess.Move]]]:
    board = chess.Board()
    history = []
    while not board.is_game_over():
        valid_moves = list(board.generate_legal_moves())
        random_move = random.choice(valid_moves)
        history.append((chess.Board(board.fen()), valid_moves))
        board.push(random_move)
    return history


def convert_to_tensors(
        games: List[List[Tuple[chess.Board, List[chess.Move]]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_positions = []
    all_valid_moves = []
    for game in games:
        for position, valid_moves in game:
            board_tensor = board_to_tensor(position)
            moves_tensor = moves_to_tensor(valid_moves)
            all_positions.append(board_tensor)
            all_valid_moves.append(moves_tensor)
    positions = torch.stack(all_positions)
    valid_moves = torch.stack(all_valid_moves)
    return positions, valid_moves


def save_to_file(positions, moves, filename='games.pth'):
    with open(filename, 'wb') as datafile:
        torch.save({"positions": positions, "moves": moves}, datafile)


if __name__ == "__main__":
    # Check if games.pth already exists
    if os.path.exists("games.pth"):
        print("games.pth already exists. Skipping generation.")
        print("Delete games.pth if you want to regenerate the dataset.")
    else:
        print("Generating training data...")
        generate_games(1000)
        print("Training data generation complete!")
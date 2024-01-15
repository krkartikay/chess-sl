import chess
import random
import torch

from chess_utils import board_to_tensor, moves_to_tensor

from typing import List, Tuple

NUM_GAMES = 100


def main():
    games = []
    for i in range(NUM_GAMES):
        print(f"Generating game {i+1}.")
        game = generate_random_game()
        games.append(game)
    print(f"Done! Generated {NUM_GAMES} games!")

    print("Converting data to tensors.")
    positions, valid_moves = convert_to_tensors(games)

    print(f"Saving to output file.")
    save_to_file(positions, valid_moves)


def generate_random_game() -> List[Tuple[chess.Board, List[chess.Move]]]:
    board = chess.Board()
    history = []
    while not board.is_game_over():
        # print(board)
        valid_moves = list(board.generate_legal_moves())
        # print(valid_moves)
        random_move = random.choice(valid_moves)
        history.append((board, valid_moves))
        board.push(random_move)
    return history


def convert_to_tensors(
        games: List[Tuple[chess.Board, List[chess.Move]]]) -> torch.Tensor:
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
    main()

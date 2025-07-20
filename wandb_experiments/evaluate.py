import torch
import chess
import random

from typing import Tuple, List
from model import ChessModel
from agent import ChessAgent, ChessModelAgent, RandomChessAgent

def evaluate_model_vs_random(model: ChessModel,
                             num_games: int = 100) -> Tuple[float, float, List[int]]:
    # Evaluates the given model by playing it against RandomAgent and returns score
    moves_played: List[int] = []
    total_score = 0
    agent = ChessModelAgent(model)
    other = RandomChessAgent()
    print(f"Evaluating model...")
    for i in range(num_games):
        if i % 100 == 0:
            print(f"Game {i+1}/{num_games}")
        (moves, score, draws, other_score,
            forfeit) = play_agent_vs_agent(agent, other)
        moves_played.append(moves)
        total_score += score
        total_score += draws / 2
    avg_moves_played = sum(moves_played) / len(moves_played)
    return (avg_moves_played, total_score, moves_played)

def evaluate_model_vs_model(model: ChessModel, other: ChessModel,
                            num_games: int = 100) -> Tuple[float, int, int, List[int]]:
    # Evaluates the given model by playing it against another Baseline model and returns score
    moves_played: List[int] = []
    total_score = 0
    total_draws = 0
    agent = ChessModelAgent(model)
    other_agent = ChessModelAgent(other)
    for i in range(num_games):
        (moves, score, draws, other_score,
            forfeit) = play_agent_vs_agent(agent, other_agent)
        moves_played.append(moves)
        total_score += score
        total_draws += draws
    avg_moves_played = sum(moves_played) / len(moves_played)
    return (avg_moves_played, total_score, total_draws, moves_played)

def play_agent_vs_agent(agent: ChessAgent, other: ChessAgent) -> Tuple[int, int, int, int, bool]:
    # Plays a game using the given model vs random-agent
    # and returns a tuple (moves_played, model_wins, draws, model_loses, forfeit)
    # If model predicts an invalid move, it forfeits the game.
    moves_played = 0
    board = chess.Board()
    r = random.randint(0, 1)
    while not board.is_game_over():
        current_agent = [agent, other][moves_played % 2 == r]
        move_or_none = current_agent.choose_move(board)
        if move_or_none is None:
            agent_score = -1 if (current_agent == agent) else +1
            other_agent_score = (- agent_score)
            return (moves_played, agent_score, 0, other_agent_score, True)
        board.push(move_or_none)
        moves_played += 1
    # When game is terminated
    if board.is_checkmate():
        agent_score = +1 if (moves_played % 2 == r) else -1
        other_agent_score = (- agent_score)
        return (moves_played, agent_score, 0, other_agent_score, False)
    # The game ended in draw
    return (moves_played, 0, 1, 0, False)

def calculate_precision(move_probs: torch.Tensor, valid_moves: torch.Tensor, threshold: float = 0.5):
    """
    Calculates Precision given predicted move_probs and actual valid_moves.

    Precision = True positives / Predicted Positives
              = 1 - False Positives / Predicted Positives
              = 1 - False Discovery Rate (FDR)
              = True Positive Rate
    """
    predicted_positives = move_probs >= threshold
    true_positives = predicted_positives & valid_moves.bool()
    return (true_positives.sum() / predicted_positives.sum()).item()

def calculate_recall(move_probs: torch.Tensor, valid_moves: torch.Tensor, threshold: float = 0.5):
    """
    Calculates Recall given predicted move_probs and actual valid_moves.

    Recall = True positives / Actual Positives
           = True Positive Rate (TPR)
           = 1 - False Negatives / Actual Positives
           = 1 - False Negative Rate (FNR, Miss Rate)
    """
    predicted_positives = move_probs >= threshold
    true_positives = predicted_positives & valid_moves.bool()
    return (true_positives.sum() / valid_moves.sum()).item()


if __name__ == "__main__":
    # load and evaluate model vs random agent
    import model
    default_config = {
        'n_blocks': 8,
        'n_channels': 128,
        'n_hidden': 4096,
        'batch_size': 256,
        'num_epochs': 20,
        'optimizer': 'ADAM',
        'learning_rate_multiplier': 3.0
    }
    chess_model = model.ChessModel(n_blocks=default_config['n_blocks'],
                                   n_channels=default_config['n_channels'],
                                   n_hidden=default_config['n_hidden']).to('cuda' if torch.cuda.is_available() else 'cpu')
    chess_model.load_state_dict(torch.load("model.pth"))
    chess_model.eval()
    print("Evaluating model vs random agent...")
    avg_moves, score, all_moves = evaluate_model_vs_random(chess_model, num_games=1000)
    print(f"Average moves played: {avg_moves}, Score: {score}")
    print(f"All moves played: {all_moves}")

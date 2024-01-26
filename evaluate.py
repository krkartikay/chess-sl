import torch
import chess
import random

from typing import Tuple

from agent import ChessAgent

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

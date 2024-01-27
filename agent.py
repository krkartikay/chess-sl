import chess
import random
import torch

from model import ChessModel
from chess_utils import board_to_tensor, action_to_move

class ChessAgent:
    def choose_move(self, position: chess.Board) -> chess.Move | None:
        raise NotImplementedError

class RandomChessAgent(ChessAgent):
    def choose_move(self, position: chess.Board) -> chess.Move:
        return random.choice(list(position.legal_moves))

class ChessModelAgent(ChessAgent):
    def __init__(self, model: ChessModel):
        self.model = model
        self.model.eval()

    def choose_move(self, position: chess.Board) -> chess.Move | None:
        # Neural net interface here.
        # Note: TODO: we've got to make this as fast as possible.
        # Batching? Inference Server? Ideas?
        position_tensor = board_to_tensor(position).unsqueeze(0)
        position_tensor = position_tensor.to(self.model.device())
        probs = self.model(position_tensor)
        if probs.sum() <= 0:
            # This sometimes happens if the model is not trained properly
            # print(position)
            # print(probs)
            return None
        sampled_action = int(torch.multinomial(probs, 1).item())
        move = action_to_move(sampled_action, position)
        # print(move, "%.2f" % probs[0, sampled_action])
        return move if move in position.legal_moves else None

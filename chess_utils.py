import chess
import torch

from typing import List


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((7, 8, 8))
    # Iterate over the 6 piece types and add them to the appropriate plane
    # in the return tensor.
    for layer in range(1, 7):
        pt = chess.PieceType(layer)
        # add +1 for all the white pieces and -1 for black pieces
        bitboard_white = board.pieces_mask(pt, chess.WHITE)
        bitboard_black = board.pieces_mask(pt, chess.BLACK)
        for sq in chess.SQUARES:
            if bitboard_white & (1 << sq):
                row, col = divmod(sq, 8)
                tensor[layer, row, col] += 1
            if bitboard_black & (1 << sq):
                row, col = divmod(sq, 8)
                tensor[layer, row, col] -= 1
    # fill in the last layer according with +/- 1 based on whose turn it is
    if board.turn == chess.WHITE:
        tensor[0, :, :] += 1
    else:
        tensor[0, :, :] -= 1
    return tensor

def tensor_to_board(tensor: torch.Tensor) -> chess.Board:
    b = chess.Board.empty()
    for p in chess.PIECE_TYPES:
        for sq in chess.SQUARES:
            row, col = divmod(sq, 8)
            pc = tensor[p, row, col]
            if pc.item() == 1:
                b.set_piece_at(sq, chess.Piece(p, chess.WHITE))
            elif pc.item() == -1:
                b.set_piece_at(sq, chess.Piece(p, chess.BLACK))
    b.turn = (tensor[0,0,0].item() == 1)
    return b

def move_to_action(move: chess.Move) -> int:
    a = move.from_square
    b = move.to_square
    idx = (a * 64) + b
    return idx


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    a, b = divmod(action, 64)
    move = chess.Move(a, b)

    # check for possible promotion
    if (chess.square_rank(b) == (7 if board.turn == chess.WHITE else 0)
            and board.piece_type_at(a) == chess.PAWN):
        move = chess.Move(a, b, chess.QUEEN)

    return move


def moves_to_tensor(moves: List[chess.Move]) -> torch.Tensor:
    moves_tensor = torch.zeros(64*64)
    valid_actions = [move_to_action(move) for move in moves]
    moves_tensor[valid_actions] = 1
    return moves_tensor

import chess
import chess_utils


def test_board_to_tensor():
    board = chess.Board()
    tensor = chess_utils.board_to_tensor(board)
    print(tensor)

    # tensor has correct shape
    assert tensor.size() == (7, 8, 8)

    # first turn is white
    assert tensor[0, 0, 0] == 1
    assert (tensor[0] == 1).all()

    # board has equal number of pieces at the start
    for i in range(1, 7):
        assert tensor[i].sum() == 0


def test_move_encoding():
    init_board = chess.Board()
    first_move = chess.Move.from_uci('e2e4')
    action = chess_utils.move_to_action(first_move)
    print(action)

    assert type(action) == int
    assert 0 <= action < 64*64

    move = chess_utils.action_to_move(action, init_board)
    print(move)

    assert first_move == move


def test_move_encoding_castle():
    init_board = chess.Board(  # white should be able to castle now
        'r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1')
    first_move = chess.Move.from_uci('e1g1')
    action = chess_utils.move_to_action(first_move)
    print(action)

    assert type(action) == int
    assert 0 <= action < 64*64

    move = chess_utils.action_to_move(action, init_board)
    print(move)

    assert first_move == move


def test_move_encoding_promotion():
    # white can promote the pawn in this position
    init_board = chess.Board('3k4/1P6/8/8/8/8/8/1K6 w - - 0 1')
    first_move = chess.Move.from_uci('b7b8q')
    action = chess_utils.move_to_action(first_move)
    print(action)

    assert type(action) == int
    assert 0 <= action < 64*64

    move = chess_utils.action_to_move(action, init_board)
    print(move)

    assert first_move == move

import chess
from ai import evaluate_board, select_greedy_move

def test_evaluate_board_starting_position():
    board = chess.Board()
    assert evaluate_board(board) == 0

def test_evaluate_board_material_difference():
    # White has an extra rook (5 points)
    board = chess.Board("8/8/8/8/8/8/8/R7 w - - 0 1")
    assert evaluate_board(board) == 5

def test_select_greedy_move_is_legal():
    board = chess.Board()
    move = select_greedy_move(board)
    assert move in board.legal_moves

def test_select_greedy_move_prefers_capture():
    # White rook on e2, black pawn on e1 → must capture e2e1
    board = chess.Board("8/8/8/8/8/8/4R3/4p3 w - - 0 1")
    move = select_greedy_move(board)
    assert move == chess.Move.from_uci("e2e1")

def test_select_greedy_move_none_when_no_moves():
    # Empty board → no legal moves
    board = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
    assert select_greedy_move(board) is None

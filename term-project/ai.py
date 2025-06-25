# ai.py

import chess
import random

# 1) simple material eval
VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0   
}

def evaluate_board(board: chess.Board) -> float:
    """Material balance from White’pgns POV (White minus Black)."""
    score = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = VALUES[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    # if you’re playing Black, the AI will invert this later
    return score

def select_greedy_move(board: chess.Board) -> chess.Move | None:
    """Pick the legal move that maximizes material balance for current side."""
    best_moves = []
    best_score = None

    for move in board.legal_moves:
        board.push(move)
        score = evaluate_board(board)
        board.pop()

        # If Black to move, flip perspective
        if board.turn == chess.WHITE:
            score = score  
        else:
            score = -score

        if best_score is None or score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    return random.choice(best_moves) if best_moves else None

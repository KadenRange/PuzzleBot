import chess

def get_game_status(board: chess.Board) -> str | None:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    term = outcome.termination
    if term == chess.Termination.CHECKMATE:
        winner = 'White' if outcome.winner else 'Black'
        return f"Checkmate! {winner} wins."
    elif term == chess.Termination.STALEMATE:
        return "Stalemate! Draw."
    elif term == chess.Termination.INSUFFICIENT_MATERIAL:
        return "Insufficient material! Draw."
    elif term in (chess.Termination.THREEFOLD_REPETITION,
                  chess.Termination.FIVEFOLD_REPETITION):
        return "Repetition! Draw."
    elif term == chess.Termination.SEVENTYFIVE_MOVES:
        return "75‑move rule! Draw."
    else:
        return "Draw."

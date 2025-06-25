# load_pgn/categorizer.py
import chess.engine

def score_to_centipawns(score):
    """
    Converts a PovScore to an integer in centipawns.
    Mate scores become a large cp value.
    """
    if score.is_mate():
        # mate in N → large positive (winning) or negative (losing)
        n = score.mate()
        return 100000 - abs(n) if n > 0 else -100000 + abs(n)
    else:
        # cp is accessible via .score()
        return score.score()

def categorize_move(delta, score_best, score_played):
    """
    Classify the type of error based on delta and mate info.
    """
    # missed mate: best is mate but played is not
    if score_best.is_mate() and not score_played.is_mate():
        return "missed_mate"
    # big swing = blunder
    if delta > 200:
        return "blunder"
    # smaller but nontrivial = inaccuracy
    if delta > 50:
        return "inaccuracy"
    return "none"

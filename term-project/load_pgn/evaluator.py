# load_pgn/evaluator.py

import chess
import chess.engine
from load_pgn.config import ENGINE_PATH, ENGINE_DEPTH

class EngineEvaluator:
    """
    Wraps a Stockfish UCI engine.
    .evaluate(fen, played_move_uci) → (best_move_uci, score_best, score_played)
    """

    def __init__(self, path=ENGINE_PATH, depth=ENGINE_DEPTH):
        # Launch Stockfish
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth  = depth

    def evaluate(self, fen: str, played_move_uci: str):
        board = chess.Board(fen)
        orig_turn = board.turn  # ← save who’s to move

        # 1) Stockfish’s best move
        best_info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        best_move = best_info["pv"][0].uci()
        score_best = best_info["score"].pov(orig_turn)  # ← use orig_turn

        # 2) Force the user’s move and re‐evaluate
        board.push_uci(played_move_uci)
        played_info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score_played = played_info["score"].pov(orig_turn)  # ← same orig_turn

        return best_move, score_best, score_played

    def close(self):
        self.engine.quit()

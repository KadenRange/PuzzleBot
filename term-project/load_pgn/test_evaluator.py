# test_evaluator.py

from evaluator import EngineEvaluator

# A simple starting‐position test
fen    = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
played = "e7e5"

engine = EngineEvaluator()
best, score_best, score_played = engine.evaluate(fen, played)

print("Position FEN:  ", fen)
print("Played move:  ", played)
print("Stockfish’s best:", best)
print("Score(best):   ", score_best)     # PovScore, e.g. PovScore(cp=20)
print("Score(played): ", score_played)   # Should match score(best) if it’s actually best

engine.close()

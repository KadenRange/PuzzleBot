import os

HERE = os.path.dirname(__file__)          # path to load_pgn/
PGN_FOLDER = os.path.join(HERE, 'pgns')   # now points at load_pgn/pgns

ENGINE_PATH = "stockfish"
ENGINE_DEPTH = 15

OUTPUT_CSV = os.path.join(HERE, '..', 'outputs', 'blunder_results.csv')
PUZZLE_DB_CSV = os.path.join(HERE, '..', 'data', 'handled_data.csv')

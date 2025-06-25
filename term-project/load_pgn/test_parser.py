from config import PGN_FOLDER
from pgn_parser import parse_pgns

print("Testing PGN ingestion…")
gen = parse_pgns(PGN_FOLDER)

# Print the first 5 (FEN, move) pairs
for i in range(5):
    fen, move = next(gen)
    print(f"{i+1:2d}) {fen} → {move}")

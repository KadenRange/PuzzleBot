# track_errors.py
import pandas as pd
import os
from load_pgn.config       import PGN_FOLDER, OUTPUT_CSV
from load_pgn.pgn_parser   import parse_pgns
from load_pgn.evaluator    import EngineEvaluator
from load_pgn.categorizer  import score_to_centipawns, categorize_move

def analyze_pgn_file(pgn_file_path):
    """Analyze a single PGN file and identify tactical errors"""
    engine = EngineEvaluator()
    rows = []
    
    # Keep track of processed positions to avoid duplicates
    processed_positions = set()
    
    # Use a list with one file to handle the parse_pgns function that accepts either a folder or list
    pgn_files = [pgn_file_path]
            
    print(f"Analyzing PGN file: {pgn_file_path}")

    for fen, played_move in parse_pgns(pgn_files):
        # Skip already processed positions
        position_key = f"{fen}:{played_move}"
        if position_key in processed_positions:
            continue
        processed_positions.add(position_key)
        
        best_move, sb, sp = engine.evaluate(fen, played_move)
        cp_best   = score_to_centipawns(sb)
        cp_played = score_to_centipawns(sp)
        delta     = cp_best - cp_played
        category  = categorize_move(delta, sb, sp)

        # only keep actual "errors"
        if category in ('blunder', 'inaccuracy', 'missed_mate'):
            rows.append({
                'fen': fen,
                'played_move': played_move,
                'best_move': best_move,
                'category': category
            })

    engine.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} errors to {OUTPUT_CSV}")
    return df

def main():
    engine = EngineEvaluator()
    rows = []
    
    # Keep track of processed positions to avoid duplicates
    processed_positions = set()
    
    # Get all PGN files but process each only once
    pgn_files = []
    for filename in os.listdir(PGN_FOLDER):
        if filename.endswith('.pgn'):
            pgn_path = os.path.join(PGN_FOLDER, filename)
            pgn_files.append(pgn_path)
            
    print(f"Processing {len(pgn_files)} PGN file(s)...")

    for fen, played_move in parse_pgns(pgn_files):
        # Skip already processed positions
        position_key = f"{fen}:{played_move}"
        if position_key in processed_positions:
            continue
        processed_positions.add(position_key)
        
        best_move, sb, sp = engine.evaluate(fen, played_move)
        cp_best   = score_to_centipawns(sb)
        cp_played = score_to_centipawns(sp)
        delta     = cp_best - cp_played
        category  = categorize_move(delta, sb, sp)

        # only keep actual "errors"
        if category in ('blunder', 'inaccuracy', 'missed_mate'):
            rows.append({
                'fen': fen,
                'played_move': played_move,
                'best_move': best_move,
                # delta removed to ensure compatibility with 26-channel model
                'category': category
            })

    engine.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} errors to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

import os
import chess.pgn

def parse_pgns(path_or_files):
    """
    Iterate through PGN files and yield (FEN, played_move_uci).
    
    Args:
        path_or_files: Either a folder path (string) or a list of file paths
    """
    # Handle both folder path and list of files
    files_to_process = []
    
    if isinstance(path_or_files, list):
        # We were given a list of files
        files_to_process = [f for f in path_or_files if str(f).lower().endswith('.pgn')]
    else:
        # We were given a folder path
        folder_path = path_or_files
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith('.pgn'):
                continue
            filepath = os.path.join(folder_path, filename)
            files_to_process.append(filepath)
    
    # Process each file
    for filepath in files_to_process:
        try:
            with open(filepath, 'r') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        fen = board.fen()
                        move_uci = move.uci()
                        board.push(move)
                        yield fen, move_uci
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

import pygame
import chess
import random
import argparse
import pandas as pd
import os
import tempfile
from ai import select_greedy_move
from utils import get_game_status
from load_pgn.resnet.classifier import get_drills
from load_pgn.config import PGN_FOLDER, OUTPUT_CSV
from load_pgn.pgn_parser import parse_pgns
from track_errors import analyze_pgn_file, main as track_errors_main

# Constants
WIDTH, HEIGHT = 480, 480
SQUARE_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# Load piece images (named wP.png, bP.png, etc.)
def load_pieces():
    pieces = {}
    for color in ['w', 'b']:
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            try:
                img = pygame.image.load(f"pieces/{color.lower()}{piece.lower()}.png")
                pieces[color + piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
            except pygame.error:
                print(f"Warning: Could not load image for {color}{piece}")
    return pieces

def draw_board(screen, board, pieces):
    # Draw board squares
    for r in range(8):
        for c in range(8):
            clr = WHITE if (r + c) % 2 == 0 else BROWN
            pygame.draw.rect(screen, clr,
                             (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    # Draw pieces
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            row = 7 - (sq // 8)
            col = sq % 8
            key = ('w' if p.symbol().isupper() else 'b') + p.symbol().upper()
            if key in pieces:
                screen.blit(pieces[key], (col*SQUARE_SIZE, row*SQUARE_SIZE))

def save_pgn_text_to_file(pgn_text):
    """Save PGN text input to a temporary file and return the path"""
    # Create the temp file with .pgn extension
    fd, temp_path = tempfile.mkstemp(suffix=".pgn")
    try:
        with os.fdopen(fd, 'w') as temp_file:
            temp_file.write(pgn_text)
        return temp_path
    except Exception as e:
        print(f"Error saving PGN to temp file: {e}")
        os.unlink(temp_path)  # Clean up the file
        return None

def process_pgn_and_get_drills(pgn_path, user_rating, history_ids=None, max_puzzles=10):
    """Process a PGN file, find errors, classify them, and get recommended puzzles"""
    print(f"Starting analysis of PGN file: {pgn_path}")
    
    # Step 1: Track errors (this writes to OUTPUT_CSV)
    print("Analyzing moves for errors...")
    analyze_pgn_file(pgn_path)  # Use analyze_pgn_file instead of track_errors_main
    
    # Step 2: Classify errors and find appropriate puzzles
    print(f"Classifying errors and selecting puzzles for {user_rating} ELO...")
    drills_df = get_drills(user_rating=user_rating, history_ids=history_ids)
    
    # Limit to max_puzzles (default 10)
    if len(drills_df) > max_puzzles:
        print(f"Limiting to {max_puzzles} most significant puzzles")
        # Prioritize blunders over inaccuracies
        blunders = drills_df[drills_df['category'] == 'blunder']
        others = drills_df[drills_df['category'] != 'blunder']
        # Take blunders first, then fill up to max_puzzles with others
        if len(blunders) >= max_puzzles:
            drills_df = blunders.head(max_puzzles)
        else:
            drills_df = pd.concat([blunders, others.head(max_puzzles - len(blunders))])
    
    print(f"Found {len(drills_df)} tactical errors with recommended puzzles")
    return drills_df

def display_puzzle_gui(puzzle_fen, puzzle_moves, puzzle_theme):
    """Display a puzzle in a GUI for solving"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Puzzle")
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 20)
    pieces = load_pieces()
    
    board = chess.Board(puzzle_fen)
    solution = puzzle_moves.split()
    
    # Safety check - make sure solution is not empty
    if not solution:
        print(f"Error: Empty solution for puzzle with FEN: {puzzle_fen}")
        return
    
    # The bot makes the first move, then the user responds
    puzzle_index = 0
    
    # Determine player's color based on the second move in the puzzle
    # If first move is white, and user plays second, user is black
    bot_plays_white = board.turn == chess.WHITE
    player_is_white = not bot_plays_white
    player_color = "White" if player_is_white else "Black"
    
    selected = None
    running = True
    game_over = False
    result = ""
    retry_available = True
    
    # Bot makes the first move automatically
    if puzzle_index < len(solution) and not game_over:
        try:
            # Make the bot's first move from the solution
            bot_move = chess.Move.from_uci(solution[puzzle_index])
            board.push(bot_move)
            puzzle_index += 1
            print(f"Bot played {bot_move.uci()}")
        except Exception as e:
            print(f"Error with bot's first move: {e}")
            result = "Puzzle error: Invalid starting move"
            game_over = True

    while running:
        draw_board(screen, board, pieces)
        
        # Show puzzle information
        theme_txt = font.render(f"Theme: {puzzle_theme}", True, (0, 0, 0))
        color_txt = font.render(f"You play as: {player_color}", True, (0, 0, 0))
        
        # Position text at top corners
        screen.blit(theme_txt, (10, 10))
        screen.blit(color_txt, (10, 38))
        
        if board.turn == chess.WHITE:
            turn_txt = small_font.render("White to move", True, (0, 0, 0))
        else:
            turn_txt = small_font.render("Black to move", True, (0, 0, 0))
        screen.blit(turn_txt, (WIDTH - 120, 10))
        
        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))
            
            # Show result and instructions
            txt = font.render(result, True, (255, 255, 255))
            rect = txt.get_rect(center=(WIDTH//2, HEIGHT//2 - 20))
            screen.blit(txt, rect)
            
            if "Wrong" in result and retry_available:
                hint_txt = small_font.render("Press 'R' to retry or SPACE for next puzzle", True, (255, 255, 255))
            else:
                hint_txt = small_font.render("Press SPACE for next puzzle", True, (255, 255, 255))
            
            hint_rect = hint_txt.get_rect(center=(WIDTH//2, HEIGHT//2 + 20))
            screen.blit(hint_txt, hint_rect)
        
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_SPACE and game_over:
                    return
                # Reset the board to retry the puzzle
                if ev.key == pygame.K_r and game_over and "Wrong" in result and retry_available:
                    board = chess.Board(puzzle_fen)
                    puzzle_index = 0
                    game_over = False
                    selected = None
                    
                    # Bot makes the first move again when we retry
                    if puzzle_index < len(solution) and not game_over:
                        try:
                            # Make the bot's first move from the solution
                            bot_move = chess.Move.from_uci(solution[puzzle_index])
                            board.push(bot_move)
                            puzzle_index += 1
                            print(f"Bot played {bot_move.uci()}")
                        except Exception as e:
                            print(f"Error with bot's first move: {e}")
                            result = "Puzzle error: Invalid starting move"
                            game_over = True
                    
            if ev.type == pygame.MOUSEBUTTONDOWN and not game_over:
                x, y = ev.pos
                col = x // SQUARE_SIZE
                row = 7 - (y // SQUARE_SIZE)
                sq = chess.square(col, row)
                
                if selected is None:
                    # Selecting a piece
                    if board.piece_at(sq) and board.piece_at(sq).color == board.turn:
                        selected = sq
                else:
                    # Making a move
                    move = chess.Move(selected, sq)
                    
                    # Handle promotions
                    if selected is not None and board.piece_at(selected) and \
                       board.piece_at(selected).piece_type == chess.PAWN:
                        if (sq >= 56 and board.turn == chess.WHITE) or (sq <= 7 and board.turn == chess.BLACK):
                            move = chess.Move(selected, sq, promotion=chess.QUEEN)
                    
                    if move in board.legal_moves:
                        # Make the user's move
                        board.push(move)
                        
                        # Check if user's move matches solution
                        if puzzle_index < len(solution):
                            expect = solution[puzzle_index]
                            if move.uci() == expect:
                                # Correct move - increment the puzzle index
                                puzzle_index += 1
                                
                                # Check if puzzle is solved
                                if puzzle_index >= len(solution):
                                    result = "Solved!"
                                    game_over = True
                                # Make bot's next move if there are more moves in the solution
                                elif puzzle_index < len(solution):
                                    try:
                                        bot_move = chess.Move.from_uci(solution[puzzle_index])
                                        board.push(bot_move)
                                        puzzle_index += 1
                                        
                                        # Check if we've reached the end after bot's move
                                        if puzzle_index >= len(solution):
                                            result = "Solved!"
                                            game_over = True
                                    except (ValueError, IndexError, chess.IllegalMoveError) as e:
                                        print(f"Error with bot move: {e}")
                                        result = "Puzzle error: Invalid bot move"
                                        game_over = True
                            else:
                                # Wrong move
                                result = f"Wrong move! Expected {expect}"
                                game_over = True
                        else:
                            result = "Solved!"
                            game_over = True
                            
                    selected = None

    pygame.quit()


def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Chess tactics trainer')
    parser.add_argument('--pgn', type=str, help='Path to PGN file to analyze')
    parser.add_argument('--pgn-text', type=str, help='Raw PGN text to analyze')
    parser.add_argument('--rating', type=int, help='User ELO rating')
    parser.add_argument('--gui', action='store_true', help='Launch GUI for puzzle solving')
    args = parser.parse_args()

    # Interactive mode if arguments aren't provided
    pgn_path = args.pgn
    pgn_text = args.pgn_text
    user_rating = args.rating
    
    if not (pgn_path or pgn_text) or not user_rating:
        print("Welcome to Chess Tactics Trainer!")
        print("================================")
        
        if not (pgn_path or pgn_text):
            choice = input("Would you like to (1) provide a path to a PGN file or (2) paste PGN text? (1/2): ").strip()
            if choice == "1":
                pgn_path = input("Enter path to your PGN file: ")
            else:
                print("Please paste your PGN text below (press Ctrl+D on Unix or Ctrl+Z followed by Enter on Windows when done):")
                pgn_lines = []
                while True:
                    try:
                        line = input()
                        pgn_lines.append(line)
                    except EOFError:
                        break
                pgn_text = "\n".join(pgn_lines)
                
                if not pgn_text.strip():
                    print("No PGN text provided. Exiting.")
                    return
        
        if not user_rating:
            try:
                user_rating = int(input("Enter your ELO rating: "))
            except ValueError:
                print("Invalid rating. Using default 1500.")
                user_rating = 1500
    
    # If we have PGN text but no file path, save the text to a temporary file
    if pgn_text and not pgn_path:
        pgn_path = save_pgn_text_to_file(pgn_text)
        if not pgn_path:
            print("Error: Could not create temporary PGN file.")
            return
    
    # Process PGN and get drill recommendations
    drills_df = process_pgn_and_get_drills(pgn_path, user_rating)
    
    # Clean up temporary file if we created one
    if pgn_text and args.pgn is None:
        try:
            os.unlink(pgn_path)
        except:
            pass  # Ignore errors in cleanup
    
    # Display the recommended puzzles
    print("\nRecommended Puzzle Exercises:")
    print("============================")
    
    for i, row in drills_df.iterrows():
        print(f"Puzzle {i+1}:")
        print(f"  Original position from game: {row['fen']}")
        print(f"  Move played: {row['played_move']}")
        print(f"  Best move would be: {row['best_move']}")
        print(f"  Tactical motif: {row['motif']}")
        print(f"  Recommended puzzle: {row['puzzle_fen']}")
        print(f"  Puzzle solution: {row['puzzle_moves']}")
        print("")
    
    # Save the drills to a CSV
    output_path = "outputs/user_drills.csv"
    drills_df.to_csv(output_path, index=False)
    print(f"Saved {len(drills_df)} drills to {output_path}")
    
    # Launch GUI if requested
    if args.gui or input("Would you like to solve these puzzles in a GUI? (y/n): ").strip().lower() == 'y':
        print("Starting puzzle GUI. Press ESC to quit, SPACE to advance to next puzzle.")
        
        # Get the actual number of puzzles
        num_puzzles = len(drills_df)
        max_display = min(num_puzzles, 10)  # Cap at 10 puzzles
        
        for i, row in enumerate(drills_df.iterrows(), 1):
            if i > max_display:
                break
                
            _, row = row  # Unpack the row tuple
            print(f"\nDisplaying puzzle {i}/{max_display}")
            display_puzzle_gui(row['puzzle_fen'], row['puzzle_moves'], row['motif'])


if __name__ == "__main__":
    main()

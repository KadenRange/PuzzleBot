# load_pgn/cnn/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
import random

def fen_to_tensor(fen: str, last_move_uci: str = None, eval_delta: float = 0.0) -> torch.FloatTensor:
    """
    26-channel tensor:
      - 12 piece planes
      - 1 side-to-move
      - 2 last-move src/dst masks
      - 4 castling rights
      - 1 en passant possibility
      - 1 move generation features (attack/defense count)
      - 1 check indicator
      - 1 piece values under attack (material tension)
      - 1 pin detection
      - 2 distance to kings
    """
    board = chess.Board(fen)
    tensor = torch.zeros(26, 8, 8, dtype=torch.float32)

    # 12 piece planes
    mapping = {
        'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
        'p':6,'n':7,'b':8,'r':9,'q':10,'k':11
    }
    for sq, piece in board.piece_map().items():
        idx = mapping[piece.symbol()]
        row = 7 - (sq // 8)
        col = sq % 8
        tensor[idx, row, col] = 1.0

    # side-to-move plane
    tensor[12, :, :] = float(board.turn)

    # last-move masks
    if last_move_uci:
        uci = last_move_uci[:4]
        src_sq, dst_sq = uci[:2], uci[2:4]
        src_idx = chess.parse_square(src_sq)
        dst_idx = chess.parse_square(dst_sq)
        src_row, src_col = 7 - (src_idx // 8), src_idx % 8
        dst_row, dst_col = 7 - (dst_idx // 8), dst_idx % 8
        tensor[13, src_row, src_col] = 1.0
        tensor[14, dst_row, dst_col] = 1.0
    
    # Add castling rights (4 planes)
    tensor[15, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[16, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[17, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[18, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Add en passant possibility
    if board.ep_square:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[19, ep_row, ep_col] = 1.0
    
    # Add attack/defense counts
    for sq in range(64):
        row, col = 7 - (sq // 8), sq % 8
        attackers = len(board.attackers(board.turn, sq))
        defenders = len(board.attackers(not board.turn, sq))
        tensor[20, row, col] = min(1.0, (attackers - defenders) / 3.0 + 0.5)  # normalize to [0,1]
    
    # Check indicator
    tensor[21, :, :] = float(board.is_check())
    
    # NEW FEATURES
    
    # Piece values under attack (material tension)
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
    for sq, piece in board.piece_map().items():
        row, col = 7 - (sq // 8), sq % 8
        attackers = board.attackers(not piece.color, sq)
        if attackers:
            value = piece_values[piece.symbol().lower()]
            tensor[22, row, col] = min(1.0, value / 9.0)  # Normalize by queen value
    
    # Pin detection
    for sq in chess.SQUARES:
        if board.piece_at(sq):
            row, col = 7 - (sq // 8), sq % 8
            # Check if piece is pinned
            piece_color = board.piece_at(sq).color
            king_sq = board.king(piece_color)
            if king_sq is not None and board.is_pinned(piece_color, sq):
                tensor[23, row, col] = 1.0
    
    # Distance to kings
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is not None and black_king_sq is not None:
        wk_row, wk_col = 7 - (white_king_sq // 8), white_king_sq % 8
        bk_row, bk_col = 7 - (black_king_sq // 8), black_king_sq % 8
        
        for r in range(8):
            for c in range(8):
                # Distance to white king (normalized)
                w_dist = max(abs(r - wk_row), abs(c - wk_col)) / 7.0
                tensor[24, r, c] = 1.0 - w_dist  # Closer = higher value
                
                # Distance to black king (normalized)
                b_dist = max(abs(r - bk_row), abs(c - bk_col)) / 7.0
                tensor[25, r, c] = 1.0 - b_dist  # Closer = higher value

    return tensor

class TacticDataset(Dataset):
    """
    Expects a DataFrame or CSV with columns:
      - FEN
      - Moves (space-separated UCI moves; first is the motif move)
      - simple_label
      - eval_delta (evaluation delta value)
    """
    def __init__(self, data, label_map: dict, augment: bool=False):
        # data: either path string or pandas.DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'simple_label':'motif'})

        self.fens = df['fen'].tolist()
        self.moves = df['moves'].tolist()
        self.labels = [label_map[m] for m in df['motif']]
        
        # Extract delta values if available, otherwise use zeros
        self.deltas = [0] * len(self.fens)
        if 'eval_delta' in df.columns:
            self.deltas = df['eval_delta'].fillna(0).tolist()
            
        self.augment = augment

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        move = self.moves[idx].split()[0]
        delta = self.deltas[idx]
        x = fen_to_tensor(fen, last_move_uci=move, eval_delta=delta)

        if self.augment:
            # Horizontal flip (more chess-legitimate)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
                
            # Color inversion (swap white/black pieces and update turn)
            if random.random() < 0.3:
                # Swap white and black piece planes
                white_pieces = x[0:6].clone()
                black_pieces = x[6:12].clone()
                x[0:6], x[6:12] = black_pieces, white_pieces
                # Flip side to move
                x[12] = 1.0 - x[12]
                # Update castling rights
                castle_w, castle_b = x[15:17].clone(), x[17:19].clone()
                x[15:17], x[17:19] = castle_b, castle_w
                
            # Subtle noise for robustness (controlled magnitude)
            if random.random() < 0.2:
                noise = torch.randn_like(x) * 0.02
                x = torch.clamp(x + noise, 0, 1)
                
            # Tactical feature emphasis (randomly boost certain features)
            if random.random() < 0.3:
                # Emphasize attacks/defenses or check
                feature_idx = random.choice([20, 21])
                x[feature_idx] = torch.clamp(x[feature_idx] * random.uniform(1.0, 1.5), 0, 1)

        return x, self.labels[idx]
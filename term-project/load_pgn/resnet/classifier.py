# load_pgn/cnn/classifier.py

import os
import torch
import pandas as pd
import argparse

from load_pgn.config import OUTPUT_CSV, PUZZLE_DB_CSV
from load_pgn.resnet.model import TacticsResNet
from load_pgn.resnet.dataset import fen_to_tensor

# 1) Your labels must match train.py
LABELS = ['backRankMate', 'deflection', 'discoveredAttack', 'fork', 'hangingPiece', 'mateIn2', 'other', 'pin', 'promotion', 'sacrifice', 'skewer']
label_map = {i: lbl for i, lbl in enumerate(LABELS)}

class PuzzlePicker:
    def __init__(self, puzzles_df, history_ids=None):
        # normalize column names
        puzzles_df.columns = puzzles_df.columns.str.lower()
        
        # Add a new column to identify if the first player is the attacker (executing the tactic)
        if 'is_attacker' not in puzzles_df.columns:
            # We'll assume attacking puzzles typically have a material gain or checkmate in the solution
            # This is a heuristic - in a real system, this would be labeled more precisely
            puzzles_df['is_attacker'] = True
            
        # group puzzles by simple_label (motif)
        self.by_motif = {
            m: puzzles_df[puzzles_df['simple_label'] == m]
            for m in LABELS
        }
        self.history = set(history_ids) if history_ids is not None else set()

    def pick(self, motif, user_rating):
        df = self.by_motif.get(motif, pd.DataFrame())
        if df.empty:
            # Fallback to 'other' category if no puzzles for this motif
            df = self.by_motif.get('other', pd.DataFrame())
            if df.empty:  # If still empty, try any available motif
                for alt_motif in LABELS:
                    df = self.by_motif.get(alt_motif, pd.DataFrame())
                    if not df.empty:
                        break
        
        # Prioritize puzzles where you're the attacker (executing the tactic)
        attacker_puzzles = df[df['is_attacker'] == True] if 'is_attacker' in df.columns else df
        if not attacker_puzzles.empty:
            df = attacker_puzzles
        
        # filter by rating ±400 if rating column exists
        if 'rating' in df.columns and not df.empty:
            band = df[(df['rating'] >= user_rating - 400) &
                      (df['rating'] <= user_rating + 400)]
        else:
            band = df
            
        # remove seen puzzles
        band = band[~band.index.isin(self.history)]
        pool = band if not band.empty else df
        
        if pool.empty:
            # Last resort: return a placeholder if no puzzles available
            return pd.Series({
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'moves': 'e2e4',
                'rating': user_rating,
                'simple_label': 'other',
                'is_attacker': True
            })
            
        chosen = pool.sample(1).iloc[0]
        self.history.add(chosen.name)
        return chosen

def predict_with_tta(model, fen, device):
    """Test-time augmentation for more reliable predictions"""
    t = fen_to_tensor(fen).unsqueeze(0).to(device)
    t_flipped = torch.flip(t, dims=[3])  # Horizontal flip
    
    with torch.no_grad():
        pred1 = model(t).softmax(dim=1)
        pred2 = model(t_flipped).softmax(dim=1)
        
    # Average predictions
    avg_pred = (pred1 + pred2) / 2
    return avg_pred.argmax(dim=1).item(), avg_pred.max().item()

def get_drills(user_rating, history_ids=None):
    """Runs classification + picking, returns DataFrame with errors and assigned puzzles."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with more robust path handling
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    model_path = os.path.join(model_dir, 'tactic_cnn_best.pth')
    
    model = TacticsResNet(
        num_classes=len(LABELS),
        in_channels=26,
        num_filters=128,
        num_blocks=8,
        drop=0.4
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # classify errors with test-time augmentation
    errors_df = pd.read_csv(OUTPUT_CSV)
    motifs = []
    confidences = []
    for fen in errors_df['fen']:
        pred_idx, confidence = predict_with_tta(model, fen, device)
        motifs.append(label_map[pred_idx])
        confidences.append(float(confidence))
    
    errors_df['motif'] = motifs
    errors_df['confidence'] = confidences

    # pick puzzles
    puzzles_df = pd.read_csv(PUZZLE_DB_CSV)
    picker = PuzzlePicker(puzzles_df, history_ids)

    picks = []
    for motif in errors_df['motif']:
        p = picker.pick(motif, user_rating)
        picks.append({
            'puzzle_fen':   p['fen'],
            'puzzle_moves': p['moves'],
            'puzzle_rating': p.get('rating', None),
            'puzzle_motif': p['simple_label']
        })
    puzzles_out = pd.DataFrame(picks)

    return pd.concat([errors_df.reset_index(drop=True), puzzles_out], axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify errors and pick puzzles based on user rating')
    parser.add_argument('--rating', '-r', type=int, required=True,
                        help='User Elo rating for filtering puzzles')
    parser.add_argument('--history', '-h', type=int, nargs='*', default=None,
                        help='List of puzzle indices already seen')
    args = parser.parse_args()

    drills = get_drills(user_rating=args.rating, history_ids=args.history)
    out_dir = os.path.dirname(OUTPUT_CSV) or '.'
    out_path = os.path.join(out_dir, 'user_drills.csv')
    drills.to_csv(out_path, index=False)
    print(f"Wrote {len(drills)} drills to {out_path}")

# load_pgn/cnn/train.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from load_pgn.config import PUZZLE_DB_CSV
from load_pgn.cnn.dataset import TacticDataset
from load_pgn.cnn.model import TacticsResNet  # Changed from ImprovedTacticCNN

def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to a batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Applies mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    """Focal Loss to focus more on difficult examples"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    # --- load & preprocess CSV
    df = pd.read_csv(PUZZLE_DB_CSV)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'simple_label':'motif'})

    # stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['motif'], random_state=42
    )
    
    # Create validation and test sets
    val_df, test_df = train_test_split(
        temp_df, test_size=0.33, stratify=temp_df['motif'], random_state=42
    )

    # build labels
    LABELS = sorted(df['motif'].unique())
    label_map = {lbl: i for i, lbl in enumerate(LABELS)}
    print(f"Labels ({len(LABELS)}): {LABELS}")
    
    # Calculate class weights for focal loss
    counts = df['motif'].value_counts()
    class_weights = torch.FloatTensor([counts.max() / counts[lbl] for lbl in LABELS])

    # --- samplers & datasets
    # class-balanced sampler for training
    weights = train_df['motif'].map(lambda m: 1.0/counts[m]).to_list()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = TacticDataset(train_df, label_map, augment=True)
    val_ds = TacticDataset(val_df, label_map, augment=False)
    test_ds = TacticDataset(test_df, label_map, augment=False)

    # Use a slightly smaller batch size for the deeper model
    batch_size = 48 
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- model, loss, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use TacticsResNet with 27 input channels (includes delta)
    model = TacticsResNet(num_classes=len(LABELS), in_channels=27, 
                         num_filters=128, num_blocks=8, drop=0.4).to(device)
    
    print(f"Using model: TacticsResNet with 27-channel input (including eval delta)")
    
    # Use either label smoothing or focal loss
    use_focal_loss = True
    if use_focal_loss:
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    # LR warmup + cosine annealing schedule with longer warmup for deeper model
    def lr_lambda(epoch):
        warmup_epochs = 8  # Increased from 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            # Cosine decay from max_lr to min_lr
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (50 - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0
    
    # Create an absolute path to the models directory to avoid any path issues
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'tactic_cnn_best.pth')
    model_final_path = os.path.join(model_dir, 'tactic_cnn_final.pth')
    
    print(f"Model will be saved to: {model_path}")
    
    # Early stopping setup with longer patience for complex model
    patience = 15  # Increased from 10
    early_stop_counter = 0
    best_val_loss = float('inf')

    # --- training loop
    for epoch in range(1, 71):  # More epochs for deeper model (increased from 51)
        # train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Apply mixup with 50% probability
            if np.random.random() < 0.5:
                X, y_a, y_b, lam = mixup_data(X, y)
                logits = model(X)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                
                # For accuracy calculation during mixup
                pred = logits.argmax(dim=1)
                correct += (lam * pred.eq(y_a).sum().float() + 
                           (1 - lam) * pred.eq(y_b).sum().float())
            else:
                logits = model(X)
                loss = criterion(logits, y)
                
                # Standard accuracy calculation
                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total += y.size(0)
            
        train_acc = correct / total
        avg_train = total_loss / len(train_loader)

        # validate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        avg_val = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch:02d}  Train Loss: {avg_train:.4f}  Train Acc: {train_acc:.3f}  Val Loss: {avg_val:.4f}  Val Acc: {val_acc:.3f}  LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"→ New best accuracy: {best_acc:.3f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * len(LABELS)
    class_total = [0] * len(LABELS)
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Per-class accuracy
            for i in range(y.size(0)):
                label = y[i].item()
                class_correct[label] += (predicted[i] == y[i]).item()
                class_total[label] += 1
    
    test_acc = correct / total
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Print per-class accuracies
    print("\nPer-class accuracies:")
    for i in range(len(LABELS)):
        if class_total[i] > 0:
            print(f"{LABELS[i]}: {100 * class_correct[i] / class_total[i]:.1f}%")
    
    # final save
    torch.save(model.state_dict(), model_final_path)
    print(f"Done. Best val_acc: {best_acc:.3f}, Final test_acc: {test_acc:.3f}")

if __name__ == '__main__':
    main()
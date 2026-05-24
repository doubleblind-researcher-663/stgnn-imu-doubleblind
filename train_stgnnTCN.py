"""
train_stgnnTCN.py

ST-GNN for missing IMU node prediction WITH Hyperparameter Optimization (Optuna).
X=(N,20,2,4), y=(N,4), A=(2,2)

Temporal module: TCN (Temporal Convolutional Network) with dilated causal convolutions.

# Usage example: python train_stgnnTCN.py --data dataset.npz --epochs 500 --batch 64 --lr 0.001 --n-trials 15 --save-dir ./my_experiments
"""

import argparse
import math
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import optuna

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class STGNNDataset(Dataset):
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# Adjacency & Loss
# ---------------------------------------------------------------------------

def normalize_adjacency(A):
    A = A.astype(np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A_hat.sum(axis=1)
    d = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    D = np.diag(d)
    return torch.from_numpy(D @ A_hat @ D).float()

def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target)

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, X, A_norm):
        B, T, V, _ = X.shape
        X_lin = self.linear(X)
        X_r   = X_lin.reshape(B * T, V, -1)
        A     = A_norm.to(X_r.device)
        out   = torch.bmm(A.unsqueeze(0).expand(B * T, -1, -1), X_r)
        return F.relu(out.reshape(B, T, V, -1))


class TCNBlock(nn.Module):
    """
    Single residual TCN block with dilated causal convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        out = F.pad(x, (self.pad, 0))          
        out = self.drop1(F.relu(self.bn1(self.conv1(out))))
        out = F.pad(out, (self.pad, 0))
        out = self.drop2(F.relu(self.bn2(self.conv2(out))))
        return F.relu(out + res)


class TCN(nn.Module):
    """
    Stack of TCNBlocks with exponentially increasing dilation.
    """
    def __init__(self, in_channels, tcn_hidden, n_layers, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation  = 2 ** i
            in_ch     = in_channels if i == 0 else tcn_hidden
            layers.append(TCNBlock(in_ch, tcn_hidden, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)           
        return out.mean(dim=-1)         


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN where the temporal module is a TCN.
    """
    def __init__(
        self, num_input_nodes, in_feats, gcn_hidden, tcn_hidden,
        tcn_layers, tcn_kernel_size, out_feats, n_gcn_layers=2, dropout=0.1,
    ):
        super().__init__()
        self.num_input_nodes = num_input_nodes
        self.gcn_hidden      = gcn_hidden
        self.tcn_hidden      = tcn_hidden
        self.out_feats       = out_feats

        self.gcn_layers = nn.ModuleList()
        cur = in_feats
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNLayer(cur, gcn_hidden))
            cur = gcn_hidden

        self.tcn = TCN(
            in_channels=gcn_hidden,
            tcn_hidden=tcn_hidden,
            n_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )

        self.gcn_refine = GCNLayer(tcn_hidden, tcn_hidden)
        self.dropout    = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(tcn_hidden * num_input_nodes, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden, out_feats),
        )

    def forward(self, X, A_norm):
        B, T, V, _ = X.shape

        x = X
        for gcn in self.gcn_layers:
            x = gcn(x, A_norm)                      

        x = x.permute(0, 2, 3, 1)                   
        x = x.reshape(B * V, self.gcn_hidden, T)    

        pooled = self.tcn(x)
        pooled = self.dropout(pooled)
        pooled = pooled.reshape(B, V, self.tcn_hidden)

        refined = self.gcn_refine(
            pooled.unsqueeze(1), A_norm
        ).squeeze(1)                                 
        refined = self.dropout(refined)

        combined = refined.reshape(B, V * self.tcn_hidden)
        return self.fc(combined)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, A_tensor, device, clip_grad):
    model.train()
    total, n = 0.0, 0
    total_grad_norm = 0.0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb, A_tensor)
        loss = reconstruction_loss(pred, yb)
        loss.backward()
        
        # Calculate pre-clipping gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm * xb.size(0)
        
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total += loss.item() * xb.size(0)
        n     += xb.size(0)
        
    return total / n, total_grad_norm / n


def eval_epoch(model, loader, A_tensor, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, A_tensor)
            loss = reconstruction_loss(pred, yb)
            total += loss.item() * xb.size(0)
            n     += xb.size(0)
    return total / n


def eval_metrics(model, loader, A_tensor, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb, A_tensor).cpu()
            
            # CRITICAL: Normalize quaternions to unit length before angular math
            pred = F.normalize(pred, p=2, dim=1)
            yb = F.normalize(yb.cpu(), p=2, dim=1)
            
            preds.append(pred)
            targets.append(yb)
            
    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)
    
    # 1. Existing MAE
    mae_per_feat = (preds - targets).abs().mean(dim=0)
    mae_overall  = mae_per_feat.mean().item()
    
    # 2. Angular Error
    # Compute dot product along the feature dimension
    dot_product = (preds * targets).sum(dim=1)
    
    # Absolute value handles the double-cover property
    abs_dot = torch.abs(dot_product)
    
    # Clamp to prevent NaN from torch.acos due to floating point rounding
    abs_dot = torch.clamp(abs_dot, min=-1.0, max=1.0)
    
    # Calculate angular distance in radians and convert to degrees
    angular_errors_rad = 2 * torch.acos(abs_dot)
    mean_ang_err_rad = angular_errors_rad.mean().item()
    mean_ang_err_deg = mean_ang_err_rad * (180.0 / math.pi)
    
    return mae_overall, mae_per_feat.tolist(), mean_ang_err_rad, mean_ang_err_deg

# ---------------------------------------------------------------------------
# Plotting Helper
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ST-GNN Training Diagnostics', fontsize=18)
    
    # 1. Loss Curve
    axs[0, 0].plot(epochs, history['train_loss'], label='Train MSE')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val MSE')
    axs[0, 0].set_title('Loss (MSE)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('MSE')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Angular Error
    axs[0, 1].plot(epochs, history['val_ang_err'], color='darkorange', label='Val Angular Error')
    axs[0, 1].set_title('Validation Angular Error (°)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Degrees')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Learning Rate
    axs[0, 2].plot(epochs, history['lr'], color='forestgreen', label='Learning Rate')
    axs[0, 2].set_yscale('log')
    axs[0, 2].set_title('Learning Rate Decay')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('LR (log scale)')
    axs[0, 2].legend()
    axs[0, 2].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Gradient Norm
    axs[1, 0].plot(epochs, history['grad_norm'], color='firebrick', label='Avg Grad Norm')
    axs[1, 0].set_title('Pre-Clipping Gradient Norm')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('L2 Norm')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 5. Component-wise MAE
    axs[1, 1].plot(epochs, history['qw_mae'], label='qw')
    axs[1, 1].plot(epochs, history['qx_mae'], label='qx')
    axs[1, 1].plot(epochs, history['qy_mae'], label='qy')
    axs[1, 1].plot(epochs, history['qz_mae'], label='qz')
    axs[1, 1].set_title('Component-wise Validation MAE')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('MAE')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Hide the empty 6th subplot
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------

def objective(trial, train_loader, val_loader, A_tensor, V_input, F_in, F_out, device, args):
    gcn_hidden = trial.suggest_categorical("gcn_hidden", [16, 32, 48, 64, 96])
    gcn_layers = trial.suggest_int("gcn_layers", 1, 3)

    tcn_hidden      = trial.suggest_categorical("tcn_hidden", [32, 64, 96, 128, 256])
    tcn_layers      = trial.suggest_int("tcn_layers", 1, 5)
    tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size", [3, 5, 7, 9])

    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    model = STGNN(
        num_input_nodes=V_input,
        in_feats=F_in,
        gcn_hidden=gcn_hidden,
        tcn_hidden=tcn_hidden,
        tcn_layers=tcn_layers,
        tcn_kernel_size=tcn_kernel_size,
        out_feats=F_out,
        n_gcn_layers=gcn_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val    = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, A_tensor, device, args.clip_grad)
        val_loss = eval_epoch(model, val_loader, A_tensor, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val     = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_ctr >= args.patience:
            break

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ST-GNN with TCN temporal module + Optuna HPO")

    p.add_argument("--data",        type=str,   default="dataset.npz")
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--clip-grad",   type=float, default=1.0)
    
    # --- Split Parameters ---
    p.add_argument("--val-split",   type=float, default=0.1,
                   help="Proportion of each block to use for validation")
    p.add_argument("--num-blocks",  type=int,   default=5,
                   help="Number of blocks for chronological multi-block splitting")
                   
    p.add_argument("--save-dir",    type=str,   default="./checkpoints")
    p.add_argument("--seed",        type=int,   default=20)
    p.add_argument("--patience",    type=int,   default=50)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--lr-patience", type=int,   default=20)
    p.add_argument("--lr-factor",   type=float, default=0.5)

    p.add_argument("--n-trials",    type=int,   default=30)
    p.add_argument("--skip-search", action="store_true")

    p.add_argument("--gcn-hidden",  type=int,   default=48)
    p.add_argument("--gcn-layers",  type=int,   default=2)
    p.add_argument("--tcn-hidden",       type=int, default=96)
    p.add_argument("--tcn-layers",       type=int, default=3)
    p.add_argument("--tcn-kernel-size",  type=int, default=3)
    p.add_argument("--dropout",     type=float, default=0.25)

    args, _ = p.parse_known_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    data = np.load(args.data)
    X    = data["X"]   
    y    = data["y"]   
    A    = data["A"]   

    # === MANUAL ADJACENCY OVERRIDE ===  to run experiments on Adjacency matrix
    OVERRIDE_A = True 
    
    if OVERRIDE_A:
        r_val = 0.10243823  # replace with experimentally found r_value for your dataset
        A = np.array([
            [1.0, r_val],
            [r_val, 1.0]
        ], dtype=np.float32)
        print("\n" + "-"*60)
        print(f"[INFO] Overriding NPZ Adjacency Matrix with calculated A:")
        print(A)
        print("-" * 60 + "\n")
    # =================================

    # --- Multi-Block Chronological Split ---
    N = X.shape[0]
    gap = 20  # Crucial: Must be >= your sliding window size
    
    train_idx_list = []
    val_idx_list = []
    block_size = N // args.num_blocks
    
    for i in range(args.num_blocks):
        # Determine boundaries for this chunk of data
        b_start = i * block_size
        b_end = N if i == args.num_blocks - 1 else (i + 1) * block_size
        b_len = b_end - b_start
        
        # Determine how many frames are val vs train in this block
        v_size = int(b_len * args.val_split)
        t_size = b_len - v_size
        
        # Calculate indices, applying the gap to prevent sliding window leakage
        t_end_gapped = b_start + t_size - gap
        v_end_gapped = b_end - gap if i < args.num_blocks - 1 else b_end
        
        # Only add to the lists if the chunk is large enough to survive the gap subtraction
        if t_end_gapped > b_start:
            train_idx_list.append(np.arange(b_start, t_end_gapped))
        if v_end_gapped > (b_start + t_size):
            val_idx_list.append(np.arange(b_start + t_size, v_end_gapped))
            
    train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx_list) if val_idx_list else np.array([], dtype=int)

    print(f"[INFO] Multi-Block Split ({args.num_blocks} blocks):")
    print(f"       Train samples: {len(train_idx)}")
    print(f"       Val samples:   {len(val_idx)}\n")

    train_ds = STGNNDataset(X[train_idx], y[train_idx])
    val_ds   = STGNNDataset(X[val_idx],   y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              drop_last=False, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              drop_last=False, num_workers=args.num_workers)

    A_tensor = normalize_adjacency(A).to(device)
    V_input  = X.shape[2]   # 2
    F_in     = X.shape[3]   # 4
    F_out    = y.shape[1]   # 4

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # OPTUNA HYPERPARAMETER SEARCH
    # =========================================================================
    if not args.skip_search:
        print("\n" + "="*60)
        print(f"Starting Hyperparameter Search ({args.n_trials} trials)")
        print("="*60)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        func = lambda trial: objective(
            trial, train_loader, val_loader, A_tensor,
            V_input, F_in, F_out, device, args
        )

        study.optimize(func, n_trials=args.n_trials)

        print("\nBest Trial:")
        best_trial = study.best_trial
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        best_params = best_trial.params
    else:
        best_params = {
            "gcn_hidden":      args.gcn_hidden,
            "gcn_layers":      args.gcn_layers,
            "tcn_hidden":      args.tcn_hidden,
            "tcn_layers":      args.tcn_layers,
            "tcn_kernel_size": args.tcn_kernel_size,
            "dropout":         args.dropout,
        }

    # =========================================================================
    # FINAL TRAINING WITH BEST PARAMETERS
    # =========================================================================
    print("\n" + "="*60)
    print("Training Final Model with Best Parameters:")
    print("="*60)

    final_patience = 100

    final_model = STGNN(
        num_input_nodes=V_input,
        in_feats=F_in,
        gcn_hidden=best_params["gcn_hidden"],
        tcn_hidden=best_params["tcn_hidden"],
        tcn_layers=best_params["tcn_layers"],
        tcn_kernel_size=best_params["tcn_kernel_size"],
        out_feats=F_out,
        n_gcn_layers=best_params["gcn_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val, patience_ctr = float("inf"), 0
    feat_names = ["qw", "qx", "qy", "qz"]
    
    # Initialize metric history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ang_err': [],
        'lr': [],
        'grad_norm': [],
        'qw_mae': [],
        'qx_mae': [],
        'qy_mae': [],
        'qz_mae': []
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train and get loss + gradient norm
        train_loss, grad_norm = train_epoch(final_model, train_loader, optimizer, A_tensor, device, args.clip_grad)
        val_loss = eval_epoch(final_model, val_loader, A_tensor, device)
        
        # Evaluate feature metrics and angular error every epoch for continuous plotting
        mae, per_feat, ang_err_rad, ang_err_deg = eval_metrics(final_model, val_loader, A_tensor, device)
        
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        # Append to history for plots
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ang_err'].append(ang_err_deg)
        history['lr'].append(lr_now)
        history['grad_norm'].append(grad_norm)
        history['qw_mae'].append(per_feat[0])
        history['qx_mae'].append(per_feat[1])
        history['qy_mae'].append(per_feat[2])
        history['qz_mae'].append(per_feat[3])

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {lr_now:.2e} | {dt:.1f}s")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save(final_model.state_dict(), save_dir / "stgnn_tcn_best.pt")

            feat_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(feat_names, per_feat))
            print(f"  [✓] Best model saved! Overall MAE={mae:.4f} | Angular Error: {ang_err_deg:.2f}° ({ang_err_rad:.4f} rad)")
            print(f"      {feat_str}")
        else:
            patience_ctr += 1

        if patience_ctr >= final_patience:
            print("\n[EARLY STOP]")
            break

    print(f"\nDone. Best val MSE: {best_val:.6f}")
    print(f"Best model saved to: {save_dir / 'stgnn_tcn_best.pt'}")
    
    # Generate and save the training diagnostic plots
    plot_training_curves(history, save_dir)
    print(f"[INFO] Diagnostics plot saved to {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
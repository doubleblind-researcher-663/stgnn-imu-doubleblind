"""
train_stgnnGRU.py

ST-GNN for missing IMU node prediction WITH Hyperparameter Optimization (Optuna).
X=(N,20,2,4), y=(N,4), A=(2,2)

# Usage example: python train_stgnnGRU.py --data dataset.npz --epochs 400 --batch 32 --lr 1e-3 --n-trials 30 --save-dir ./gru_checkpoints
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


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        weights = F.softmax(self.attn(seq).squeeze(-1), dim=-1)
        return (weights.unsqueeze(-1) * seq).sum(dim=1)


class STGNN(nn.Module):
    def __init__(self, num_input_nodes, in_feats, gcn_hidden,
                 gru_hidden, out_feats, n_gcn_layers=2, dropout=0.1):
        super().__init__()
        self.num_input_nodes = num_input_nodes
        self.gcn_hidden      = gcn_hidden
        self.gru_hidden      = gru_hidden
        self.out_feats       = out_feats

        self.gcn_layers = nn.ModuleList()
        cur = in_feats  # 4 (qw, qx, qy, qz)
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNLayer(cur, gcn_hidden))
            cur = gcn_hidden

        self.gru = nn.GRU(gcn_hidden, gru_hidden, batch_first=True)
        self.temporal_pool = TemporalAttentionPool(gru_hidden)
        self.gcn_refine = GCNLayer(gru_hidden, gru_hidden)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * num_input_nodes, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, out_feats)  # out_feats = 4
        )

    def forward(self, X, A_norm):
        B, T, V, _ = X.shape
        x = X
        for gcn in self.gcn_layers:
            x = gcn(x, A_norm)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * V, T, self.gcn_hidden)
        gru_out, _ = self.gru(x)

        pooled = self.temporal_pool(gru_out)
        pooled = self.dropout(pooled)
        pooled = pooled.reshape(B, V, self.gru_hidden)

        refined = self.gcn_refine(
            pooled.unsqueeze(1), A_norm
        ).squeeze(1)
        refined = self.dropout(refined)

        combined = refined.reshape(B, V * self.gru_hidden)
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
            
            # Normalize quaternions to unit length before angular math
            pred = F.normalize(pred, p=2, dim=1)
            yb = F.normalize(yb.cpu(), p=2, dim=1)
            
            preds.append(pred)
            targets.append(yb)
            
    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)
    
    # Existing MAE
    mae_per_feat = (preds - targets).abs().mean(dim=0)
    mae_overall  = mae_per_feat.mean().item()
    
    # Angular Error Calculation
    dot_product = (preds * targets).sum(dim=1)
    abs_dot = torch.abs(dot_product)
    abs_dot = torch.clamp(abs_dot, min=-1.0, max=1.0)
    
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
# Optuna Objective Function
# ---------------------------------------------------------------------------
def objective(trial, train_loader, val_loader, A_tensor, V_input, F_in, F_out, device, args):
    gcn_hidden = trial.suggest_categorical("gcn_hidden", [16, 32, 48, 64, 96])
    gru_hidden = trial.suggest_categorical("gru_hidden", [32, 64, 96, 128, 256])
    gcn_layers = trial.suggest_int("gcn_layers", 1, 3)
    dropout    = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    model = STGNN(
        num_input_nodes=V_input,
        in_feats=F_in,        # 4
        gcn_hidden=gcn_hidden,
        gru_hidden=gru_hidden,
        out_feats=F_out,      # 4
        n_gcn_layers=gcn_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        # We unpack train_loss and ignore grad_norm during optuna search
        train_loss, _ = train_epoch(model, train_loader, optimizer, A_tensor, device, args.clip_grad)
        val_loss = eval_epoch(model, val_loader, A_tensor, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
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
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=str,   default="imu_stgnn_quaternion_only.npz")
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--clip-grad",   type=float, default=1.0)
    
    # --- Split Parameters ---
    p.add_argument("--val-split",   type=float, default=0.1)
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
    args, _ = p.parse_known_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    data = np.load(args.data)
    X    = data["X"]   # (N, 20, 2, 4)
    y    = data["y"]   # (N, 4)
    A    = data["A"]   # (2, 2)
    
    # === MANUAL ADJACENCY OVERRIDE ===
    OVERRIDE_A = True 
    if OVERRIDE_A:
        r_val = 0.10243823 
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
        b_start = i * block_size
        b_end = N if i == args.num_blocks - 1 else (i + 1) * block_size
        b_len = b_end - b_start
        
        v_size = int(b_len * args.val_split)
        t_size = b_len - v_size
        
        t_end_gapped = b_start + t_size - gap
        v_end_gapped = b_end - gap if i < args.num_blocks - 1 else b_end
        
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
        print("\n" + "="*50)
        print(f"Starting Hyperparameter Search ({args.n_trials} trials)")
        print("="*50)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )

        func = lambda trial: objective(
            trial, train_loader, val_loader, A_tensor,
            V_input, F_in, F_out, device, args
        )

        study.optimize(func, n_trials=args.n_trials)

        print("\n" + "="*50)
        print("Study Statistics:")
        print(f"  Finished trials:  {len(study.trials)}")
        print(f"  Pruned trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Complete trials:  {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

        print("\nBest Trial:")
        trial = study.best_trial
        print(f"  Value (Validation MSE): {trial.value:.6f}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        best_params = trial.params
    else:
        best_params = {"gcn_hidden": 48, "gru_hidden": 96, "gcn_layers": 1, "dropout": 0.25}

    # =========================================================================
    # FINAL TRAINING WITH BEST PARAMETERS
    # =========================================================================
    print("\n" + "="*50)
    print("Training Final Model with Best Parameters:")
    print(best_params)
    print("="*50)

    final_patience = 100

    final_model = STGNN(
        num_input_nodes=V_input,
        in_feats=F_in,                          # 4
        gcn_hidden=best_params["gcn_hidden"],
        gru_hidden=best_params["gru_hidden"],
        out_feats=F_out,                        # 4
        n_gcn_layers=best_params["gcn_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val, patience_ctr = float("inf"), 0
    feat_names = ["qw", "qx", "qy", "qz"]  # 4 quaternion components
    
    # Initialize metric history dictionary for plotting
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
        val_loss   = eval_epoch(final_model, val_loader, A_tensor, device)
        
        # Evaluate feature metrics and angular error
        mae, per_feat, ang_err_rad, ang_err_deg = eval_metrics(final_model, val_loader, A_tensor, device)
        
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        
        # Append to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ang_err'].append(ang_err_deg)
        history['lr'].append(lr_now)
        history['grad_norm'].append(grad_norm)
        history['qw_mae'].append(per_feat[0])
        history['qx_mae'].append(per_feat[1])
        history['qy_mae'].append(per_feat[2])
        history['qz_mae'].append(per_feat[3])

        print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e} | {dt:.1f}s")

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save(final_model.state_dict(), save_dir / "stgnn_best.pt")

            feat_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(feat_names, per_feat))
            print(f"  [✓] Best model saved! Overall MAE={mae:.4f} | Angular Error: {ang_err_deg:.2f}° ({ang_err_rad:.4f} rad)")
            print(f"      {feat_str}")
        else:
            patience_ctr += 1

        if patience_ctr >= final_patience:
            print("\n[EARLY STOP]")
            break

    print(f"\nDone. Best val MSE: {best_val:.6f}")
    print(f"Best model saved to: {save_dir / 'stgnn_best.pt'}")
    
    # Generate and save diagnostic plots
    plot_training_curves(history, save_dir)
    print(f"[INFO] Diagnostics plot saved to {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
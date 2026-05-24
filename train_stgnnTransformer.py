"""
train_stgnnTransformer.py

ST-GNN for missing IMU node prediction WITH Hyperparameter Optimization (Optuna).
X=(N,20,2,4), y=(N,4), A=(2,2)

Temporal module: Transformer encoder with learned positional embeddings.

Positional encoding: nn.Embedding(T, d_model) — a learned lookup table, one
vector per timestep position, added to the projected input before the
Transformer encoder. For a fixed window of T=20 this is preferable to
sinusoidal encoding because the model can learn whatever positional structure
is actually useful for the IMU signal rather than using a generic frequency
decomposition.

# Usage example: python train_stgnnTransformer.py --data dataset.npz --epochs 300 --batch 32 --lr 1e-3 --n-trials 20 --save-dir ./transformer_checkpoints
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


class TemporalTransformer(nn.Module):
    """
    Transformer encoder with learned positional embeddings for temporal modelling.

    Pipeline (per node sequence)
    ----------------------------
    Input  : (B*V, T, gcn_hidden)
    1. Linear projection  → (B*V, T, d_model)
    2. Learned positional embedding  → added in-place
    3. Dropout on embedded input
    4. Transformer encoder (n_layers x TransformerEncoderLayer)
    5. CLS-token pooling  → (B*V, d_model)
    """
    def __init__(self, seq_len, in_channels, d_model, nhead,
                 n_layers, ffn_dim, dropout):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.d_model  = d_model
        self.seq_len  = seq_len

        # 1. Input projection: gcn_hidden → d_model
        self.input_proj = nn.Linear(in_channels, d_model)

        # 2. Learned [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3. Learned positional embedding
        self.pos_embed = nn.Embedding(seq_len + 1, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (B*V, T, gcn_hidden)
        BV, T, _ = x.shape

        # 1. Project to d_model
        x = self.input_proj(x)                                  # (BV, T, d_model)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(BV, -1, -1)                 # (BV, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                        # (BV, T+1, d_model)

        # 3. Add learned positional embeddings
        positions = torch.arange(T + 1, device=x.device)        # [0, 1, ..., T]
        x = x + self.pos_embed(positions)                       # broadcast over BV
        x = self.embed_dropout(x)

        # 4. Transformer encoder
        x = self.transformer(x)                                 # (BV, T+1, d_model)

        # 5. Extract CLS output as sequence summary
        return x[:, 0, :]                                       # (BV, d_model)


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN where the temporal module is a Transformer encoder
    with learned positional embeddings.
    """
    def __init__(
        self, num_input_nodes, seq_len, in_feats, gcn_hidden, d_model,
        nhead, tf_layers, ffn_dim, out_feats, n_gcn_layers=2, dropout=0.1,
    ):
        super().__init__()
        self.num_input_nodes = num_input_nodes
        self.gcn_hidden      = gcn_hidden
        self.d_model         = d_model

        # --- Spatial: GCN stack ---
        self.gcn_layers = nn.ModuleList()
        cur = in_feats
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNLayer(cur, gcn_hidden))
            cur = gcn_hidden

        # --- Temporal: Transformer ---
        self.temporal_transformer = TemporalTransformer(
            seq_len=seq_len,
            in_channels=gcn_hidden,
            d_model=d_model,
            nhead=nhead,
            n_layers=tf_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # --- Spatial refinement ---
        self.gcn_refine = GCNLayer(d_model, d_model)
        self.dropout    = nn.Dropout(dropout)

        # --- Prediction head ---
        self.fc = nn.Sequential(
            nn.Linear(d_model * num_input_nodes, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_feats),
        )

    def forward(self, X, A_norm):
        B, T, V, _ = X.shape

        # 1. GCN stack
        x = X
        for gcn in self.gcn_layers:
            x = gcn(x, A_norm)

        # 2. Reshape: one sequence per node
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * V, T, self.gcn_hidden)

        # 3. Transformer -> CLS-pooled
        pooled = self.temporal_transformer(x)
        pooled = self.dropout(pooled)

        # 4. Reshape back
        pooled = pooled.reshape(B, V, self.d_model)

        # 5. GCN refinement
        refined = self.gcn_refine(
            pooled.unsqueeze(1), A_norm
        ).squeeze(1)
        refined = self.dropout(refined)

        # 6. Flatten + MLP
        combined = refined.reshape(B, V * self.d_model)
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
            
            # Normalize quaternions to unit length
            pred = F.normalize(pred, p=2, dim=1)
            yb = F.normalize(yb.cpu(), p=2, dim=1)
            
            preds.append(pred)
            targets.append(yb)
            
    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)
    
    # MAE per feature
    mae_per_feat = (preds - targets).abs().mean(dim=0)
    mae_overall  = mae_per_feat.mean().item()
    
    # Angular Error
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
# Optuna objective — searches GCN + Transformer hyperparameters jointly
# ---------------------------------------------------------------------------

DMODEL_NHEAD_PAIRS = [
    (32,  2), (32,  4),
    (64,  2), (64,  4), (64,  8),
    (96,  4), (96,  8),
    (128, 4), (128, 8),
    (256, 4), (256, 8),
]

def objective(trial, train_loader, val_loader, A_tensor,
              V_input, F_in, F_out, seq_len, device, args):

    gcn_hidden = trial.suggest_categorical("gcn_hidden", [16, 32, 48, 64, 96])
    gcn_layers = trial.suggest_int("gcn_layers", 1, 3)

    pair_idx = trial.suggest_categorical(
        "dmodel_nhead_idx", list(range(len(DMODEL_NHEAD_PAIRS)))
    )
    d_model, nhead = DMODEL_NHEAD_PAIRS[pair_idx]

    tf_layers = trial.suggest_int("tf_layers", 1, 4)
    ffn_dim   = trial.suggest_categorical("ffn_dim", [64, 128, 256, 512])

    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)

    model = STGNN(
        num_input_nodes=V_input,
        seq_len=seq_len,
        in_feats=F_in,
        gcn_hidden=gcn_hidden,
        d_model=d_model,
        nhead=nhead,
        tf_layers=tf_layers,
        ffn_dim=ffn_dim,
        out_feats=F_out,
        n_gcn_layers=gcn_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val     = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_epoch(model, train_loader, optimizer, A_tensor, device, args.clip_grad)
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
    p = argparse.ArgumentParser(
        description="ST-GNN with Transformer temporal module + learned positional embedding + Optuna HPO"
    )

    # --- Data / infrastructure ---
    p.add_argument("--data",        type=str,   default="dataset.npz")
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
    p.add_argument("--patience",    type=int,   default=50,
                   help="Early-stop patience (epochs without val improvement)")
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--lr-patience", type=int,   default=20)
    p.add_argument("--lr-factor",   type=float, default=0.5)

    # --- Optuna ---
    p.add_argument("--n-trials",    type=int,   default=30)
    p.add_argument("--skip-search", action="store_true",
                   help="Skip HPO and use CLI fallback params")

    # --- GCN fallback ---
    p.add_argument("--gcn-hidden",  type=int,   default=48)
    p.add_argument("--gcn-layers",  type=int,   default=2)

    # --- Transformer fallback ---
    p.add_argument("--d-model",     type=int,   default=64,
                   help="Transformer internal dimension; must be divisible by --nhead (fallback)")
    p.add_argument("--nhead",       type=int,   default=4,
                   help="Number of self-attention heads; must divide --d-model (fallback)")
    p.add_argument("--tf-layers",   type=int,   default=2,
                   help="Number of Transformer encoder layers (fallback)")
    p.add_argument("--ffn-dim",     type=int,   default=128,
                   help="Feedforward hidden dim inside each encoder layer (fallback)")

    # --- Shared regularisation fallback ---
    p.add_argument("--dropout",     type=float, default=0.1)

    args, _ = p.parse_known_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    data    = np.load(args.data)
    X       = data["X"]   # (N, 20, 2, 4)
    y       = data["y"]   # (N, 4)
    A       = data["A"]   # (2, 2)
    
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

    N       = X.shape[0]
    seq_len = X.shape[1]  # 20 — fixed window, drives positional embedding size
    gap     = seq_len     # Crucial gap for sliding windows
    
    # --- Multi-Block Chronological Split ---
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
        print("\n" + "="*60)
        print(f"Starting Hyperparameter Search ({args.n_trials} trials)")
        print("Searching: gcn_hidden, gcn_layers,")
        print("           (d_model, nhead) joint pair, tf_layers, ffn_dim, dropout")
        print("="*60)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        func = lambda trial: objective(
            trial, train_loader, val_loader, A_tensor,
            V_input, F_in, F_out, seq_len, device, args
        )

        study.optimize(func, n_trials=args.n_trials)

        print("\n" + "="*60)
        print("Study Statistics:")
        print(f"  Finished trials : {len(study.trials)}")
        pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"  Pruned trials   : {len(pruned)}")
        print(f"  Complete trials : {len(complete)}")

        print("\nBest Trial:")
        best_trial = study.best_trial
        print(f"  Value (Validation MSE): {best_trial.value:.6f}")
        print("  Raw params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        raw = best_trial.params
        pair_idx        = raw["dmodel_nhead_idx"]
        d_model, nhead  = DMODEL_NHEAD_PAIRS[pair_idx]
        best_params = {
            "gcn_hidden": raw["gcn_hidden"],
            "gcn_layers": raw["gcn_layers"],
            "d_model":    d_model,
            "nhead":      nhead,
            "tf_layers":  raw["tf_layers"],
            "ffn_dim":    raw["ffn_dim"],
            "dropout":    raw["dropout"],
        }
    else:
        assert args.d_model % args.nhead == 0, (
            f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead})"
        )
        best_params = {
            "gcn_hidden": args.gcn_hidden,
            "gcn_layers": args.gcn_layers,
            "d_model":    args.d_model,
            "nhead":      args.nhead,
            "tf_layers":  args.tf_layers,
            "ffn_dim":    args.ffn_dim,
            "dropout":    args.dropout,
        }

    # =========================================================================
    # FINAL TRAINING WITH BEST PARAMETERS
    # =========================================================================
    print("\n" + "="*60)
    print("Training Final Model with Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*60)

    final_patience = 100

    final_model = STGNN(
        num_input_nodes=V_input,
        seq_len=seq_len,
        in_feats=F_in,
        gcn_hidden=best_params["gcn_hidden"],
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        tf_layers=best_params["tf_layers"],
        ffn_dim=best_params["ffn_dim"],
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
    
    # Metrics history dictionary
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
        train_loss, grad_norm = train_epoch(final_model, train_loader, optimizer, A_tensor, device, args.clip_grad)
        val_loss   = eval_epoch(final_model, val_loader, A_tensor, device)
        
        mae, per_feat, ang_err_rad, ang_err_deg = eval_metrics(final_model, val_loader, A_tensor, device)
        
        lr_now = optimizer.param_groups[0]["lr"]
        dt         = time.time() - t0
        
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
            torch.save(final_model.state_dict(), save_dir / "stgnn_transformer_best.pt")

            feat_str = "  ".join(f"{nm}={v:.4f}" for nm, v in zip(feat_names, per_feat))
            print(f"  [✓] Best model saved! Overall MAE={mae:.4f}  |  Angular Error: {ang_err_deg:.2f}° ({ang_err_rad:.4f} rad)")
            print(f"      {feat_str}")
        else:
            patience_ctr += 1

        if patience_ctr >= final_patience:
            print("\n[EARLY STOP]")
            break

    print(f"\nDone. Best val MSE: {best_val:.6f}")
    print(f"Best model saved to: {save_dir / 'stgnn_transformer_best.pt'}")
    
    # Save training curves
    plot_training_curves(history, save_dir)
    print(f"[INFO] Diagnostics plot saved to {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
import argparse
import time
import csv
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pythonosc import dispatcher, osc_server, udp_client

# =============================================================================
# MODEL ARCHITECTURE (GRU Variant)
# =============================================================================

def normalize_adjacency(A):
    A = A.astype(np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A_hat.sum(axis=1)
    d = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    D = np.diag(d)
    return torch.from_numpy(D @ A_hat @ D).float()

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
        cur = in_feats  
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
            nn.Linear(gru_hidden, out_feats) 
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

# =============================================================================
# INFERENCE LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="stgnn_best.pt", help="Path to model checkpoint")
    parser.add_argument("--enable-real-forearm", action="store_true", help="Listen for, log, and send real forearm data (node 0)")
    args = parser.parse_args()

    # Configuration
    UE5_IP = "10.116.219.91"
    UE5_PORT_PRED = 4210
    UE5_PORT_REAL = 4211
    OSC_LISTEN_PORT = 9000

    WINDOW = 20
    V_INPUT = 2
    F_IN = 4
    F_OUT = 4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    # 1. Setup Adjacency Matrix, replace 0.10243823 with the r_value of your dataset, for your specific application of IMU reconstruction.
    A_raw = np.array([
        [1.0, 0.10243823],
        [0.10243823, 1.0]
    ], dtype=np.float32)
    A_tensor = normalize_adjacency(A_raw).to(DEVICE)

    # 2. Load Model (Using fallback GRU hyperparameters)
    model = STGNN(
        num_input_nodes=V_INPUT,
        in_feats=F_IN,
        gcn_hidden=48,
        gru_hidden=96,
        out_feats=F_OUT,
        n_gcn_layers=1,
        dropout=0.25
    ).to(DEVICE)

    try:
        ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
        if isinstance(ck, dict) and "model_state" in ck:
            model.load_state_dict(ck["model_state"])
        else:
            model.load_state_dict(ck)
        print(f"Loaded checkpoint from {args.ckpt}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {args.ckpt}. Error: {e}")
    model.eval()

    # 3. Buffers
    buffers = {
        1: deque(maxlen=WINDOW), # Upper arm
        2: deque(maxlen=WINDOW), # Palm
    }
    if args.enable_real_forearm:
        buffers[0] = deque(maxlen=WINDOW) # Real Forearm

    # 4. OSC Clients
    ue5_client_pred = udp_client.SimpleUDPClient(UE5_IP, UE5_PORT_PRED)
    ue5_client_real = udp_client.SimpleUDPClient(UE5_IP, UE5_PORT_REAL)

    # 5. CSV Logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"imu_gru_inference_log_{timestamp}.csv"
    
    CSV_HEADER = ["timestamp"]
    if args.enable_real_forearm:
        CSV_HEADER += [f"real0_{k}" for k in ["qx","qy","qz","qw"]]
    CSV_HEADER += [f"real1_{k}" for k in ["qx","qy","qz","qw"]]
    CSV_HEADER += [f"real2_{k}" for k in ["qx","qy","qz","qw"]]
    CSV_HEADER += [f"pred0_{k}" for k in ["qx","qy","qz","qw"]]

    csv_file = open(log_filename, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    def log_to_csv(real0, real1, real2, pred0):
        row = [time.time()]
        if args.enable_real_forearm:
            row += real0
        row += real1 + real2 + pred0
        csv_writer.writerow(row)
        csv_file.flush()

    # 6. Prediction Logic
    def predict_and_send():
        X = np.zeros((1, WINDOW, V_INPUT, F_IN), dtype=np.float32)

        arr1 = np.stack(buffers[1], axis=0)
        arr2 = np.stack(buffers[2], axis=0)

        # Index 0 = upper arm, Index 1 = palm
        X[0, :, 0, :] = arr1
        X[0, :, 1, :] = arr2

        X_tensor = torch.from_numpy(X).to(DEVICE)

        with torch.no_grad():
            Y = model(X_tensor, A_tensor)
            pred_quat = Y.cpu().numpy()[0] # [qw, qx, qy, qz]

        # Reorder [qw, qx, qy, qz] -> [qx, qy, qz, qw] for UE5 and CSV
        def reorder_for_ue5(q):
            return [q[1], q[2], q[3], q[0]]

        pred0_out = reorder_for_ue5(pred_quat)
        real1_out = reorder_for_ue5(arr1[-1].tolist())
        real2_out = reorder_for_ue5(arr2[-1].tolist())
        
        real0_out = None

        try:
            # Send Predictions (Padding IMU slots with 0)
            ue5_client_pred.send_message("/imu/forward", pred0_out + [0] + [0]*6) 
            ue5_client_pred.send_message("/imu/forward", real1_out + [1] + [0]*6)
            ue5_client_pred.send_message("/imu/forward", real2_out + [2] + [0]*6)

            # Process Real Forearm (if toggled)
            if args.enable_real_forearm and len(buffers[0]) > 0:
                real0_out = reorder_for_ue5(buffers[0][-1].tolist())
                ue5_client_real.send_message("/imu/forward", real0_out + [0] + [0]*6)
                ue5_client_real.send_message("/imu/forward", real1_out + [1] + [0]*6)
                ue5_client_real.send_message("/imu/forward", real2_out + [2] + [0]*6)
                
                print(f"Pred: {pred0_out} | Real: {real0_out}")
            else:
                print("Sent prediction to UE5.")

        except Exception as e:
            print("UE5 send error:", e)

        log_to_csv(real0_out, real1_out, real2_out, pred0_out)

    # 7. OSC Handler
    def imu_handler(address, *args_osc):
        if len(args_osc) < 5:
            return

        qx, qy, qz, qw, node_id = args_osc[0], args_osc[1], args_osc[2], args_osc[3], int(args_osc[4])
        
        # Model expects [qw, qx, qy, qz]
        feat = np.array([qw, qx, qy, qz], dtype=np.float32)

        if node_id in buffers:
            buffers[node_id].append(feat)

        if len(buffers[1]) == WINDOW and len(buffers[2]) == WINDOW:
            predict_and_send()

    # 8. Start Server
    disp = dispatcher.Dispatcher()
    disp.map("/imu/1", imu_handler)
    disp.map("/imu/*", imu_handler)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", OSC_LISTEN_PORT), disp)
    print(f"Listening on 0.0.0.0:{OSC_LISTEN_PORT}")
    if args.enable_real_forearm:
        print("Real forearm tracking is ENABLED.")
    else:
        print("Real forearm tracking is DISABLED.")
        
    server.serve_forever()

if __name__ == "__main__":
    main()
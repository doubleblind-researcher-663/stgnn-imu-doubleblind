import argparse
import numpy as np
import pandas as pd

#usage: python build_dataset.py --input collected_data.csv --output dataset.npz

def check_quaternion_norms(quaternions, name):
    """
    Calculates the norms of the quaternions along the last axis.
    Prints a warning if the mean norm deviates from 1.0 by more than 0.05.
    """
    norms = np.linalg.norm(quaternions, axis=-1)
    mean_norm = np.mean(norms)
    if abs(1.0 - mean_norm) > 0.05:
        print(f"WARNING: Mean quaternion norm for {name} is {mean_norm:.4f}, "
              f"deviating by more than 0.05 from 1.0.")

def main():
    parser = argparse.ArgumentParser(description="Build NPZ dataset from raw quaternion CSV.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the output NPZ file")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # 1. Extract columns in STRICT [qx, qy, qz, qw] order to match the model's training
    # Node 0: real0 (forearm - target)
    y_full_seq = df[['real0_qx', 'real0_qy', 'real0_qz', 'real0_qw']].to_numpy()

    # Node 1: real1 (upper arm - input node 0)
    real1_seq = df[['real1_qx', 'real1_qy', 'real1_qz', 'real1_qw']].to_numpy()

    # Node 2: real2 (palm - input node 1)
    real2_seq = df[['real2_qx', 'real2_qy', 'real2_qz', 'real2_qw']].to_numpy()
    
    # Extract timestamps if they exist
    timestamps = df['timestamp'].to_numpy() if 'timestamp' in df.columns else np.arange(len(df))

    # 2. Perform norm checks
    check_quaternion_norms(y_full_seq, "real0 (forearm)")
    check_quaternion_norms(real1_seq, "real1 (upper arm)")
    check_quaternion_norms(real2_seq, "real2 (palm)")

    # 3. Combine input nodes into shape (T, 2, 4)
    # Index 0 = upper arm, Index 1 = palm. Forearm (real0) is excluded.
    X_full_seq = np.stack((real1_seq, real2_seq), axis=1)

    # 4. Generate sliding windows (window=20, step=1, no shuffling)
    window_size = 20
    T = len(df)
    N = T - window_size + 1

    if N <= 0:
        raise ValueError(f"Sequence length {T} is too short for a window size of {window_size}.")

    print(f"Generating {N} samples with sliding window size {window_size}...")
    
    # Pre-allocate arrays
    X = np.empty((N, window_size, 2, 4), dtype=np.float32)
    y = np.empty((N, 4), dtype=np.float32)
    t = np.empty((N,), dtype=np.float64) # Keep precision for timestamps

    for i in range(N):
        X[i] = X_full_seq[i : i + window_size]
        # Label is the target (forearm) at the LAST timestep of the window
        y[i] = y_full_seq[i + window_size - 1]
        # Record the timestamp for the target prediction
        t[i] = timestamps[i + window_size - 1]

    # 5. Adjacency Matrix (2x2)
    A = np.array([
        [1.0, 0],
        [0, 1.0]
    ], dtype=np.float32)

    # 6. Save to NPZ
    print(f"Saving arrays to {args.output}...")
    np.savez(args.output, X=X, y=y, A=A, t=t)

    # 7. Print final sanity check
    print("\n--- Final Data Shapes ---")
    print(f"X shape: {X.shape}  -> (N, timesteps, nodes, features)")
    print(f"y shape: {y.shape}        -> (N, features)")
    print(f"t shape: {t.shape}        -> (N,)")
    print(f"A shape: {A.shape}           -> (nodes, nodes)")
    print("Done!")

if __name__ == "__main__":
    main()
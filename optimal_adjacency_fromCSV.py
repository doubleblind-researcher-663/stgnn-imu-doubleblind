import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def unwrap_quaternions(q_seq):
    """
    Ensures temporal continuity of a quaternion time series 
    by checking the dot product between consecutive frames.
    Prevents artificial sign flips (q and -q represent the same rotation).
    """
    q_unwrapped = np.copy(q_seq)
    
    for t in range(1, len(q_unwrapped)):
        # If the dot product is negative, the shortest path 
        # is to the negated quaternion.
        if np.dot(q_unwrapped[t], q_unwrapped[t-1]) < 0:
            q_unwrapped[t] = -q_unwrapped[t]
            
    return q_unwrapped

def compute_optimal_adjacency(csv_path):
    # 1. Load the data
    try:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found.")
        return

    # 2. Extract continuous sequence directly from CSV columns
    # Upper arm is real1, Palm is real2
    upper_arm_cols = ['real1_qw', 'real1_qx', 'real1_qy', 'real1_qz']
    palm_cols      = ['real2_qw', 'real2_qx', 'real2_qy', 'real2_qz']
    
    # Convert to numpy arrays for calculation
    upper_arm_quat = df[upper_arm_cols].to_numpy(dtype=np.float32)
    palm_quat      = df[palm_cols].to_numpy(dtype=np.float32)

    # 3. Apply Continuous Unwrapping (CRITICAL for Quaternions)
    upper_arm_quat = unwrap_quaternions(upper_arm_quat)
    palm_quat      = unwrap_quaternions(palm_quat)

    # 4. Calculate correlation for each quaternion component (qw, qx, qy, qz)
    correlations = []
    for i in range(4):
        # Calculate Pearson r for this specific axis
        r, _ = pearsonr(upper_arm_quat[:, i], palm_quat[:, i])
        
        # Use absolute value because a negative correlation 
        # still represents a strong structural edge.
        correlations.append(abs(r))

    # 5. Average the correlations to get a single edge weight
    r_avg = np.mean(correlations)

    # 6. Construct the Raw Optimal A matrix
    A_optimal = np.array([
        [1.0, r_avg],
        [r_avg, 1.0]
    ])

    # 7. Construct the Row-Normalized A matrix
    # A_norm(i, j) = A(i, j) / sum(A(i, :))
    A_norm = A_optimal / A_optimal.sum(axis=1, keepdims=True)

    # --- Print Results ---
    print("\n--- Correlation Metrics ---")
    print(f"Component-wise Correlations (qw, qx, qy, qz): {[round(c, 4) for c in correlations]}")
    print(f"Average Quaternion Correlation (r): {r_avg:.4f}\n")

    print("--- Raw Optimal Adjacency Matrix (A) ---")
    print(A_optimal)
    print("\n")

    print("--- Row-Normalized Adjacency Matrix (A_norm) ---")
    print(A_norm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate optimal adjacency matrix from a CSV file.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV file")
    args = parser.parse_args()
    
    compute_optimal_adjacency(args.input)
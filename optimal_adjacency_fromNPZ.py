import numpy as np
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

def compute_optimal_adjacency():
    # 1. Load the data
    try:
        data = np.load('dataset.npz')
        X = data['X']
    except FileNotFoundError:
        print("Error: 'dataset.npz' not found.")
        return

    # 2. Extract continuous sequence without overlapping duplicates
    # X shape is (N, T, V, F). We take T=0 to stitch the sliding window 
    # back into a continuous 100Hz recording. 
    upper_arm_quat = X[:, 0, 0, :]  # Node 0
    palm_quat      = X[:, 0, 1, :]  # Node 1

    # 3. Apply Continuous Unwrapping (CRITICAL for Quaternions)
    upper_arm_quat = unwrap_quaternions(upper_arm_quat)
    palm_quat      = unwrap_quaternions(palm_quat)

    # 4. Calculate correlation for each quaternion component (qx, qy, qz, qw)
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
    print("--- Correlation Metrics ---")
    print(f"Component-wise Correlations (qx, qy, qz, qw): {[round(c, 4) for c in correlations]}")
    print(f"Average Quaternion Correlation (r): {r_avg:.4f}\n")

    print("--- Raw Optimal Adjacency Matrix (A) ---")
    print(A_optimal)
    print("\n")

    print("--- Row-Normalized Adjacency Matrix (A_norm) ---")
    print(A_norm)

if __name__ == "__main__":
    compute_optimal_adjacency()
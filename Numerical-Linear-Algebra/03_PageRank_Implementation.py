import numpy as np
import time
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

#######################################################################

# Function to construct the diagonal matrix D
def compute_D(matrix):
    # Avoid division errors and NaN values
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the reciprocal of the sum of each column
        diag_values = np.divide(1., np.asarray(matrix.sum(axis=0)).reshape(-1))
    # Replace any invalid entries with zero
    diag_values = np.nan_to_num(diag_values)
    # Construct and return a sparse diagonal matrix
    return sp.diags(diag_values)

#######################################################################

# Power Method: Matrix Storing Variant
def power_method_storing(A, alpha=0.15, tolerance=1e-05):
    n = A.shape[0]  # Number of nodes
    start_time = time.time()  # Measure runtime
    e_vector = np.ones((n, 1))  # Vector of ones
    jump_probs = []  # Vector to store random jump probabilities

    # Compute the random jump probabilities
    for i in range(n):
        column_sum = A[:, i].count_nonzero()
        jump_probs.append(alpha / n if column_sum > 0 else 1. / n)

    jump_probs = np.asarray(jump_probs)  # Convert to NumPy array
    x_prev = np.ones((n, 1))  # Initial vector
    x_curr = np.ones((n, 1)) / n  # Normalized starting point

    # Iterate until convergence
    while np.linalg.norm(x_prev - x_curr, np.inf) > tolerance:
        x_prev = x_curr
        random_jump_effect = jump_probs.dot(x_curr)
        x_curr = (1 - alpha) * A.dot(x_curr) + e_vector * random_jump_effect

    print("Runtime (storing matrices): {:.4f} seconds".format(time.time() - start_time))
    return x_curr / np.sum(x_curr)  # Normalize and return

#######################################################################

# Function to create column-to-row mappings
def build_L(row_indices, col_indices):
    mapping = {}
    for i in range(len(col_indices)):
        col = col_indices[i]
        if col in mapping:
            mapping[col] = np.append(mapping[col], row_indices[i])
        else:
            mapping[col] = np.array([row_indices[i]])
    return mapping

#######################################################################

# Power Method: Non-Storing Variant
def power_method_non_storing(matrix, alpha, tolerance):
    n = matrix.shape[0]  # Number of nodes
    start_time = time.time()  # Measure runtime
    col_to_rows = build_L(matrix.nonzero()[0], matrix.nonzero()[1])  # Map columns to non-zero rows
    x_prev = np.ones((n, 1))  # Initial vector
    x_curr = np.ones((n, 1)) / n  # Normalized starting point

    # Iterate until convergence
    while np.linalg.norm(x_prev - x_curr, np.inf) > tolerance:
        x_prev = x_curr
        x_curr = np.zeros((n, 1))  # Initialize for this iteration

        for col in range(n):
            if col in col_to_rows:
                if len(col_to_rows[col]) > 0:
                    x_curr[col_to_rows[col]] += x_prev[col] / len(col_to_rows[col])
                else:
                    x_curr += x_prev[col] / n
            else:
                x_curr += x_prev[col] / n

        x_curr = (1 - alpha) * x_curr + alpha / n

    print("Runtime (non-storing matrices): {:.4f} seconds".format(time.time() - start_time))
    return x_curr / np.sum(x_curr)  # Normalize and return

#######################################################################

# Input file reading and preprocessing
matrix_data = sio.mmread("p2p-Gnutella30.mtx")
matrix_sparse = sp.csr_matrix(matrix_data)
D_matrix = compute_D(matrix_sparse)
A_matrix = sp.csr_matrix(matrix_sparse.dot(D_matrix))

# Prompt the user for damping factor and tolerance
damping_factor = float(input("\nEnter the damping factor (between 0 and 1; default is 0.15):\n"))
if not (0 < damping_factor < 1):
    damping_factor = 0.15
    print("Invalid input. Damping factor set to default value: 0.15.")

tolerance = float(input("\nEnter the tolerance (between 1e-04 and 1e-10; default is 1e-05):\n"))
if not (1e-11 < tolerance < 1e-04):
    tolerance = 1e-05
    print("Invalid input. Tolerance set to default value: 1e-05.")

#######################################################################

# Method selection and execution
method = -1
while method not in [1, 2]:
    method = int(input("\nSelect the method:\n 1. Power Method (Storing Matrices)\n 2. Power Method (Non-Storing Matrices)\n"))
    if method == 1:
        print("\nRunning Power Method with matrix storage...")
        page_rank = power_method_storing(A_matrix, damping_factor, tolerance)
        print("Normalized PageRank (Storing Matrices):\n", np.round(page_rank, 6))
    elif method == 2:
        print("\nRunning Power Method without matrix storage...")
        page_rank = power_method_non_storing(A_matrix, damping_factor, tolerance)
        print("Normalized PageRank (Non-Storing Matrices):\n", np.round(page_rank, 6))
    else:
        print("Please choose either option 1 or 2.")

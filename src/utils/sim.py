import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *


def computeSim(X, Y, filtering=False, shrinkage=40, k_filtering=200):
    """
    Returns X_shape[0]xY_shape[1]
    """
    chunksize = 1000
    mat_len = X.shape[0]
    S=None
    x_norm = norm(X, axis=1)
    x_norm[x_norm==0] = 1
    X = X.multiply(csr_matrix(np.reciprocal(x_norm)).transpose())
    y_norm = norm(Y, axis = 0)
    y_norm[y_norm==0] = 1
    Y = Y.multiply(np.reciprocal(y_norm))
    Y_ones = Y.copy()
    Y_ones.data = np.ones_like(Y_ones.data)
    for chunk in range(0, mat_len, chunksize):
        if chunk + chunksize > mat_len:
            end = mat_len
        else:
            end = chunk + chunksize
        print(('Building cosine similarity matrix for [' +
               str(chunk) + ', ' + str(end) + ') ...'))
        # First compute similarity
        S_prime = X[chunk:end].tocsr().dot(Y)
        print("S_prime prime built.")
        # compute common features
        X_ones = X[chunk:end]
        X_ones.data = np.ones_like(X_ones.data)
        S_num = X_ones.dot(Y_ones)
        S_den = S_num.copy()
        S_den.data += shrinkage
        S_den.data = np.reciprocal(S_den.data)
        S_prime = S_prime.multiply(S_num).multiply(S_den)
        print("S_prime applied shrinkage")
        # Top-K filtering.
        # We only keep the top K similarity weights to avoid considering many
        # barely-relevant neighbors
        for row_i in range(0, S_prime.shape[0]):
            row = S_prime.data[S_prime.indptr[row_i]:S_prime.indptr[row_i + 1]]
            if row.shape[0] > k_filtering:
                sorted_idx = np.argpartition(row, row.shape[0] - k_filtering)[:-k_filtering]
                row[sorted_idx] = 0

        print("S_prime filtered")
        S_prime.eliminate_zeros()
        if S is None:
            S = S_prime
        else:
            # stack matrices vertically
            S = vstack([S, S_prime], format="csr")
    return S

import scipy.sparse as sps
import numpy as np


def top_k_filtering(matrix, topK):
    # Check if matrix is sparse
    if sps.issparse(matrix):
        matrix = matrix.tocsr()
        for row_i in range(0, matrix.shape[0]):
            row = matrix.data[matrix.indptr[row_i]:matrix.indptr[row_i + 1]]
            # delete some items
            if row.shape[0] >= topK:
                sorted_idx = np.argpartition(
                    row, row.shape[0] - topK)[:-topK]
                row[sorted_idx] = 0
        matrix.eliminate_zeros()
    else:
        for row_i in range(0, matrix.shape[0]):
            row = matrix[row_i]
            # delete some items
            if row.shape[0] >= topK:
                sorted_idx = np.argpartition(
                    row, row.shape[0] - topK)[:-topK]
                matrix[row_i][sorted_idx] = 0
    return matrix


def compute_cosine(X, Y, k_filtering, shrinkage=False):
    """
    Returns X_shape[0]xY_shape[1]
    """
    from scipy.sparse.linalg import norm
    chunksize = 1000
    mat_len = X.shape[0]
    S = None
    x_norm = norm(X, axis=1)
    x_norm[x_norm == 0] = 1
    X = X.multiply(sps.csr_matrix(np.reciprocal(x_norm)).transpose())
    y_norm = norm(Y, axis=0)
    y_norm[y_norm == 0] = 1
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
        if shrinkage:
            # Apply shrinkage
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
        S_prime = top_k_filtering(S_prime, k_filtering)
        print("S_prime filtered")
        S_prime.eliminate_zeros()
        # Combine result
        if S is None:
            S = S_prime
        else:
            # stack matrices vertically
            S = sps.vstack([S, S_prime], format="csr")
    return S


def dot_chunked(X, Y, topK, chunksize=1000, n_threads=0):
    """
    Compute dot product of X * Y in chunks of CHUNKSIZE and keep
    only topK elements for each row

    Returns a CSR matrix
    """
    import multiprocessing as mp
    if n_threads == 0:
        n_threads = mp.cpu_count()

    worker_matrix_chunks = []
    worker_chunksize = X.shape[0] // n_threads
    for i in range(0, X.shape[0], worker_chunksize):
        worker_matrix_chunks.append({'start': i, 'end': i + worker_chunksize})

    # Build a list of parameters to ship to pool workers
    separated_tasks = []
    for chunk in worker_matrix_chunks:
        separated_tasks.append([chunk, X, Y, topK, chunksize])

    result = None
    with mp.Pool(n_threads) as pool:
        print('Running {:d} workers...'.format(n_threads))
        submatrices = pool.map(_worker_dot_chunked, separated_tasks)
        submatrices.sort(key=lambda x: x['start'])

        for submatrix in submatrices:
            if result is None:
                result = submatrix['result']
            else:
                result = sps.vstack([result, submatrix['result']])
    return result


def _worker_dot_chunked(params):
    # Unpack parameters
    bounds = params[0]
    X = params[1]
    Y = params[2]
    topK = params[3]
    chunksize = params[4]

    result = None
    start = bounds['start']
    mat_len = bounds['end']
    for chunk in range(start, mat_len, chunksize):
        if chunk + chunksize > mat_len:
            end = mat_len
        else:
            end = chunk + chunksize
        print('Computing dot product for chunk [{:d}, {:d})...'
              .format(chunk, end))
        X_chunk = X[chunk:end]
        sub_matrix = X_chunk.dot(Y)
        sub_matrix = sps.csr_matrix(top_k_filtering(sub_matrix, topK))
        if result is None:
            result = sub_matrix
        else:
            result = sps.vstack([result, sub_matrix], format='csr')
    return {'result': result, 'start': start}

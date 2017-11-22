import scipy.sparse as sps
import numpy as np
from sklearn.cluster import KMeans


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


def cluster_per_n_rating(urm, tg_playlist, ds, n_cluster):
    urm = urm[[ds.get_playlist_index_from_id(x) for x in tg_playlist]]
    n_rating = urm.sum(axis=1)
    rating_cluster = KMeans(n_clusters=n_cluster).fit_predict(n_rating)
    return rating_cluster

def cluster_per_ucm(urm, tg_playlist, ds, n_cluster):
    ucm = ds.build_ucm()
    ucm = ucm[:, [ds.get_playlist_index_from_id(x) for x in tg_playlist]].transpose()
    ucm_cluster = KMeans(n_clusters=n_cluster).fit_predict(ucm)
    return ucm_cluster


def compute_cosine(X, Y, k_filtering, shrinkage=False, n_threads=0, chunksize=100):
    """
    Returns X_shape[0]xY_shape[1]
    """
    if sps.issparse(X):
        from scipy.sparse.linalg import norm
        import multiprocessing as mp

        x_norm = norm(X, axis=1)
        x_norm[x_norm == 0] = 1
        X = X.multiply(sps.csr_matrix(np.reciprocal(x_norm)).transpose())
        y_norm = norm(Y, axis=0)
        y_norm[y_norm == 0] = 1
        Y = Y.multiply(np.reciprocal(y_norm))
        Y_ones = Y.copy()
        Y_ones.data = np.ones_like(Y_ones.data)

        if n_threads == 0:
            n_threads = mp.cpu_count()

        worker_matrix_chunks = []
        worker_chunksize = X.shape[0] // n_threads
        for i in range(0, X.shape[0], worker_chunksize):
            if i + worker_chunksize > X.shape[0]:
                end = X.shape[0]
            else:
                end = i + worker_chunksize
            worker_matrix_chunks.append({'start': i, 'end': end})

        # Build a list of parameters to ship to pool workers
        separated_tasks = []
        for chunk in worker_matrix_chunks:
            separated_tasks.append([chunk,
                                    X,
                                    Y,
                                    Y_ones,
                                    k_filtering,
                                    shrinkage,
                                    chunksize])

        result = None
        with mp.Pool(n_threads) as pool:
            print('Running {:d} workers...'.format(n_threads))
            submatrices = pool.map(_work_compute_cosine, separated_tasks)
            submatrices.sort(key=lambda x: x['start'])

            for submatrix in submatrices:
                if result is None:
                    result = submatrix['result']
                else:
                    result = sps.vstack([result, submatrix['result']])
    else:
        # if not sparse the cosine is only the chunked dot product
        from scipy.linalg import norm
        x_norm = norm(X, axis=1)
        x_norm[x_norm == 0] = 1
        x_norm = np.reciprocal(x_norm)
        x_norm = np.reshape(x_norm, (-1, 1))
        X = np.multiply(X, x_norm)
        y_norm = norm(Y, axis=0)
        y_norm[y_norm == 0] = 1
        Y = np.multiply(Y, np.reciprocal(y_norm))
        result = sps.csr_matrix(dot_chunked(X, Y, k_filtering))
    return result


def _work_compute_cosine(params):
    import os
    # Unpack parameters
    bounds = params[0]
    X = params[1]
    Y = params[2]
    Y_ones = params[3]
    k_filtering = params[4]
    shrinkage = params[5]
    chunksize = params[6]

    start = bounds['start']
    mat_len = bounds['end']

    S = None
    for chunk in range(start, mat_len, chunksize):
        if chunk + chunksize > mat_len:
            end = mat_len
        else:
            end = chunk + chunksize
        print(('[ {:d} ] Building cosine similarity matrix '
               'for [{:d}, {:d})...').format(os.getpid(), chunk, end))

        # First compute similarity
        S_prime = X[chunk:end].tocsr().dot(Y)

        if shrinkage:
            # Apply shrinkage
            X_ones = X[chunk:end]
            X_ones.data = np.ones_like(X_ones.data)
            S_num = X_ones.dot(Y_ones)
            S_den = S_num.copy()
            S_den.data += shrinkage
            S_den.data = np.reciprocal(S_den.data)
            S_prime = S_prime.multiply(S_num).multiply(S_den)

        # Top-K filtering.
        # We only keep the top K similarity weights to avoid considering many
        # barely-relevant neighbors
        S_prime = top_k_filtering(S_prime, k_filtering)
        S_prime.eliminate_zeros()

        # Combine result
        if S is None:
            S = S_prime
        else:
            # stack matrices vertically
            S = sps.vstack([S, S_prime], format="csr")
    return {'result': S, 'start': start}


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
        if i + worker_chunksize > X.shape[0]:
            end = X.shape[0]
        else:
            end = i + worker_chunksize
        worker_matrix_chunks.append({'start': i, 'end': end})

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


def max_normalize(X):
        """
        Normalizes X by rows dividing each row by its max
        """
        max_r = X.max(axis=1)
        max_r.data = np.reciprocal(max_r.data)
        return X.multiply(max_r)


def normalize_by_row(X):
    """
    Normalizes matrix by rows
    """
    # normalize S_mixed
    x_norm = X.sum(axis=1)
    x_norm[x_norm == 0] = 1
    # normalize s
    X = X.multiply(sps.csr_matrix(np.reciprocal(x_norm)))
    return X

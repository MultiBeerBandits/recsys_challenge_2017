from cython cimport floating
from cython.parallel import parallel, prange

import numpy as np
from scipy.sparse import *
cimport numpy as np


cdef class ExplicitCosine:
    # A Cosine similarity matrix evaluator.
    
    cdef double shrinkage
    cdef int top_k
    
    def __cinit__(self, double shrinkage, int top_k):
        self.shrinkage = shrinkage
        self.top_k = top_k

    def __call__(self, X):
        """
        Computes a cosine similarity matrix of sparse matrix X.
        """
        cdef int n_users = X.shape[0]
        cdef int i

        S = lil_matrix((n_users, n_users))
        Xt_global = X.transpose()

        for i in range(n_users):
            row = X.getrow(i)
            # Mask of items rated by user i
            corated_m = (row != 0).transpose()
            # Keep only corated items from other users
            Xt = corated_m.multiply(Xt_global)

            # Compute the norm of user i and divide its ratings by it
            i_norm = np.sqrt(row.multiply(row).sum(axis=1))
            i_norm[i_norm == 0] = 1
            i_norm = np.reciprocal(i_norm)
            row = csr_matrix(row.multiply(i_norm))

            # Compute the norm of other users and divide the urm by it
            urm_norm = np.sqrt(Xt.multiply(Xt).sum(axis=0))
            urm_norm[urm_norm == 0] = 1
            urm_norm = np.reciprocal(urm_norm)
            Xt = csr_matrix(Xt.multiply(urm_norm))

            # Compute unshrinked similarity
            Si = csr_matrix(row.dot(Xt))

            row_ones = row.copy()
            row_ones[row_ones.nonzero()] = 1
            Xt_ones = csr_matrix(Xt.copy())
            Xt_ones[Xt_ones.nonzero()] = 1
            common = row_ones.dot(Xt_ones)
            # Apply numerator
            Si = Si.multiply(common)

            common.data += self.shrinkage
            common.data = 1 / common.data
            # Apply denominator
            Si = Si.multiply(common)

            # Apply top-k
            Si = Si.toarray().squeeze()
            print(Si, Si.shape)
            sorted_idx = np.argpartition(Si, Si.shape[0] - self.top_k)[:-self.top_k]
            print(sorted_idx)
            Si[sorted_idx] = 0

            # Merge Si into final similarity matrix
            S[i] = Si
        S.setdiag(0)
        return S.tocsr()
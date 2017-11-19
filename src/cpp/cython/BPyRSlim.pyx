# distutils: language = c++
# distutils: sources = ../src/BPRSlim.cpp

from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from cpython cimport array
import array
import scipy.sparse as sps

# Fix data type. In our case, since the URM is made of ones, int is fine
# Import Eigen::StorageOptions
cdef extern from "<eigen3/Eigen/Core>" namespace "Eigen":
    ctypedef enum StorageOptions_type "Eigen::StorageOptions":
        STORAGEColMajor = 0,
        RowMajor = 1,
        AutoAlign = 0,
        DontAlign = 2

# Import Eigen::SparseMatrix
cdef extern from "<eigen3/Eigen/SparseCore>" namespace "Eigen":
    cdef cppclass SparseMatrix[T]:
        SparseMatrix() except +
        int *innerIndexPtr()
        int *outerIndexPtr()
        T *valuePtr()
        long rows()
        long cols()
        long nonZeros()

# Import mbb::Utils
cdef extern from "../include/Utils.h" namespace "mbb":
    SparseMatrix[T] buildSparseMatrix[T](vector[int] indPtr,
                                                   vector[T] data,
                                                   vector[int] indices,
                                                   int n_rows,
                                                   int n_cols)

# Import mbb::BPRSlim
cdef extern from "../include/BPRSlim.h" namespace "mbb":
    cdef cppclass BPRSlim[T]:
        BPRSlim() except +
        BPRSlim(SparseMatrix[T] urm,
                double learningRate,
                double lpos,
                double lneg,
                int topK,
                double epochMultiplier) except +
        void fit(int epochs)
        SparseMatrix[double] getParameters()

# Cython wrapper of mbb::BPRSlim
cdef class BPyRSlim:
    cdef BPRSlim[int] bpr
    def __cinit__(self,
                  urm,
                  double learningRate,
                  double lpos,
                  double lneg,
                  int topK,
                  double epochMultiplier):
        assert urm.dtype == int
        # cdef int[:] indPtr = urm.indptr
        # cdef int[:] indices = urm.indices
        # cdef int[:] data = urm.data
        cdef int n_rows = urm.shape[0]
        cdef int n_cols = urm.shape[1]
        cdef SparseMatrix[int] c_urm = buildSparseMatrix[int](urm.indptr,
                                                              urm.data,
                                                              urm.indices,
                                                              n_rows,
                                                              n_cols)
        self.bpr = BPRSlim[int](c_urm, learningRate, lpos, lneg,
                                   topK, epochMultiplier)

    def fit(self, int epochs):
        self.bpr.fit(epochs)

    def getParameters(self):
        cdef SparseMatrix[double] W = self.bpr.getParameters()
        # build memoryviews and return sparse matrix
        cdef int *indPtr = W.outerIndexPtr()
        cdef int *indices = W.innerIndexPtr()
        cdef double *data = W.valuePtr()
        cdef int n_rows = W.rows()
        cdef int n_cols = W.cols()
        cdef int nonZeros = W.nonZeros()
        cdef int i, j, n_elemsAt
        cdef int counter = 0

        cdef int [:] toIndPtr = np.zeros((n_rows + 1), dtype=int)
        cdef int [:] toIndices = np.zeros((nonZeros), dtype=int)
        cdef double [:] toData = np.zeros((nonZeros), dtype=float)

        for i in range(n_rows):
            n_elemsAt = indPtr[i + 1] - indPtr[i]
            # Set indPtr[i]
            toIndPtr[i] = counter
            # Populate data and indices arrays
            for j in range(n_elemsAt):
                toData[counter + j] = data[indPtr[i] + j]
                toIndices[counter + j] = indices[indPtr[i] + j]
            # Increment number of copied values
            counter += n_elemsAt
        return sps.csr_matrix((toData, toIndices, toIndPtr),
                              shape=(n_rows, n_cols))
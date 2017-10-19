from cython cimport integral
import numpy as np
cimport numpy as np
from scipy.sparse import *

cdef class IALS:

    cdef:
        double alfa
        double[:,::1] X, Y
        int features
        double l_reg

    def __cinit__(self, double alfa, double l_reg, int features):
        # confidence factor
        self.alfa = alfa
        # regularization
        self.l_reg = l_reg
        # number of features of the latent factors
        self.features = features

    cpdef fit(self, urm, int n_iterations):
        self.X = np.random.rand(urm.shape[0], self.features)
        self.Y = np.ones((urm.shape[1], self.features))
        cdef int i, u
        cdef int n_items = urm.shape[1]
        cdef int n_users = urm.shape[0]
        cdef int iter_number
        cdef double[:, ::1] XtX, YtY
        cdef double[:, ::1] lI = np.eye(self.features) * self.l_reg
        cdef int[:] urm_data = urm.data
        cdef int[:] urm_indices = urm.indeces
        cdef int[:] urm_indptr = urm.indptr

        for iter_number in range(n_iterations):
            XtX = self.X.transpose().dot(self.X)
            # iterate over items
            for i in range(n_items):
                # build Ri, diag matrix of UxU with ri in the diag
                ri = urm.getcol(i).todense()
                Ri = diags(ri)
                aXtRX = self.alfa * self.X.transpose().dot(Ri).dot(self.X)
                A = XtX + aXtRX
                A = A + lI
                b = self.X.transpose().dot(ri + self.alfa * ri)
                self.Y[i] = np.linalg.solve(A,b)
                print("Iteration_item", i)
            YtY = self.Y.transpose().dot(self.Y)
            # iterate over users
            for u in range(n_users):
                # build Ri, diag matrix of UxU with ri in the diag
                ru = urm.getrow(u).todense()
                Ru = diags(ru)
                aYtRY = self.alfa * self.Y.transpose().dot(Ru).dot(self.Y)
                A = YtY + aYtRY
                A = A + lI
                b = self.Y.transpose().dot(ru + self.alfa * ru)
                self.X[i] = np.linalg.solve(A,b)
                print("Iteration_user", u)
            print("Iteration", iter_number)










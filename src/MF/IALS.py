from scipy.sparse import *
import numpy as np
import _thread


class IALS():
    """docstring for NMF"""

    def __init__(self, urm, features, learning_steps):
        # Number of latent factors
        self.features = features

        # Number of learning iterations
        self.learning_steps = learning_steps

        # Store a reference to URM
        self.urm = urm

        # User factor matrix
        self.X = np.random.rand(urm.shape[0], features) * 0.01
        # Item factor matrix
        self.Y = np.random.rand(urm.shape[1], features) * 0.01

    def fit(self, u_reg, v_reg):
        # try parallelization (?)
        threads = []
        results = [None] * 2
        for i in range(self.learning_steps):
            t_user = _thread.start_new_thread(self.als, (self.urm, self.X.copy(), self.Y, u_reg, results, 0))
            t_item = _thread.start_new_thread(self.als, (self.urm, self.Y.copy(), self.X, u_reg, results, 1))
            # Add threads to thread list
            threads.append(t_user)
            threads.append(t_item)

            # Wait for all threads to complete
            for t in threads:
                t.join()
            threads.clear()
            # get results
            self.X = results[0]
            self.Y = results[1]
            print("iteration ",i)

    def predict(self, target_playlist, target_tracks, dataset, at=5):
        self.X = self.X[[dataset.get_playlist_index_from_id(x)
                         for x in target_playlist]]
        self.Y = self.Y[[dataset.get_track_index_from_id(x)
                         for x in target_tracks]]
        self.urm = self.urm[[dataset.get_playlist_index_from_id(x)
                             for x in target_playlist]]
        self.urm = self.urm[:, [dataset.get_track_index_from_id(x)
                                for x in target_tracks]]
        R_hat = self.X.dot(self.Y.transpose())
        R_hat[self.urm.nonzero()] = 0
        R_hat = csr_matrix(R_hat)
        recs = {}
        for row_i in range(R_hat.shape[0]):
            pl_id = target_playlist[row_i]
            row = R_hat.data[R_hat.indptr[row_i]:R_hat.indptr[row_i + 1]]
            sorted_mask = np.flip(row.argsort(), axis=0)[0:at]
            track_cols = [R_hat.indices[R_hat.indptr[row_i] + x]
                          for x in sorted_mask]
            tracks_ids = [target_tracks[x] for x in track_cols]
            recs[pl_id] = tracks_ids
        return recs

    def als(Cui, X, Y, regularization, result, index):
        users, factors = X.shape
        YtY = Y.T.dot(Y)
        for u in range(users):
            # accumulate YtCuY + regularization * I in A
            A = YtY + regularization * np.eye(factors)

            # accumulate YtCuPu in b
            b = np.zeros(factors)

            for i, confidence in nonzeros(Cui, u):
                factor = Y[i]
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor

            # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
            X[u] = np.linalg.solve(A, b)
        # save result back
        result[index] = X

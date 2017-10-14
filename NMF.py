from scipy.sparse import *
import numpy as np


class NMF():
    """docstring for NMF"""

    def __init__(self, urm, features, learning_steps):
        # Number of latent factors
        self.features = features

        # Number of learning iterations
        self.learning_steps = learning_steps

        # Store a reference to URM
        self.urm = urm

        # User factor matrix
        self.U = np.zeros((urm.shape[0], features))
        self.U.fill(0.1)

        # Item factor matrix
        self.V = np.zeros((urm.shape[1], features))
        self.V.fill(0.1)

    def fit(self, u_reg, v_reg):
        for i in range(self.learning_steps):
            print(('[NMF][Fit with u_reg = ' + str(u_reg) +
                   ', v_reg = ' + str(v_reg) + '] Step ' + str(i)))
            # Update user factor matrix
            u_num = self.urm.dot(self.V)
            u_num = u_num - self.U * u_reg
            u_den = self.V.transpose().dot(self.V)
            u_den = self.U.dot(u_den)
            u_den += 1e-10
            U_prime = u_num / u_den
            U_prime = np.multiply(U_prime, self.U)
            U_prime[(U_prime < 0)] = 0
            self.U = U_prime
            print(('[NMF][Fit with u_reg = ' + str(u_reg) +
                   ', v_reg = ' + str(v_reg) +
                   '] User factor matrix updated...'))

            # Update item factor matrix
            v_num = self.urm.transpose().dot(self.U)
            v_num = v_num - self.V * v_reg
            v_den = self.U.transpose().dot(self.U)
            v_den = self.V.dot(v_den)
            v_den += 1e-10
            v_den = np.reciprocal(v_den)
            v_num = np.multiply(v_num, v_den)
            v_num = np.multiply(v_num, self.V)
            v_num[(v_num < 0)] = 0
            self.V = v_num
            print(('[NMF][Fit with u_reg = ' + str(u_reg) +
                   ', v_reg = ' + str(v_reg) +
                   '] Item factor matrix updated...'))

    def predict(self, target_playlist, target_tracks, dataset, at=5):
        self.U = self.U[[dataset.get_playlist_index_from_id(x)
                         for x in target_playlist]]
        self.V = self.V[[dataset.get_track_index_from_id(x)
                         for x in target_tracks]]
        self.urm = self.urm[[dataset.get_playlist_index_from_id(x)
                             for x in target_playlist]]
        self.urm = self.urm[:, [dataset.get_track_index_from_id(x)
                                for x in target_tracks]]
        R_hat = self.U.dot(self.V.transpose())
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

from src.utils.loader import *
from scipy.sparse import *
import numpy as np


class UserBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=130, k_filtering=95):
        """
        urm: user rating matrix
        target playlist is a list of playlist id
        target_tracks is a list of track id
        shrinkage: shrinkage factor for significance weighting
        """
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        print("target playlist ", len(self.pl_id_list))
        print("target tracks ", len(self.tr_id_list))
        # calculate similarity between users:
        # S_ij = (sum for k belonging to items r_ik*r_jk)/norm of two vectors
        # first calculate norm
        # sum over columns (obtaining a column vector)
        norm = np.sqrt(urm.sum(axis=1))
        norm[(norm == 0)] = 1
        # divide urm by the norm
        S_num = urm.dot(urm.transpose())
        S = S_num.multiply(csr_matrix(1 / norm))
        S = S.multiply(csr_matrix(1 / norm.T))
        # zero out diagonal
        # in the diagonal there is the sim between i and i (1)
        S.setdiag(0)
        S.eliminate_zeros()
        # keep only rows of target playlist
        S = S[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        print("Similarity matrix done.")
        # apply shrinkage factor:
        # Let I_uv be the set of items rated both by users u and v
        # Let H be the shrinkage factor
        #   Multiply element-wise for the matrix of I_uv (ie: the sim matrix)
        #   Divide element-wise for the matrix of I_uv incremented by H
        shr_num = S
        shr_den = S.copy()
        shr_den.data += shrinkage
        shr_den.data = 1 / shr_den.data
        S = S.multiply(shr_num)
        S = csr_matrix(S.multiply(shr_den))
        # Top-K filtering.
        # We only keep the top K similarity weights to avoid considering many
        # barely-relevant neighbors
        for row_i in range(0, S.shape[0]):
            row = S.data[S.indptr[row_i]:S.indptr[row_i + 1]]

            sorted_idx = row.argsort()[:-k_filtering]
            row[sorted_idx] = 0
        S.eliminate_zeros()
        # get a column vector of the similarities of user i (i is the row)
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(csr_matrix(1 / s_norm))
        # compute ratings
        R_hat = S.dot(urm).tocsr()
        print("R_hat done")
        # apply mask for eliminating already rated items
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        # eliminate tracks that are not target
        R_hat = R_hat[:, [dataset.get_track_index_from_id(
            x) for x in self.tr_id_list]]
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

    def predict(self, at=5):
        """
        returns a dictionary of
        'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
        """
        recs = {}
        for i in range(0, self.R_hat.shape[0]):
            pl_id = self.pl_id_list[i]
            pl_row = self.R_hat.data[self.R_hat.indptr[i]:
                                     self.R_hat.indptr[i + 1]]
            # get top 5 indeces. argsort, flip and get first at-1 items
            sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
            track_cols = [self.R_hat.indices[self.R_hat.indptr[i] + x]
                          for x in sorted_row_idx]
            tracks_ids = [self.tr_id_list[x] for x in track_cols]
            recs[pl_id] = tracks_ids
        return recs

    def get_model(self):
        """
        Returns the complete R_hat
        """
        return self.R_hat.copy()

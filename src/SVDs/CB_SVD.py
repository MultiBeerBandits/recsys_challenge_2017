from src.utils.loader import *
from scipy.sparse import *
# from scipy.sparse.linalg import svds 
# sparsesvd should be faster!
from sparsesvd import sparsesvd
import numpy as np
import numpy.linalg as LA
from sklearn.feature_extraction.text import TfidfTransformer


class ContentBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=95, features=200):
        """
        urm: user rating matrix
        target playlist is a list of playlist id
        target_tracks is a list of track id
        shrinkage: shrinkage factor for significance weighting
        S = ICM' ICM
        R = URM S
        In between eliminate useless row of URM and useless cols of S
        """
        # initialization

        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        S = None
        print("target playlist ", len(self.pl_id_list))
        print("target tracks ", len(self.tr_id_list))
        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm()
        # Apply SVD on ICM
        # v actually is v'. It's features*items
        _, s, v = sparsesvd(icm.tocsc(), features)

        print("SVD Done!")
        print(v.shape)
        # calculate similarity between items:
        # S_ij=(sum for k belonging to attributes t_ik*t_jk)/norm_i * norm_k
        # first calculate norm
        # norm over rows (obtaining a row vector)
        s = np.diag(s.data)
        # the new icm is (v' * s)'. in this way is feature*items
        icm = v.transpose().dot(s).transpose()
        norm = LA.norm(icm, axis=0)
        norm[(norm == 0)] = 1
        # normalize
        icm = np.multiply(icm, np.reciprocal(norm))
        print("Normalization done!")
        icm_t = icm.transpose()
        # clean the transposed matrix, we do not need tracks not target
        icm_t = icm_t[[dataset.get_track_index_from_id(x)
                       for x in self.tr_id_list]]
        S_prime = icm_t.dot(icm)
        print("S prime computed")
        # compute common features
        icm_t_ones = icm_t
        icm_t_ones[icm_t.nonzero()] = 1
        icm_ones = icm
        icm_ones[icm_ones.nonzero()] = 1
        S_num = icm_t_ones.dot(icm_ones)
        S_den = S_num.copy()
        S_den += shrinkage
        S_den = np.reciprocal(S_den)
        S_prime = np.multiply(S_prime, S_num)
        S_prime = np.multiply(S_prime, S_den)
        print("S_prime applied shrinkage")
        indices = np.argpartition(S_prime, S_prime.shape[1] - k_filtering, axis=1)[:, :-k_filtering] # keep all rows but until k columns
        for i in range(S_prime.shape[0]):
            S_prime[i, indices[i]] = 0
        S_prime = csr_matrix(S_prime)

        print("S_prime filtered")
        print("Similarity matrix ready, let's normalize it!")
        # zero out diagonal
        # in the diagonal there is the sim between i and i (1)
        # maybe it's better to have a lil matrix here
        S = S_prime
        # S.setdiag(0)
        # keep only target rows of URM and target columns
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        # get a column vector of the similarities of item i (i is the row)
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(np.reciprocal(s_norm))
        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose())
        print("R_hat done")
        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
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

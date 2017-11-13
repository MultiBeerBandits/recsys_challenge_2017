from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from sklearn.metrics.pairwise import pairwise_distances
from src.utils.sim import computeSim


class ContentBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=200, test_dict={}, content_penalty=10):
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
        self.dataset = dataset
        print("CBF started")
        # get ICM from dataset, assume it already cleaned
        icm = csr_matrix(dataset.build_icm())
        i_sim = computeSim(icm.transpose(), icm, k_filtering=1000)
        i_sim_norm = i_sim.sum(axis=1)
        # normalize
        i_sim = i_sim.multiply(np.reciprocal(i_sim_norm))
        # compute augmented urm
        aUrm = urm.dot(i_sim.transpose())
        # scale pseudo rating of each user:
        # Rationale: more rating -> more correct Content Based
        # Multiply rating of u: #u / (#u + H)
        rating_number = urm.sum(axis=1)
        rating_den = rating_number + content_penalty
        scaling = np.multiply(rating_number, np.reciprocal(rating_den))
        aUrm = csr_matrix(aUrm.multiply(scaling))
        print("Augmented URM ready!")
        # restore original ratings
        aUrm[urm.nonzero()] = 1
        # keep only 500 items for each user
        k_filtering = 2000
        for row_i in range(0, aUrm.shape[0]):
            row = aUrm.data[aUrm.indptr[row_i]:aUrm.indptr[row_i + 1]]
            # delete some items
            if row.shape[0] >= k_filtering:
                sorted_idx = np.argpartition(
                    row, row.shape[0] - k_filtering)[:-k_filtering]
                row[sorted_idx] = 0
        aUrm.eliminate_zeros()
        # do collaborative filtering on aUrm
        # start with user feature matrix
        ufm = urm.dot(icm.transpose())
        # Iu contains for each user the number of tracks rated
        Iu = urm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # Add a term for shrink
        Iu = Iu + 10
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        ufm = csr_matrix(ufm.multiply(Iu)).transpose()
        ucm = csr_matrix(dataset.build_ucm())
        # UCM is user x features
        ucm = vstack([ucm, aUrm.transpose(), ufm], format='csr').transpose()
        # compute sim only for target users, u_sim is tg_user x users
        u_sim = computeSim(ucm[[dataset.get_playlist_index_from_id(x)
                                for x in self.pl_id_list]], ucm.transpose())
        R_hat = u_sim.dot(aUrm[:, [dataset.get_track_index_from_id(x)
                                   for x in self.tr_id_list]])
        urm_red = urm[[dataset.get_playlist_index_from_id(x)
                       for x in self.pl_id_list]]
        urm_red = urm_red[:, [dataset.get_track_index_from_id(x)
                              for x in self.tr_id_list]]
        R_hat[urm_red.nonzero()] = 0
        R_hat.eliminate_zeros()

        print("R_hat done")
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

    def getW(self):
        """
        Returns the similary matrix with dimensions I x I
        S is IxT
        """
        W = lil_matrix((self.S.shape[0], self.S.shape[0]))
        W[:, [self.dataset.get_track_index_from_id(
            x) for x in self.tr_id_list]] = self.S
        return W

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

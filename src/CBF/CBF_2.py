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

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=200, test_dict={}):
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
        S = None
        print("CBF started")
        # get ICM from dataset, assume it already cleaned
        icm = csr_matrix(dataset.build_icm())
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
        ucm = vstack([ucm, urm.transpose(), ufm], format='csr')
        u_sim = computeSim(ucm.transpose(), ucm)
        u_sim_norm = u_sim.sum(axis=1)
        # normalize
        u_sim = u_sim.multiply(np.reciprocal(u_sim_norm))
        # compute augmented urm
        aUrm = u_sim.dot(urm)
        print("Augmented URM ready!")
        aUrm[urm.nonzero()] = 1
        # keep only 500 items for each user
        k_filtering = 0
        for row_i in range(0, aUrm.shape[0]):
            row = aUrm.data[aUrm.indptr[row_i]:aUrm.indptr[row_i + 1]]
            # augment only for users with few items rated
            k_filtering = 1000
            if row.shape[0] >= k_filtering:
                sorted_idx = np.argpartition(
                    row, row.shape[0] - k_filtering)[:-k_filtering]
                row[sorted_idx] = 0
        aUrm.eliminate_zeros()
        # aUrm.data = np.ones_like(aUrm.data)
        # put all to one:
        icm = dataset.add_playlist_to_icm(icm, aUrm, 0.5)
        S = computeSim(icm.transpose()[[dataset.get_track_index_from_id(x)
                                       for x in self.tr_id_list]], icm).transpose()
        R_hat = aUrm[[dataset.get_playlist_index_from_id(x)
                     for x in self.pl_id_list]].dot(S)
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

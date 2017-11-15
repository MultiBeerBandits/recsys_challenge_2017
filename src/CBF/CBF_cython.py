from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.CBF.cosineSim import Cosine_Similarity


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
        icm = dataset.build_icm()
        # icm = self.applytfidf(icm)
        icm = dataset.add_playlist_to_icm(icm, urm, 0.5)
        print("SHAPE of ICM: ", icm.shape)
        cosine_cython = Cosine_Similarity(icm, TopK=500)

        start_time = time.time()

        S = cosine_cython.compute_similarity()

        print("Similarity computed in {:.2f} seconds".format(time.time()-start_time))

        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        # compute ratings
        print("Similarity matrix ready, let's normalize it!")
        # compute augmented urm
        aUrm = urm.dot(S)
        # scale pseudo rating of each user:
        # Rationale: more rating -> more correct Content Based
        # Multiply rating of u: #u / (#u + H)
        rating_number = urm.sum(axis=1)
        rating_den = rating_number + 10
        scaling = np.multiply(rating_number, np.reciprocal(rating_den))
        aUrm = csr_matrix(aUrm.multiply(scaling))
        print("Augmented URM ready!")
        # restore original ratings
        aUrm[urm.nonzero()] = 1
        aUrm = top_k_filtering(aUrm, topK=500)
        # do collaborative filtering on aUrm
        # compute sim only for target users, u_sim is tg_user x users
        cosine_cython = Cosine_Similarity(aUrm.transpose(), TopK=500)

        start_time = time.time()

        u_sim = cosine_cython.compute_similarity().transpose()[[dataset.get_playlist_index_from_id(x)
                       for x in self.pl_id_list]]
        u_sim_norm = u_sim.sum(axis=1)
        # normalize
        u_sim = u_sim.multiply(np.reciprocal(u_sim_norm))
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
        return self.S.tocsr()

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

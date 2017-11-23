from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering


class ContentBasedFiltering():
    # The prediction is done S_user*(1-alfa) + S_cbf(alfa)

    def __init__(self, alfa):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

        # save alfa value
        self.alfa = alfa


    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=200):
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

        # get ICM from dataset
        icm = dataset.build_icm()

        # add urm
        icm = dataset.add_playlist_to_icm(icm, urm, 0.4)

        # add n_ratings to icm
        icm = dataset.add_tracks_num_rating_to_icm(icm, urm)

        # build user content matrix
        ucm = dataset.build_ucm()

        # build item user-feature matrix: UFxI
        iucm = ucm.dot(urm)

        iucm_norm = urm.sum(axis=0)
        iucm_norm[iucm_norm == 0] = 1
        iucm_norm = np.reciprocal(iucm_norm)
        iucm = csr_matrix(iucm.multiply(iucm_norm))

        S_user = compute_cosine(iucm.transpose()[[dataset.get_track_index_from_id(x)
                                                  for x in self.tr_id_list]],
                                iucm, k_filtering=k_filtering, shrinkage=shrinkage)

        # To filter or not to filter? Who knows?

        # title = dataset.build_title_matrix(iucm)
        # title = top_k_filtering(title.transpose(), topK=100)
        # title.data = np.ones_like(title.data)
        # title = title.multiply(0.05)
        # # owner = dataset.build_owner_matrix(iucm)
        # # owner = top_k_filtering(owner.transpose(), topK=500)
        # # owner.data = np.ones_like(owner.data)
        # # owner = owner.multiply(0.05)
        # created_at = dataset.build_created_at_matrix(iucm)
        # print(created_at.shape)
        # created_at = top_k_filtering(created_at.transpose(), topK=10)
        # created_at.data = np.ones_like(created_at.data)
        # created_at = created_at.multiply(0.01)

        # compute cosine similarity (only for tg tracks) wrt to all tracks
        S = compute_cosine(icm.transpose()[[dataset.get_track_index_from_id(x)
                                            for x in self.tr_id_list]],
                           icm, k_filtering=k_filtering, shrinkage=shrinkage)

        # compute a weighted average
        S = S.multiply(self.alfa) + S_user.multiply(1 - self.alfa)

        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))

        # keep only target rows of URM and target columns
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]

        # save S
        self.S = S.transpose()

        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose()).tocsr()

        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()

        print("R_hat done")
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

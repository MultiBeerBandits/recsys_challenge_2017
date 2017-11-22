from src.utils.loader import *
from scipy.sparse import *
import numpy as np
from src.utils.BaseRecommender import BaseRecommender
from src.utils.matrix_utils import compute_cosine, normalize_by_row


class ItemBasedFiltering(BaseRecommender):

    def __init__(self, shrinkage=50, k_filtering=200):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

        self.shrinkage = shrinkage
        self.k_filtering = k_filtering

    def fit(self, urm, target_playlist, target_tracks, dataset):
        """
        urm: user rating matrix
        target playlist is a list of playlist id
        target_tracks is a list of track id
        shrinkage: shrinkage factor for significance weighting
        """
        # initialize model fields
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)

        urm = csr_matrix(urm)

        # Build collaborative similarity
        S_cf = compute_cosine(urm.transpose()[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]], urm, k_filtering=self.k_filtering, shrinkage=self.shrinkage)

        # normalize
        S_cf = normalize_by_row(S_cf)

        self.R_hat = urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S_cf.transpose())

        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        self.R_hat[urm_cleaned.nonzero()] = 0
        self.R_hat.eliminate_zeros()

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

    def getR_hat(self):
        return self.R_hat

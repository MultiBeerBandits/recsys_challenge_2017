from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering, max_normalize, normalize_by_row


class ContentBasedFiltering():
    # The prediction is done S_user*(1-alfa) + S_cbf(alfa)

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=50):
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

        urm = urm.tocsr()

        # get ICM from dataset
        icm = dataset.build_icm()

        # add n_ratings to icm
        icm = dataset.add_tracks_num_rating_to_icm(icm, urm)

        # add urm
        icm = dataset.add_playlist_to_icm(icm, urm, 0.4)

        # compute cosine similarity (only for tg tracks) wrt to all tracks
        S = compute_cosine(icm.transpose(),
                           icm, k_filtering=k_filtering, shrinkage=shrinkage)

        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))

        # save S
        self.S = S.transpose()

        # compute ratings
        R_hat = urm.dot(S.transpose()).tocsr()

        # keep only top n ratings of the prediction
        R_hat[urm.nonzero()] = 0
        R_hat.eliminate_zeros()

        R_hat = top_k_filtering(R_hat, 20)

        # re add initial ratings
        R_hat[urm.nonzero()] = 1

        # do collaborative filtering
        S_cf = compute_cosine(R_hat.transpose()[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]], R_hat, k_filtering=k_filtering, shrinkage=shrinkage)

        # normalize
        S_cf = normalize_by_row(S_cf)

        self.R_hat = csr_matrix(urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S_cf.transpose()))

        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        self.R_hat[urm_cleaned.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        print("R_hat done")
        self.R_hat = csr_matrix(self.R_hat)

        print(self.R_hat.shape, len(self.pl_id_list), len(self.tr_id_list))

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


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2, 0.1)
    ds.set_playlist_attr_weights(0.2, 0, 0.5, 0.05, 0.05)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = cbf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)

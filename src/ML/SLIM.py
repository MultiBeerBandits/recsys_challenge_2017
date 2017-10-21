from scipy.sparse import *
from sklearn.linear_model import ElasticNet, SGDRegressor
import numpy as np

from src.utils.loader import *
from src.utils.evaluator import *


class SLIM():
    """docstring for SLIM"""

    def __init__(self, l1_reg=0.00001, l2_reg=0.000001):
        self.alpha = l1_reg + l2_reg
        self.l1_ratio = l1_reg / self.alpha
        self.W = None
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None

    def fit(self, urm, target_items, target_users, dataset):
        # Store target playlists and tracks
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)
        # Initialize W matrix
        self.W = lil_matrix((urm.shape[1], urm.shape[1]))

        # We access URM by columns
        urm = urm.tocsc()

        # For each target item train an ElasticNet model
        count = 0
        for t in [dataset.get_track_index_from_id(x)
                  for x in target_items]:
            if count % 100 == 0:
                print(count, '/', len(target_items), 'ElasticNet trained...')
            # Zero-out the t-th column to meet the w_tt = 0 constraint
            urm_z = urm.copy()
            urm_z.data[urm_z.indptr[t]:urm_z.indptr[t + 1]] = 0
            urm_z.eliminate_zeros()

            # Prepare data for model fit
            model = ElasticNet(alpha=self.alpha,
                               fit_intercept=False,
                               l1_ratio=self.l1_ratio,
                               positive=True)
            r_t = urm.getcol(t).toarray().ravel()

            # Fit
            model.fit(urm_z, r_t)
            # print(model.coef_.shape)
            self.W[:, t] = model.sparse_coef_.transpose()
            count += 1
            # print('ElasticNet model for item', t, 'fit...')
            # print(repr(self.W[:, t]))
            # print(self.W[:, t][self.W[:, t].nonzero()][:10])

        # clean urm from unwanted users
        urm = urm[[dataset.get_playlist_index_from_id(x)
                   for x in self.pl_id_list]]

        # Compute prediction matrix (n_target_users X n_items)
        self.R_hat = urm.dot(self.W).tocsr()
        print('R_hat evaluated...')

        # Clean R_hat from already rated entries
        self.R_hat[urm.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        # Keep only target_item columns
        self.R_hat = self.R_hat[:, [dataset.get_track_index_from_id(x)
                                    for x in target_items]]

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


if __name__ == '__main__':
    ds = Dataset()
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    ubf = SLIM()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ubf.fit(urm, list(tg_tracks), list(tg_playlist), ds)
        recs = ubf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5", map_at_five)

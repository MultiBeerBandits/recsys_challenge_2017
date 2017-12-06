from lightfm import LightFM
import lightfm.evaluation
import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *
from math import sqrt, ceil
from src.utils.matrix_utils import top_k_filtering


class WARP():

    def __init__(self):
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None
        self.model = None
        self.icm_t = None
        self.ucm_t = None
        self.urm = None

# MAP@5: 0.029395310261630093 after 5 epochs with 500 components
    def fit(self, urm, dataset, tg_playlist, tg_tracks, no_components=50, n_epochs=5, item_alpha=1e-3, l_rate=1e-3):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        self.urm = urm

        # Initialize model.
        # Need an identity matrix stacked horizontaly
        # Unknown reason, refer to:
        # https://github.com/lyst/lightfm/blob/master/lightfm/lightfm.py
        if self.icm_t is None:
            id_item = eye(urm.shape[1], urm.shape[1]).tocsr()
            icm = dataset.build_icm_2().tocsr()
            self.icm_t = hstack((id_item,
                                 icm.transpose() * 40)).tocsr().astype(np.float32)

        if self.ucm_t is None:
            id_playlist = eye(urm.shape[0], urm.shape[0]).tocsr()
            ucm = dataset.build_ucm().tocsr()
            self.ucm_t = hstack((id_playlist,
                                 ucm.transpose())).tocsr().astype(np.float32)

        #no_components = ceil(sqrt(urm.shape[0] + urm.shape[1] + self.icm_t.shape[0] + self.ucm_t.shape[0]))
        print(no_components)

        if self.model is None:
            self.model = LightFM(loss='warp',
                                 learning_schedule='adadelta',
                                 learning_rate=l_rate,
                                 random_state=2016,
                                 no_components=no_components,
                                 item_alpha=item_alpha,
                                 max_sampled=100,
                                 user_alpha=item_alpha)

        self.model.fit_partial(urm,
                               epochs=n_epochs,
                               item_features=self.icm_t,
                               user_features=self.ucm_t,
                               num_threads=4,
                               verbose=True)
        print("Training finished")

        tg_items = np.array(
            [dataset.get_track_index_from_id(x) for x in tg_tracks])
        tg_users = np.array([dataset.get_playlist_index_from_id(x)
                             for x in tg_playlist])

        n_tg_items = len(tg_items)
        n_tg_users = len(tg_users)

        # repeat every user n_tg_items times
        user_ids = np.repeat(tg_users, n_tg_items)

        # tile track indices n_tg_user times
        item_ids = np.tile(tg_items, n_tg_users)

        predictions = self.model.predict(user_ids,
                                         item_ids,
                                         num_threads=4,
                                         item_features=self.icm_t,
                                         user_features=self.ucm_t)

        self.R_hat = self.build_R_hat(predictions,
                                      user_ids,
                                      item_ids,
                                      tg_items,
                                      tg_users)

    def predict(self, at=5):
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

    def build_R_hat(self, predictions, user_ids, item_ids, tg_items, tg_users):
        R_hat = coo_matrix((predictions, (user_ids, item_ids)),
                           shape=self.urm.shape).tocsr()

        # clean urm
        R_hat[urm.nonzero()] = 0
        R_hat.eliminate_zeros()

        # keep only target user and tg items
        R_hat = R_hat[tg_users]
        R_hat = R_hat[:, tg_items]

        R_hat = top_k_filtering(R_hat, 50)

        return R_hat



if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True, weight_tag=False)
    ds.set_track_attr_weights_2(1, 1, 1, 1, 1, num_rating_weight=1, inferred_album=1, inferred_duration=1, inferred_playcount=1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        warp = WARP()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        test_urm = ev.get_test_matrix(i, ds)
        for epoch in range(50):
            warp.fit(urm, ds, list(tg_playlist), list(tg_tracks))
            recs = warp.predict()
            ev.evaluate_fold(recs)
        # ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

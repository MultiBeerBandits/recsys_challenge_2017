import implicit
from scipy.sparse import *
import numpy as np
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering
from src.utils.BaseRecommender import BaseRecommender


class IALS(BaseRecommender):
    """Results:
    0,027 with 100, 50, 1e-6, 40
    0,033 with 100, 50, 1e-4, 400
    0.049 with 200, 50, 1e-4, 400
    0.051 with 200, 50, 1e-4, 800
    0.073 with 500, 50, 1e-4, 800
    0.071 with 500, 50, 1e-5, 800
    0.069 with 500, 70. 1e-4, 800
    """

    def __init__(self, features, learning_steps, reg, confidence):
        # Number of latent factors
        self.features = features

        # Number of learning iterations
        self.learning_steps = learning_steps

        # Store a reference to URM
        self.urm = None

        # regularization weight
        self.reg = reg

        # model for making recommendation
        self.model = None

        # confidence, gets multiplied by urm
        self.confidence = confidence

        # reference to tg playlist and tg tracks
        self.pl_id_list = None
        self.tr_id_list = None

    def fit(self, urm, tg_playlist, tg_tracks, dataset):
        self.urm = urm
        self.pl_id_list = list(tg_playlist)
        self.tr_id_list = list(tg_tracks)
        # initialize a model
        self.model = implicit.als.AlternatingLeastSquares(factors=self.features, regularization=self.reg, iterations=self.learning_steps)

        # train the model on a sparse matrix of item/user/confidence weights
        self.model.fit(self.urm.transpose().multiply(self.confidence))

        print(self.model.item_factors.shape)
        print(self.model.user_factors.shape)

        # keep only useful user and item factors
        user_factors = self.model.user_factors[[dataset.get_playlist_index_from_id(x) for x in tg_playlist]]
        item_factors = self.model.item_factors[[dataset.get_track_index_from_id(x) for x in tg_tracks]]

        # clean urm
        self.urm = self.urm[[dataset.get_playlist_index_from_id(x) for x in tg_playlist]]
        self.urm = self.urm[:, [dataset.get_track_index_from_id(x) for x in tg_tracks]]

        self.R_hat = user_factors.dot(item_factors.transpose())

        # clean from useless row and columns
        self.R_hat[self.urm.nonzero()] = 0
        self.R_hat = top_k_filtering(self.R_hat, topK=500)
        # convert to csr
        self.R_hat = csr_matrix(self.R_hat)

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

    def getR_hat(self):
        return self.R_hat


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ials = IALS(500, 50, 1e-4, 800)
        ials.fit(list(tg_playlist), list(tg_tracks), ds)
        recs = ials.predict()
        ev.evaluate_fold(recs)
        rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, 10)
        mpc = ev.map_per_cluster(tg_playlist, rating_cluster, 10)
        visualize_2d(range(10), mpc, "Cluster N Rating", "Map@5", "MAP per cluster POP")
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

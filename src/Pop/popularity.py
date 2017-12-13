from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering, max_normalize, cluster_per_n_rating
from src.utils.BaseRecommender import BaseRecommender


class Popularity(BaseRecommender):

    def __init__(self, topK=10000):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

        self.topK = topK

    def fit(self, urm, target_playlist, target_tracks, dataset):
        """
        urm: user rating matrix
        target playlist is a list of playlist id
        target_tracks is a list of track id
        """
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.dataset = dataset
        urm = csr_matrix(urm)
        self.R_hat = lil_matrix((len(self.pl_id_list), len(self.tr_id_list)), dtype=urm.dtype)

        print("Pop started")

        # sum all ratings of urm
        pop = csr_matrix(urm[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]].sum(axis=0))

        # convert to csr matrix
        pop = csr_matrix(top_k_filtering(pop, topK=self.topK))
        # scale function, high popularity have the same importance
        pop.data = np.log(pop.data)

        # build R_hat by stacking pop as the number of target user
        # use low level structure of csr matrix in order to be fast
        for i in range(len(self.pl_id_list)):
            self.R_hat[i] = pop

        print("R_hat built")
        # maybe we need to normalize here to obtain a more precise value
        # for the probability of like
        self.R_hat = self.R_hat.tocsr()

        # clean urm
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        # clean urm from already rated items
        self.R_hat[urm_cleaned.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        # normalize urm
        # self.R_hat = max_normalize(self.R_hat)


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

    def getR_hat(self):
        return self.R_hat


def main():
    # best_map = 0
    # best_album_w = 0
    # best_artist_w = 0
    # shr_w = 100
    # k_f = 50
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    pop = Popularity()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        pop.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = pop.predict()
        ev.evaluate_fold(recs)
        rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, 10)
        mpc = ev.map_per_cluster(tg_playlist, rating_cluster, 10)
        visualize_2d(range(10), mpc, "Cluster N Rating", "Map@5", "MAP per cluster POP")
    map_at_five = ev.get_mean_map()
    print("MAP@5 ", map_at_five)


if __name__ == '__main__':
    main()

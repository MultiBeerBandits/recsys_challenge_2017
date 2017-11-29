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
from src.utils.plotter import visualize_2d
from src.Pop.popularity import Popularity
from src.CBF.CBF import ContentBasedFiltering


class PopularityCBF(BaseRecommender):

    def __init__(self, topK=500):
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

        pop = Popularity()
        pop.fit(urm, target_playlist, target_tracks, dataset)

        R_hat_pop = pop.getR_hat()

        cbf = ContentBasedFiltering()
        cbf.fit(urm, target_playlist, target_tracks, dataset)

        R_hat_cbf = cbf.getR_hat()
        R_hat_cbf = top_k_filtering(R_hat_cbf, 10)

        # mix the two predictions
        self.R_hat = R_hat_cbf.multiply(R_hat_pop)

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
    ds.set_track_attr_weights_2(1, 0.9, 0.2, 0.2, 0.2, 0.1, 0.8, 0.1, 0.1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    pop = PopularityCBF()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        pop.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = pop.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 ", map_at_five)


if __name__ == '__main__':
    main()

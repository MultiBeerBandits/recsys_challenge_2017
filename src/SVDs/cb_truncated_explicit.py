from src.utils.loader import *
from scipy.sparse import *
from scipy.sparse.linalg import svds
import numpy as np
import numpy.linalg as LA
from src.utils.evaluator import *
import src.utils.similarity as similarity


class ContentBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=100, k_filtering=95, features=50):
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
        S = None
        print("target playlist ", len(self.pl_id_list))
        print("target tracks ", len(self.tr_id_list))
        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm()
        # Apply SVD on ICM
        _, s, icm = svds(icm, features, return_singular_vectors=True)

        print("SVD Done!")
        # calculate similarity between items:
        # S_ij=(sum for k belonging to attributes t_ik*t_jk)/norm_i * norm_k
        # first calculate norm
        # norm over rows (obtaining a row vector)
        s = np.diag(s.data)
        icm = s.dot(icm)
        icm = icm.transpose()[[dataset.get_track_index_from_id(x)
                               for x in self.tr_id_list]]
        cos = similarity.ExplicitCosine(shrinkage, k_filtering)
        time1 = time.time()
        S = cos(csr_matrix(icm))  # returns csr_matrix
        time2 = time.time()
        print('Explicit similarity took {:.3f} s'.format((time2 - time1)))
        # keep only target rows of URM and target columns
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]
        # get a column vector of the similarities of item i (i is the row)
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(np.reciprocal(s_norm))
        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose())
        print("R_hat done")
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        self.R_hat = R_hat

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
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = cbf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print(("MAP@5 [shrinkage: " + str(100) +
           " k_filtering: " + str(50) +
           " album_w: " + str(1.0) +
           " artist_w: " + str(1.0) + "]: "), map_at_five)
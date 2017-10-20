from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
# from scipy.sparse.linalg import svds
from sparsesvd import sparsesvd
import numpy as np
import numpy.linalg as LA


class CollaborativeSVD(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, album_w=1.0, artist_w=1.0, shrinkage=100, k_filtering=95, features=1000):
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
        print("target playlist ", len(self.pl_id_list))
        print("target tracks ", len(self.tr_id_list))
        # Apply SVD on URM and get the item features
        u, s, icm = sparsesvd(urm.tocsc(), features)
        s = np.diag(s)
        u = u[[dataset.get_playlist_index_from_id(x)
               for x in self.pl_id_list]]
        icm = icm[:, [dataset.get_track_index_from_id(x)
                      for x in self.tr_id_list]]
        print("SVD Done!")
        R_hat = u.dot(s).dot(icm)
        self.R_hat = R_hat
        print("R_hat done")
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]

        self.R_hat[urm_cleaned.nonzero()] = 0
        self.R_hat = csr_matrix(self.R_hat)
        # eliminate playlist that are not target, already done, to check
        #R_hat = R_hat[:, [dataset.get_track_index_from_id(
        #    x) for x in self.tr_id_list]]
        print("Shape of final matrix: ", R_hat.shape)

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


if __name__ == "__main__":
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cf = CollaborativeSVD()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cf.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = cf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print(("MAP@5 [shrinkage: " + str(100) +
           " k_filtering: " + str(50) +
           " album_w: " + str(1.0) +
           " artist_w: " + str(1.0) + "]: "), map_at_five)

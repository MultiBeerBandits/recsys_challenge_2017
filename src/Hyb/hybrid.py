from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from sklearn.metrics.pairwise import pairwise_distances
from src.utils.sim import computeSim
from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *

class SymInj(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=100, test_dict={}):
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
        # get ICM from dataset, assume it already cleaned
        icm = csr_matrix(dataset.build_icm())
        S_cb = computeSim(icm.transpose()[[dataset.get_track_index_from_id(x)
                          for x in self.tr_id_list]], icm, k_filtering=200)
        S_cf = computeSim(urm.transpose()[[dataset.get_track_index_from_id(x)
                          for x in self.tr_id_list]], urm, k_filtering=200)
        # Merge Them!
        S = S_cb
        S_lil = lil_matrix(S)
        for row_i in range(0, S.shape[0]):
            if row_i % 500 == 0:
                print("500 done")
            # get indices
            ind = S.indices[S.indptr[row_i]:S.indptr[row_i + 1]]
            # add the k most relevant items of S_cb 
            # get row of S_cb
            row_cf = S_cb.data[S_cf.indptr[row_i]:S_cf.indptr[row_i + 1]]
            sorted_idx = np.flip(row_cf.argsort(), axis=0)
            # clean from elements already present in S_cf
            filtered_idx = [S_cf.indices[S_cf.indptr[row_i] + x] for x in sorted_idx if S_cf.indices[S_cf.indptr[row_i] + x] not in ind][:k_filtering]
            cf_indices = S_cf.indices[S_cf.indptr[row_i]:S_cf.indptr[row_i+1]]
            for i in filtered_idx:
                # this is a mess
                S_lil[row_i, i] = row_cf[np.where(cf_indices == i)]
        S = csr_matrix(S_lil)
        R_hat = urm[[dataset.get_playlist_index_from_id(x)
                     for x in self.pl_id_list]].dot(S.transpose())
        urm_red = urm[[dataset.get_playlist_index_from_id(x)
                       for x in self.pl_id_list]]
        urm_red = urm_red[:, [dataset.get_track_index_from_id(x)
                              for x in self.tr_id_list]]
        R_hat[urm_red.nonzero()] = 0
        R_hat.eliminate_zeros()

        print("R_hat done")
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

    def getW(self):
        """
        Returns the similary matrix with dimensions I x I
        S is IxT
        """
        W = lil_matrix((self.S.shape[0], self.S.shape[0]))
        W[:, [self.dataset.get_track_index_from_id(
            x) for x in self.tr_id_list]] = self.S
        return W

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


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.5, 0.5, 0.5)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = SymInj()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        test_dict = ev.get_test_dict(i)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds, test_dict=test_dict)
        recs = cbf.predict()
        ev.evaluate_fold(recs)
        # ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 ", map_at_five)

    # export csv
    cbf_exporter = ContentBasedFiltering()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cbf_exporter.fit(urm,
                     tg_playlist,
                     tg_tracks,
                     ds)
    recs = cbf_exporter.predict()
    with open('submission_cbf.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})


if __name__ == '__main__':
    main()

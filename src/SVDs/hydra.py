from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from src.utils.feature_weighting import *
import src.utils.matrix_utils as utils
from sparsesvd import sparsesvd
from src.utils.evaluator import *


class ContentBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=200, features=100):
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
        print("CBF started")
        # Compute URM matrix factorization and predict missing ratings
        print('SVD on eURM')
        icm = dataset.build_icm()
        ucm = dataset.build_ucm().transpose() # ucm is playlist x attributes
        zeros_matrix = lil_matrix((icm.shape[0], ucm.shape[1])).tocsr()
        left_part = vstack([urm.multiply(0.5), icm], format='csr')
        # Stack all matrices:
        right_part = vstack([ucm, zeros_matrix], format='csr')
        eURM = hstack([left_part, right_part], format='csr')
        # DO SVD!
        u, s, v = sparsesvd(eURM.tocsc(), features)
        print("SHAPE of V: ", v.shape)
        print("SHAPE OF U", u.shape)
        # get only the features of the items
        v = csr_matrix(v[:, 0:icm.shape[1]])
        # get only the item part and compute cosine
        S = utils.compute_cosine(v.transpose()[[dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]],
                                 v,
                                 k_filtering=k_filtering,
                                 shrinkage=shrinkage)
        print("Similarity matrix ready, let's normalize it!")
        # zero out diagonal
        # in the diagonal there is the sim between i and i (1)
        # maybe it's better to have a lil matrix here
        # S.setdiag(0)
        # S.eliminate_zeros()
        # keep only target rows of URM and target columns
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        self.S = S.transpose()
        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose().tocsc()).tocsr()
        print("R_hat done")
        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        # eliminate playlist that are not target, already done, to check
        # R_hat = R_hat[:, [dataset.get_track_index_from_id(
        #    x) for x in self.tr_id_list]]
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

    def getW(self):
        """
        Returns the similary matrix with dimensions I x I
        S is IxT
        """
        W = lil_matrix((self.S.shape[0],self.S.shape[0]))
        W[:,[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]] = self.S
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
    # best_map = 0
    # best_album_w = 0
    # best_artist_w = 0
    # shr_w = 100
    # k_f = 50
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.3, 0.3, 0.3)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        test_dict = ev.get_test_dict(i)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = cbf.predict()
        ev.evaluate_fold(recs)
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

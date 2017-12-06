from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.matrix_utils import compute_cosine, top_k_filtering, dot_chunked, normalize_by_row


class UserItemFiltering():
    """
    Rationale:
    build a user feature matrix:
    UFM = URM * ICM'
    For each user we have how much is present a feature in its playlist
    Then:
    ufm_u = ufm_u / |Iu| where |Iu| is the number of tracks in this playlist
    [ |U| number of user, UF(f) number of user containing that feature]
    is the inverse user frequency of a feature
    The Inverse User Frequency of a feature is low,
    if it occurs in many users’ profiles
    whereas it is high, if the feature occurs in few users profiles
    Weight feature of UFM by the IUF:
    W (u, f ) = F F (u, f ) ∗ I U F (f )
    So it's the UFM multiplied by the row vector of IUF
    Build user similarity:
    S = UFM*UFM' normalized (and maybe shrinked)
    R_hat = S*URM, find best items in the row of a user
    Idea: two user are similar if the like items with similar features
    """

    """
    Result: MAP@5: 0.10272261177712429 with dot_chunked ufm 0.8 and ofm 0.2
    """

    def __init__(self):
        self.pl_id_list = []
        self.tr_id_list = []
        self.ufm = None
        self.icm = None
        self.urm = None
        self.dataset = None
        self.R_hat = None

    def fit(self, urm, target_playlist, target_tracks, dataset, k_feature=1000, k_similar=1000):
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.dataset = dataset
        self.urm = urm
        print("FWUM started!")

        # get ICM from dataset
        icm = dataset.build_icm_2()
        # icm = dataset.add_tracks_num_rating_to_icm(icm, urm).tocsr()

        # CONTENT BASED USER PROFILE
        # ucm_red = dataset.build_ucm()
        # ucm_red = dataset.add_playlist_num_rating_to_icm(ucm_red, urm)

        # build the user feature matrix
        # UxF
        ufm = urm.dot(icm.transpose())

        # Iu contains for each user the number of tracks rated
        Iu = urm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        # FxU
        ufm = csr_matrix(ufm.multiply(Iu).transpose())
        print("UCM ready")

        self.R_hat_fwum = compute_cosine(ufm.transpose()[[dataset.get_playlist_index_from_id(
            x) for x in target_playlist]], icm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]], k_filtering=500)

        # stack the urm
        # ufm = vstack([ufm, urm.transpose().multiply(2)], format='csr')

        # ufm = TfidfTransformer().fit_transform(ufm.transpose()).transpose()

        # # compute profile based prediction
        # # R_hat_1 = compute_cosine(ufm.transpose(),
        # #                          icm[:, [dataset.get_track_index_from_id(x)
        # #                                  for x in target_tracks]],
        # #                          k_filtering=500)

        # S_user = compute_cosine(ufm.transpose()[[dataset.get_playlist_index_from_id(
        #     x) for x in target_playlist]], ufm, k_filtering=500, shrinkage=100)
        # # normalize
        # S_user = normalize_by_row(S_user)
        # self.R_hat_ubf = S_user.dot(urm[:,
        #     [dataset.get_track_index_from_id(x) for x in target_tracks]])

        # # compute content based predictions
        # icm = dataset.add_playlist_to_icm(icm, urm, 0.4)
        # S_cbf = compute_cosine(icm.transpose()[
        #     [dataset.get_track_index_from_id(x) for x in target_tracks]],
        #     icm, k_filtering=200, shrinkage=50)

        # # normalize
        # norm = S_cbf.sum(axis=1)
        # # save from divide by zero!
        # norm[norm == 0] = 1
        # # since we have to divide the ufm get the reciprocal of this vector
        # norm = np.reciprocal(norm)
        # S_cbf = csr_matrix(S_cbf.multiply(norm))

        # # compute content based predictions
        # self.R_hat_cbf = urm[[dataset.get_playlist_index_from_id(
        #     x) for x in target_playlist]].dot(S_cbf.transpose())

        self.urm = self.urm[:, [
             dataset.get_track_index_from_id(x) for x in target_tracks]]
        self.urm = self.urm[[dataset.get_playlist_index_from_id(
             x) for x in target_playlist]]
        # self.R_hat_cbf[self.urm.nonzero()] = 0
        # self.R_hat_cbf.eliminate_zeros()
        # self.R_hat_cbf = top_k_filtering(self.R_hat_cbf, 10)

        # self.R_hat_ubf[self.urm.nonzero()] = 0
        # self.R_hat_ubf.eliminate_zeros()

        self.R_hat_fwum[self.urm.nonzero()] = 0
        self.R_hat_fwum.eliminate_zeros()

        # R_hat computation
        self.R_hat = self.R_hat_fwum

        print("R_hat done")

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

    def predict_cbf(self, at=5):
        return self._predict(self.R_hat_cbf)

    def predict_ubf(self, at=5):
        return self._predict(self.R_hat_ubf)

    def predict_mix1(self, at=5):
        return self._predict(self.R_hat)

    def predict_mix2(self, at=5):
        return self._predict(self.R_hat_cbf.multiply(self.R_hat_fwum))

    def predict_fwum(self, at=5):
        return self._predict(self.R_hat_fwum)

    def _predict(self, R_hat, at=5):
        recs = {}
        for i in range(0, R_hat.shape[0]):
            pl_id = self.pl_id_list[i]
            pl_row = R_hat.data[R_hat.indptr[i]:
                                     R_hat.indptr[i + 1]]
            # get top 5 indeces. argsort, flip and get first at-1 items
            sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
            track_cols = [R_hat.indices[R_hat.indptr[i] + x]
                          for x in sorted_row_idx]
            tracks_ids = [self.tr_id_list[x] for x in track_cols]
            recs[pl_id] = tracks_ids
        return recs

    def filter_by_topic(self, matrix, dataset):
        # filter ufm
        artist = dataset.build_artist_matrix(matrix)
        artist = top_k_filtering(artist, 50)

        album = dataset.build_album_matrix(matrix)
        album = top_k_filtering(album, 50)

        tags = dataset.build_tag_matrix(matrix)
        tags = top_k_filtering(tags, 50)

        duration = dataset.build_duration_matrix(matrix)
        duration = top_k_filtering(duration, 10)

        playcount = dataset.build_playcount_matrix(matrix)
        playcount = top_k_filtering(playcount, 10)

        return vstack([artist, album, tags, duration, playcount], format='csr')


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights_2(1, 1, 0.1, 0.1, 0.1, num_rating_weight=1, inferred_album=1, inferred_duration=0.1, inferred_playcount=0.1)
    # ds.set_playlist_attr_weights(0.1, 0.1, 0.1, 0.05, 0.05)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    xbf = UserItemFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        xbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = xbf.predict()
        ev.evaluate_fold(recs)
        # recs = xbf.predict_ubf()
        # ev.evaluate_fold(recs)
        # recs = xbf.predict_mix1()
        # ev.evaluate_fold(recs)
        # recs = xbf.predict_mix2()
        # ev.evaluate_fold(recs)
        # recs = xbf.predict_fwum()
        # ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)

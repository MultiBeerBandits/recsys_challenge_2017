from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.FWUM.UICF import xSquared
from src.CBF.CBF import ContentBasedFiltering
from src.UBF.UBF import UserBasedFiltering
from src.utils.cluster import build_user_cluster
from src.MF.iALS import IALS


class Ensemble(object):

    def __init__(self):
        self.dataset = None
        self.xbf = xSquared()
        self.cbf = ContentBasedFiltering()
        self.ubf = UserBasedFiltering()
        self.ials = None

    def mix(self, models, params, normalize_ratings=False):
        """
        takes an array of models all with R_hat as atttribute
        and mixes them using params
        params: array of attributes
        """
        R_hat_mixed = lil_matrix(
            (models[0].R_hat.shape[0], models[0].R_hat.shape[1]))
        for i in range(len(models)):
            if normalize_ratings:
                current_r_hat = self.max_normalize(models[i].R_hat)
            else:
                current_r_hat = models[i].R_hat
            R_hat_mixed += current_r_hat.multiply(params[i])
        return R_hat_mixed.tocsr()

    def mix_cluster(self, models, params, tg_playlist, urm=None, icm=None, ds=None):

        urm_red = urm[[ds.get_playlist_index_from_id(x) for x in tg_playlist]]
        ucm = ds.build_ucm()[:, [ds.get_playlist_index_from_id(x)
                                 for x in tg_playlist]]

        # user cluster contains only cluster of target users
        user_cluster = build_user_cluster(
            urm_red, icm, ucm, int(len(params)))  # / 3))

        R_hat_mixed = lil_matrix(
            (models[0].R_hat.shape[0], models[0].R_hat.shape[1]))
        for i in range(len(models)):
            # normalize weights to 0,1 if needed
            if normalize_ratings:
                current_r_hat = max_normalize(models[i].R_hat)
            else:
                current_r_hat = models[i].R_hat

            # build a column vector of weights for each user
            weights = [params[user_cluster(x) + i * len(user_cluster)]
                       for x in range(len(user_cluster))]
            # weights as column vector
            np_weights = np.reshape(np.array(weights), (-1, 1))

            # multiply weights by the current matrix
            R_hat_mixed += current_r_hat.multiply(np_weights)

        return R_hat_mixed.tocsr()

    def max_normalize(self, X):
        """
        Normalizes X by rows dividing each row by its max
        """
        max_r = X.max(axis=1)
        max_r.data = np.reciprocal(max_r.data)
        return X.multiply(max_r)

    def predict(self, params, at=5):
        # Mix them all
        models = [self.xbf, self.cbf, self.ubf, self.ials]
        R_hat = self.mix(models, params, normalize_ratings=True)
        """
        returns a dictionary of
        'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
        """
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

    def fit(self, urm, tg_tracks, tg_playlist, ds):
        """
        Fit all models
        """
        self.tr_id_list = tg_tracks
        self.pl_id_list = tg_playlist
        self.ials = IALS(urm.copy(), 500, 50, 1e-4, 800)
        self.xbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        self.cbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        self.ubf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        self.ials.fit(tg_playlist, tg_tracks, ds)

    def fit_cluster(self, params):
        ds = Dataset(load_tags=True, filter_tag=True)
        ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
        ds.set_playlist_attr_weights(1, 1, 1, 1, 1)
        ev = Evaluator()
        ev.cross_validation(5, ds.train_final.copy())
        xbf = xSquared()
        # cbf = ContentBasedFiltering()
        # ubf = UserBasedFiltering()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        xbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        # cbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        # ubf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
        recs_xbf = xbf.predict()
        # recs_cbf = cbf.predict()
        # recs_ubf = ubf.predict()
        ev.evaluate_fold(recs_xbf)
        # ev.evaluate_fold(recs_cbf)
        # ev.evaluate_fold(recs_ubf)
        # Mix them all
        models = [xbf]  # cbf, ubf]
        R_hat_mixed = mix(models, params, normalize_ratings=True)
        recs_mix = predict(R_hat_mixed, list(tg_playlist), list(tg_tracks))
        map_5 = ev.evaluate_fold(recs_mix)

        # multiply both
        R_hat_mult = max_normalize(xbf.R_hat).multiply(
            max_normalize(cbf.R_hat)).multiply(max_normalize(ubf.R_hat))
        recs_mult = predict(R_hat_mult, list(tg_playlist), list(tg_tracks))
        ev.evaluate_fold(recs_mult)

        print("MAP@5 :", map_5)
        return -map_5

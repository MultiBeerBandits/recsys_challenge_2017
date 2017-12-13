from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.utils.cluster import build_user_cluster
from src.Pop.popularity import Popularity
from src.FWUM.UICF import xSquared
from src.CBF.CBF_tfidf import ContentBasedFiltering
from src.IBF.IBF import ItemBasedFiltering
from src.UBF.UBF2 import UserBasedFiltering


class Ensemble(object):

    def __init__(self, models, normalize_ratings=False):
        """
        models is a list of recommender models implementing fit
        each model should implement BaseRecommender class
        """
        self.dataset = None
        self.models = models
        self.normalize_ratings = normalize_ratings

    def mix(self, params):
        """
        takes an array of models all with R_hat as atttribute
        and mixes them using params
        params: array of attributes
        """
        R_hat_mixed = lil_matrix(
            (self.models[0].R_hat.shape[0], self.models[0].R_hat.shape[1]))
        for i in range(len(self.models)):
            if self.normalize_ratings:
                current_r_hat = self.max_normalize(self.models[i].R_hat)
            else:
                current_r_hat = self.models[i].R_hat
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
        R_hat = self.mix(params)
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

    def predict_interleave(self, params, at=5):
        """
        params is a vector of length of models
        in the position of the model there's how many 
        items to get from that model
        """
        recs_all = [model.predict() for model in self.models]
        print("Length ", len(recs_all))
        recs = {}
        for i in range(0, len(self.pl_id_list)):
            pl_id = self.pl_id_list[i]
            tracks_ids = []
            model_index = 0
            while len(tracks_ids) < at:

                # get the current model
                model_index = model_index % len(self.models)

                # track of the current model
                track_ids = []

                if len(recs_all[model_index][pl_id]) > 0:
                    j = 0
                    # here we need to get the items for that playlist
                    while j < params[model_index] and len(recs_all[model_index][pl_id]) > 0:
                        tr_id = recs_all[model_index][pl_id][0]
                        if tr_id not in tracks_ids:
                            track_ids.append(tr_id)
                        recs_all[model_index][pl_id].remove(tr_id)
                        j += 1

                    # append the current tracks
                    tracks_ids = tracks_ids + track_ids

                model_index += 1

            # get only the first slice of the list
            tracks_ids = tracks_ids[:at]
            recs[pl_id] = tracks_ids

            if i % 1000 == 0:
                print("Done: ", (i / len(self.pl_id_list)) * 100)
        return recs

    def fit(self, urm, tg_tracks, tg_playlist, ds):
        """
        Fit all models
        """
        self.tr_id_list = tg_tracks
        self.pl_id_list = tg_playlist

        # call fit on all models
        for model in self.models:
            model.fit(urm.copy(), tg_playlist, tg_tracks, ds)

    def fit_cluster(self, params):

        print("MAP@5 :", map_5)
        return -map_5


def main():
    ds = Dataset(load_tags=True, filter_tag=False, weight_tag=False)
    ds.set_track_attr_weights_2(1.0, 1.0, 0.0, 0.0, 0.0,
                                1.0, 1.0, 0.0, 0.0)
    ds.set_playlist_attr_weights(1, 1, 1, 1, 1)
    ev = Evaluator(seed=False)
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)

    # create models
    cbf = ContentBasedFiltering()
    ubf = UserBasedFiltering()
    ibf = ItemBasedFiltering()

    # add models to list of models
    models = [cbf, ubf, ibf]

    # create the ensemble
    ensemble = Ensemble(models, normalize_ratings=True)

    # call fit on ensemble to fit all models
    ensemble.fit(urm, list(tg_tracks), list(tg_playlist), ds)

    # Mix them all
    recs_mix = ensemble.predict_interleave([2, 2, 1])
    map_5 = ev.evaluate_fold(recs_mix)

    recs_mix = ensemble.predict_interleave([3, 1, 1])
    map_5 = ev.evaluate_fold(recs_mix)

    recs_mix = ensemble.predict_interleave([1, 1, 1])
    map_5 = ev.evaluate_fold(recs_mix)

    recs_mix = ensemble.predict_interleave([2, 1, 2])
    map_5 = ev.evaluate_fold(recs_mix)

    recs_mix = ensemble.predict_interleave([3, 2, 1])
    map_5 = ev.evaluate_fold(recs_mix)


if __name__ == '__main__':
    print("Ensemble started")
    main()

import multiprocessing
from scipy.sparse import *
from sklearn.linear_model import ElasticNet, SGDRegressor
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from src.utils.loader import *
from src.utils.evaluator import *
import random


class BPRSLIM():
    """docstring for SLIM"""

    def __init__(self, l_pos=1e-5, l_neg=1e-6, l_rate=0.05):
        self.l_rate = l_rate
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.Theta = None
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None

    def fit(self, urm, target_items, target_users, dataset, n_iter):
        # Store target playlists and tracks
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)
        Theta = initTheta(urm.shape[1]).tocsc()
        draw = create_sampler(urm)
        for n in range(n_iter):
            # sample from Ds
            u, i, j = draw()
            # get user row
            x_u = urm.getrow(u)
            w_i = Theta.getcol(i)
            w_j = Theta.getcol(j)
            x_hat = x_u.multiply(w_i - w_j).sum()
            dsigm = exp(-x_hat) / (1 + exp(x_hat))

            ## TODO: CONTINUE WITH UPDATE OF THETA!


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


def initTheta(n_items):
    # generate a random matrix with a given density
    return rand(n_items, n_items, density=0.05)


def create_sampler(urm):
    def gen_sample():
        # generate a random user
        u = int(random.random() * urm.shape[0])
        # sample from the indices of urm[u]
        indices = urm.indices[urm.indptr[u]:urm.indptr[u + 1]]
        idx = int(random.random() * len(indices))
        i = indices[idx]
        j = int(random.random() * urm.shape[1])
        while j in indices:
            j = int(random.random() * urm.shape[1])
        return u, i, j
    return gen_sample


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        ubf = SLIM()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ubf.fit(urm, list(tg_tracks), list(tg_playlist), ds)
        recs = ubf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)
    # export
    cslim = SLIM()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cslim.fit(urm,
              tg_tracks,
              tg_playlist, ds)
    recs = cslim.predict()
    with open('submission_cslim.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})

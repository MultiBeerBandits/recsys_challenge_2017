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


class SLIM():
    """docstring for SLIM"""

    def __init__(self, l1_reg=7e-6, l2_reg=5e-5, feature_reg=4):
        """
        On 2017-10-22 we scored 0.081887527481395 with
            l1_reg=0.00001,
            l2_reg=0.000001,
            feature_reg=0.5
        ICM weights:
            art_w=1,
            alb_w=1,
            dur_w=0.2,
            playcount_w=0.2,
            tags_w=0.2
        """
        self.alpha = l1_reg + l2_reg
        self.l1_ratio = l1_reg / self.alpha
        # this is the weight of |F-FW|, importance of the icm
        self.feature_reg = feature_reg
        self.W = None
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None

    def fit(self, urm, target_items, target_users, dataset):
        # Store target playlists and tracks
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)

        # Set ICM weights
        dataset.set_track_attr_weights(art_w=1,
                                       alb_w=0.9,
                                       dur_w=0.2,
                                       playcount_w=0.2,
                                       tags_w=0.2)

        # get icm
        # icm = dataset.add_playlist_to_icm(
        #    dataset.build_icm(), 0.5) * np.sqrt(self.feature_reg)
        icm = dataset.build_icm() * np.sqrt(self.feature_reg)
        # Apply tf idf
        # transformer = TfidfTransformer()
        # icm = transformer.fit_transform(icm.transpose()).transpose()
        # icm = normalize(icm, axis=0)
        # Build training matrix
        M = vstack([urm, icm]).tocsc()

        model = ElasticNet(alpha=self.alpha,
                           fit_intercept=False,
                           l1_ratio=self.l1_ratio,
                           positive=True)

        # LET's PARALLEL!!!
        # First we get a sorted list of target items column indices.
        #   We want them sorted to improve data locality.
        target_indeces = sorted([dataset.get_track_index_from_id(x)
                                 for x in target_items])

        # Then we split the target items indices into chunks to ship
        # to pool workers.
        chunks = []
        chunksize = len(target_indeces) // multiprocessing.cpu_count()
        for i in range(0, len(target_indeces), chunksize):
            chunks.append(target_indeces[i:i + chunksize])

        # Build a list of parameters to ship to pool workers, containing:
        #   - A chunk of the target items indices
        #   - The URM
        #   - The ElasticNet model
        separated_tasks = []
        for ti in chunks:
            separated_tasks.append([ti, M, model])

        start = time.time()

        pool = multiprocessing.Pool()
        result = pool.map(_work, separated_tasks)

        # Merge results from workers and close the pool
        self.W = None
        for chunk in result:
            if self.W is None:
                self.W = chunk
            else:
                self.W = lil_matrix(self.W)
                self.W = self.W + chunk

        pool.close()
        pool.join()

        end = time.time()

        print("Time elapsed: ", (end - start) / 60)

        # clean urm from unwanted users
        urm = urm[[dataset.get_playlist_index_from_id(x)
                   for x in self.pl_id_list]]

        self.W = csr_matrix(self.W)
        print(urm.shape, self.W.shape)
        # Compute prediction matrix (n_target_users X n_items)
        self.R_hat = urm.dot(self.W).tocsr()
        print('R_hat evaluated...')

        # Clean R_hat from already rated entries
        self.R_hat[urm.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        # Keep only target_item columns
        self.R_hat = self.R_hat[:, [dataset.get_track_index_from_id(x)
                                    for x in target_items]]

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


def _work(params):
    # get params
    target_indeces = params[0]
    M = csc_matrix(params[1])
    model = params[2]
    count = 0
    pid = os.getpid()

    W = lil_matrix((M.shape[1], M.shape[1]))
    for t in target_indeces:
        if count % 100 == 0:
            print('[', pid, ']', count, '/',
                  len(target_indeces), 'ElasticNet trained...')
        # Zero-out the t-th column to meet the w_tt = 0 constraint
        r_t = M.getcol(t).toarray().ravel()
        M.data[M.indptr[t]:M.indptr[t + 1]] = 0
        # Fit
        model.fit(M, r_t)
        # restore matrix
        M[:, t] = csc_matrix(r_t).transpose()

        # Build a W matrix with column indeces from 0 to to_col - from_col
        W[:, t] = model.sparse_coef_.transpose()
        count += 1
    return W


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
        ev.print_worst(ds)
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

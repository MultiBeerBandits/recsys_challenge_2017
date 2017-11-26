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
from src.utils.BaseRecommender import BaseRecommender


class RCSLIM(BaseRecommender):
    """docstring for SLIM"""

    def __init__(self, l1_reg=1e-3, l2_reg=1e-4, feature_reg=4, beta1=2):
        self.alpha = l1_reg + l2_reg
        self.l1_ratio = l1_reg / self.alpha
        # this is the weight of |F-FW|, importance of the icm
        self.feature_reg = feature_reg
        self.beta1 = beta1
        self.W = None
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None
        self.dataset = None

    def fit(self, urm, tg_playlist, tg_tracks, dataset):
        # Store target playlists and tracks
        self.pl_id_list = list(tg_playlist)
        self.tr_id_list = list(tg_tracks)

        # store dataset
        self.dataset = dataset

        # Use the iucm
        icm = dataset.build_icm()
        # weight icm
        self.icm = icm * np.sqrt(self.feature_reg)

        self.urm = urm

        # side effect: save S
        self.solve_cslim()

        # Initialization of Q
        self.Q = lil_matrix((self.S.shape[0], self.S.shape[1])).tocsr()

        # start alternating solving
        # model for solving with S
        model = SGDRegressor(penalty='elasticnet',
                             fit_intercept=False,
                             alpha=self.alpha,
                             l1_ratio=self.l1_ratio)

        for i in range(50):
            id = eye(self.S.shape[0])
            self.Q = self.solve_alternating(self.icm, self.S, self.icm, id, self.Q, model, self.feature_reg, self.beta1)
            self.S = self.solve_alternating(self.urm, self.Q, self.urm, id, self.S, model, 1, self.beta1)

        # clean urm from unwanted users
        urm = urm[[dataset.get_playlist_index_from_id(x)
                   for x in self.pl_id_list]]

        self.Q = csr_matrix(self.Q)

        # Compute prediction matrix (n_target_users X n_items)
        self.R_hat = urm.dot(self.Q).tocsr()
        print('R_hat evaluated...')

        # Clean R_hat from already rated entries
        self.R_hat[urm.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        # Keep only target_item columns
        self.R_hat = self.R_hat[:, [dataset.get_track_index_from_id(x)
                                    for x in self.tr_id_list]]

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

    def solve_cslim(self):
        # Build training matrix
        M = vstack([self.urm, self.icm], format='csc')

        model = ElasticNet(alpha=self.alpha,
                           fit_intercept=False,
                           l1_ratio=self.l1_ratio,
                           positive=True)

        # LET's PARALLEL!!!
        # First we get a sorted list of target items column indices.
        #   We want them sorted to improve data locality.
        target_indeces = sorted([self.dataset.get_track_index_from_id(x)
                                 for x in self.tr_id_list])

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
        self.S = None
        for chunk in result:
            if self.S is None:
                self.S = chunk
            else:
                self.S = lil_matrix(self.S)
                self.S = self.S + chunk

        pool.close()
        pool.join()

        end = time.time()

        print("Time elapsed: ", (end - start) / 60)

    def solve_alternating(self, y1, y2, x1, x2, init_matrix, model, alfa1, alfa2):
        """
        wrapper around work for multithreading
        stack matrices and start multithreading
        Y = [alfa1*y1; alfa2*y2]
        X = [alfa1*x1, alfa2*x2]
        """
        # Build training matrix
        Y = vstack([y1.multiply(alfa1), y2.multiply(alfa2)], format='csc')
        X = vstack([x1.multiply(alfa1), x2.multiply(alfa2)], format='csc')

        # LET's PARALLEL!!!
        # First we get a sorted list of target items column indices.
        #   We want them sorted to improve data locality.
        target_indeces = sorted([self.dataset.get_track_index_from_id(x)
                                 for x in self.tr_id_list])

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
            separated_tasks.append([ti, Y, X, W, model])

        start = time.time()

        pool = multiprocessing.Pool()
        result = pool.map(_work_alt, separated_tasks)

        # Merge results from workers and close the pool
        S = None
        for chunk in result:
            if S is None:
                S = chunk
            else:
                S = lil_matrix(S)
                S = S + chunk

        pool.close()
        pool.join()

        end = time.time()

        print("Time elapsed: ", (end - start) / 60)
        return S

    def getW(self):
        """
        Returns the similary matrix with dimensions I x I
        """
        return self.W

    def getR_hat(self):
        return self.R_hat


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


def _work_alt(params):
    # get params
    target_indeces = params[0]
    Y = csc_matrix(params[1])
    X = csc_matrix(params[2])
    W = csc_matrix(params[3])
    model = params[4]
    count = 0
    pid = os.getpid()

    W = lil_matrix((M.shape[1], M.shape[1]))
    for t in target_indeces:
        if count % 100 == 0:
            print('[', pid, ']', count, '/',
                  len(target_indeces), 'ElasticNet trained...')
        # Zero-out the t-th column to meet the w_tt = 0 constraint
        r_t = Y.getcol(t).toarray().ravel()
        X.data[M.indptr[t]:M.indptr[t + 1]] = 0
        # Initial coeffs of the regression
        coef_init = W.getcol(t).toarray().ravel()
        # Fit
        model.fit(X, r_t, coef_init=coef_init)
        # restore matrix
        X[:, t] = csc_matrix(r_t).transpose()

        # Build a W matrix with column indeces from 0 to to_col - from_col
        W[:, t] = model.sparse_coef_.transpose()
        count += 1
    return W


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        rcslim = RCSLIM()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        rcslim.fit(urm, list(tg_playlist), list(tg_tracks), ds)
        recs = rcslim.predict()
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

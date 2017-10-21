import multiprocessing
import ctypes
from scipy.sparse import *
from sklearn.linear_model import ElasticNet, SGDRegressor
import numpy as np
import os

from src.utils.loader import *
from src.utils.evaluator import *

# PARALLEL STUFF
# Loading shared array to be used in results
shared_array_base = 0
shared_array = 0


class SLIM():
    """docstring for SLIM"""

    def __init__(self, l1_reg=0.000001, l2_reg=0.0000001, feature_reg=0.5):
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

        # get icm
        icm = dataset.build_icm() * np.sqrt(feature_reg)
        M = vstack([urm, icm]).tocsc()

        # LET's PARALLEL!!!
        total_columns = M.shape[1]
        ranges = _generate_slices(total_columns)
        separated_tasks = []

        for from_j, to_j in ranges:
            separated_tasks.append([from_j, to_j, Mline, model])

        pool = multiprocessing.Pool()
        pool.map(_work, separated_tasks)
        pool.close()
        pool.join()

        # clean urm from unwanted users
        urm = urm[[dataset.get_playlist_index_from_id(x)
                   for x in self.pl_id_list]]

        self.W = csr_matrix(shared_array)
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


def _generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    from multiprocessing import cpu_count

    cores = cpu_count()

    segment_length = total_columns / cores

    ranges = []
    now = 0

    while now < total_columns:
        end = now + segment_length

        # The last part can be a little greater that others in some cases
        # but we can't generate more than #cores ranges
        end = end if end + segment_length <= total_columns else total_columns
        ranges.append((now, end))
        now = end

    return ranges


def _work(params, W=shared_array):
    # get params
    from_col = params[0]
    to_col = params[1]
    model = params[2]
    M = params[3]
    count = 0
    pid = os.getpid()
    for t in range(from_col, to_col, 1):
        if count % 100 == 0:
            print('[', pid, ']', count / float(to_col - from_col) * 100,
                  'ElasticNet trained...')
        # Zero-out the t-th column to meet the w_tt = 0 constraint
        mlinej = M[:, t].copy()
        M[:, t] = 0
        M.eliminate_zeros()
        r_t = M.getcol(t).toarray().ravel()

        # Fit
        model.fit(M_z, r_t)
        # print(model.coef_.shape)
        W[:, t] = model.coef_.transpose()
        count += 1


if __name__ == '__main__':
    ds = Dataset()
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    ubf = SLIM()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        shared_array_base = multiprocessing.Array(ctypes.c_double, urm.shape[1]**2)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(urm.shape[1], urm.shape[1])
        ubf.fit(urm, list(tg_tracks), list(tg_playlist), ds)
        recs = ubf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5", map_at_five)

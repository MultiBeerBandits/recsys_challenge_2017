from src.utils.loader import *
from src.utils.evaluator import Evaluator
from scipy.sparse import *
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.utils.BaseRecommender import BaseRecommender
from fastFM.sgd import FMRegression



class FM(BaseRecommender):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset):
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

        print("FM started")
        urm = urm.tocsr()

        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm_2().tocsc()

        n_users = urm.shape[0]
        n_items = urm.shape[1]

        # users + artist + tracks attributes
        n_cols = urm.shape[0] + urm.shape[1] + icm.shape[0]
        n_rows = 2 * urm.nnz

        M = lil_matrix((n_rows, n_cols))

        # better to have coo matrix for building model for FM
        urm_coo = urm.tocoo()

        for i in range(urm.data.shape[0]):

            # add artist and tracks
            user_index = urm_coo.row[i]
            track_index = urm_coo.col[i]

            M[i, user_index] = 1
            M[i, n_users + track_index] = 1

            # add attributes
            row_icm = icm.data[icm.indptr[track_index]:icm.indptr[track_index + 1]]
            indices = icm.indices[icm.indptr[track_index]: icm.indptr[track_index + 1]]

            assert(row_icm.shape[0] == indices.shape[0])

            for j in range(row_icm.shape[0]):
                M[i, indices[j]] = row_icm[j]

            if i % 5000 == 0:
                print("Done", (i / urm.data.shape[0]) * 100)

        # add some zero rating
        # get i and j, check if they are not in nnz of urm
        user_shuffled = np.random.choice(n_users, urm.nnz, replace=True)
        index = 0
        for i in range(urm.nnz, 2 * urm.nnz):

            user_index = user_shuffled[index]

            selected = False

            while not selected:
                track_index = np.random.choice(n_items)

                if track_index not in urm.indices[urm.indptr[user_index]: urm.indptr[user_index+1]]:
                    selected = True

            M[i, user_index] = 1
            M[i, n_users + track_index] = 1

            # add attributes
            row_icm = icm.data[icm.indptr[track_index]:icm.indptr[track_index + 1]]
            indices = icm.indices[icm.indptr[track_index]: icm.indptr[track_index + 1]]

            assert(row_icm.shape[0] == indices.shape[0])

            for j in range(row_icm.shape[0]):
                M[i, indices[j]] = row_icm[j]

            if i % 5000 == 0:
                print("Done", ((i - urm.nnz) / urm.nnz) * 100)

        M = M.tocsc()

        y = np.ones(M.shape[0])
        y[urm.nnz:y.shape[0]] = 0

        # build x test
        n_tg_user = len(pl_id_list)
        n_tg_items = len(tr_id_list)

        n_rows_test = n_tg_user * n_tg_items

        X_test = lil_matrix((n_rows_test, n_cols))

        tg_pl_index = [dataset.get_playlist_index_from_id(x)
                       for x in self.pl_id_list]
        tg_tr_index = [dataset.get_playlist_index_from_id(x)
                       for x in self.tr_id_list]

        # index for rows
        index = 0
        for user_index, track_index in zip(tg_pl_index, tg_tr_index):
            X_test[index, user_index] = 1
            X_test[index, n_users + track_index] = 1

            # add attributes
            row_icm = icm.data[icm.indptr[track_index]:icm.indptr[track_index + 1]]
            indices = icm.indices[icm.indptr[track_index]: icm.indptr[track_index + 1]]

            assert(row_icm.shape[0] == indices.shape[0])

            for j in range(row_icm.shape[0]):
                X_test[index, indices[j]] = row_icm[j]

            index += 1

        X_test = X_test.tocsc()

        print("Start learning")
        # start learning
        fmreg = FMRegression()
        fmreg.fit(M, y)

        print("Learning finished")
        # prediction
        y_test = fmreg.predict(X_test)

        print("prediction finished")
        # build r_hat from y_test
        R_hat = lil_matrix((n_tg_user, n_tg_items))

        # build R_hat
        index = 0
        for user_index in range(n_tg_user):
            user_row = y_test[index * n_tg_items: (index + 1) * n_tg_items]
            user_row_filtered = top_k_filtering(user_row, topK=5)
            R_hat[user_index] = user_row_filtered

            index += 1

        print("R_hat ready")

        self.R_hat = R_hat.tocsr()


    def getW(self):
        """
        Returns the similary matrix with dimensions I x I
        S is IxT
        """
        return self.S.tocsr()

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

    def getR_hat(self):
        return self.R_hat


    def get_model(self):
        """
        Returns the complete R_hat
        """
        return self.R_hat.copy()


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights_2(1, 1, 1, 1, 1, num_rating_weight=1, inferred_album=1, inferred_duration=1, inferred_playcount=1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        fm = FM()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        fm.fit(urm, list(tg_playlist), list(tg_tracks), ds)
        recs = fm.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

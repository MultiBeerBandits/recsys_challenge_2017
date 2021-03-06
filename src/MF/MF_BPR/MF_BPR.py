import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering, dot_chunked
import subprocess
import os, sys
from sklearn.linear_model import SGDRegressor

# best params: rmsprop, user_reg 1e-1 item_reg 1e-2 l_rate 5e-2 (or 1e-2) epochMult = 5, n_components = 500
#  no_components=100, n_epochs=2, user_reg=1e-2, item_reg=1e-3, l_rate=1e-2, epoch_multiplier=5 0.045 after 20 epochs
class MF_BPR():

    def __init__(self, U=None, V=None):
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None
        self.cythonEpoch = None
        self.icm_t = None
        self.current_epoch = 0
        self.reset_cache = False
        self.eligibleUsers = None
        self.urm = None
        self.icm = None
        self.target_users = None
        self.U = U
        self.V = V
        self.S = None

        if _requireCompilation():
            self.runCompilationScript()
        pass

    def fit(self, urm, dataset, tg_playlist, tg_tracks, no_components=500, n_epochs=2, user_reg=1e-1, pos_item_reg=1e-2, neg_item_reg=1e-3, l_rate=1e-1, epoch_multiplier=2, use_icm=False):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        self.dataset = dataset

        if self.urm is None:
            urm = urm.tocsr()
            self.urm = urm
            self.icm = dataset.build_icm_2()

        if use_icm:
            urm_ext = vstack([urm, self.icm], format='csr')
            num_features = self.icm.shape[0]
            num_user = self.urm.shape[0]
        else:
            urm_ext = urm
            num_features = None
            num_user = None

        if self.eligibleUsers is None:
            self.eligibleUsers = []
            for user_id in range(urm_ext.shape[0]):

                start_pos = urm_ext.indptr[user_id]
                end_pos = urm_ext.indptr[user_id + 1]

                numUserInteractions = len(urm_ext.indices[start_pos:end_pos])

                if numUserInteractions > 0:
                    self.eligibleUsers.append(user_id)

            # self.eligibleUsers contains the userID having at least one positive interaction and one item non observed
            self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)

        if self.target_users is None:
            self.target_users = np.array([dataset.get_playlist_index_from_id(x) for x in tg_playlist], dtype=np.int64)
            self.target_items = np.array([dataset.get_track_index_from_id(x) for x in tg_tracks], dtype=np.int64)

        if self.cythonEpoch is None:
            from src.MF.MF_BPR.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch
            self.cythonEpoch = MF_BPR_Cython_Epoch(urm_ext,
                                                   self.eligibleUsers,
                                                   target_items=self.target_items,
                                                   target_users=self.target_users,
                                                   num_factors=no_components,
                                                   learning_rate=l_rate,
                                                   batch_size=1,
                                                   sgd_mode='adam',
                                                   user_reg=user_reg,
                                                   positive_reg=pos_item_reg,
                                                   negative_reg=neg_item_reg,
                                                   epoch_multiplier=epoch_multiplier,
                                                   opt_mode='bpr',
                                                   W=self.U,
                                                   H=self.V,
                                                   num_features=num_features,
                                                   num_user_sample=num_user)


        # start learning
        for i in range(n_epochs):
            # if self.current_epoch >= 6 and self.reset_cache:
            #     self.reset_cache = False
            self.cythonEpoch.epochIteration_Cython(l_rate, reset_cache=False)
            self.current_epoch += 1

        print("Training finished")

        # get W and H to make predictions
        # UxF
        self.W = self.cythonEpoch.get_W()
        # IxF
        self.H = self.cythonEpoch.get_H()

        # build the similarity matrix
        S = csr_matrix(self.m_dot_chunked(self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]], self.H.transpose(), topK=100))
        print("S done")

        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        self.S = S.transpose()

        print("R_hat done")


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

    def predict_dot(self, at=5):
        W = self.W[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        H = self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        R_hat = np.dot(W, H.transpose())

        urm_cleaned = self.urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat = csr_matrix(R_hat)
        R_hat = csr_matrix(top_k_filtering(R_hat, 10))
        return self._predict(R_hat)

    def predict_dot_custom(self, urm, at=5):
        W = self.W[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        H = self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        R_hat = np.dot(W, H.transpose())

        urm_cleaned = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat = csr_matrix(R_hat)
        R_hat = csr_matrix(top_k_filtering(R_hat, 10))
        return self._predict(R_hat)

    def getR_hat(self, urm):
        W = self.W[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        H = self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        R_hat = np.dot(W, H.transpose())

        urm_cleaned = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat = csr_matrix(R_hat)
        R_hat = csr_matrix(top_k_filtering(R_hat, 50))
        return R_hat

    def predict_knn(self, at=5):
        S = csr_matrix(self.m_dot_chunked(self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]], self.H.transpose(), topK=100))
        print("S done")
        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        self.S = S.transpose()
        R_hat = self.urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S.transpose())
        urm_cleaned = self.urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        R_hat = csr_matrix(top_k_filtering(R_hat, 10))
        return self._predict(R_hat)

    def getR_hat_knn(self, urm, at=5):
        S = csr_matrix(self.m_dot_chunked(self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]], self.H.transpose(), topK=100))
        print("S done")
        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        R_hat = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S.transpose())
        urm_cleaned = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        R_hat = csr_matrix(top_k_filtering(R_hat, 50))
        return R_hat

    def predict_knn_custom(self, urm, at=5):
        S = csr_matrix(self.m_dot_chunked(self.H[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]], self.H.transpose(), topK=100))
        print("S done")

        # Normalize S
        s_norm = S.sum(axis=1)
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        self.S = S.transpose()
        R_hat = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S.transpose())
        urm_cleaned = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        R_hat = csr_matrix(top_k_filtering(R_hat, 10))
        return self._predict(R_hat)

    def m_dot_chunked(self, X, Y, topK, chunksize=1000):
        result = None
        start = 0
        mat_len = X.shape[0]
        for chunk in range(start, mat_len, chunksize):
            if chunk + chunksize > mat_len:
                end = mat_len
            else:
                end = chunk + chunksize
            print('Computing dot product for chunk [{:d}, {:d})...'
                  .format(chunk, end))
            X_chunk = X[chunk:end]
            sub_matrix = np.dot(X_chunk, Y)
            sub_matrix = csr_matrix(top_k_filtering(sub_matrix, topK))
            if result is None:
                result = sub_matrix
            else:
                result = vstack([result, sub_matrix], format='csr')
        return result

    def runCompilationScript(self):

        # Run compile script setting the working directory to
        # ensure the compiled
        # file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/src/MF/MF_BPR"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python3',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        #python compileCython.py MF_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "MF_BPR_Cython_Epoch.pyx"])


def _requireCompilation():
    import os
    moduleSubFolder = 'src/MF/MF_BPR/'
    pyx_source = moduleSubFolder + 'MF_BPR_Cython_Epoch.pyx'
    c_source = moduleSubFolder + 'MF_BPR_Cython_Epoch.c'
    if os.path.isfile(c_source):
        if os.path.getmtime(pyx_source) < os.path.getmtime(c_source):
            return False
    return True


if __name__ == '__main__':
    from src.CBF.CBF_MF import ContentBasedFiltering as CBF

    ds = Dataset(load_tags=True, filter_tag=True)
    # ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ds.set_track_attr_weights_2(1.5, 1.6, 0, 0, 0.0,
                                 1.0, 0, 0.0, 0.0)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf = CBF()
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        # get R_hat
        R_hat_aug = cbf.getR_hat()
        print(R_hat_aug.nnz)

        mf = MF_BPR()
        for epoch in range(50):
            mf.fit(R_hat_aug, ds, list(tg_playlist), list(tg_tracks), n_epochs=1, no_components=500, epoch_multiplier=1, l_rate=1e-2, use_icm=False)
            recs = mf.predict_dot()
            ev.evaluate_fold(recs)
            #recs = mf.predict_knn()
            #ev.evaluate_fold(recs)
        # ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

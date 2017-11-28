import scipy.sparse as sps
import numpy as np
import time
import sys

from src.utils.matrix_utils import top_k_filtering


class BPRSLIM():
    """docstring for BPRSLIM"""

    def __init__(self,
                 epochs=500,
                 epochMultiplier=1.0,
                 sgd_mode='sgd',
                 learning_rate=5e-08,
                 topK=300,
                 urmSamplingChances=0.5,
                 icmSamplingChances=0.5):
        self.epochs = epochs
        self.epochMultiplier = epochMultiplier
        self.sgd_mode = sgd_mode
        self.learning_rate = learning_rate
        self.topK = topK
        self.urmSamplingChances = urmSamplingChances
        self.icmSamplingChances = icmSamplingChances
        self.W = None
        self.evaluator = None
        self.evaluate_every_n_epochs = None

        if _requireCompilation():
            print('Compiling in Cython...')
            _runCompileScript()
            print('Compilation complete!')

    def fit(self, urm, icm, target_users, target_items, dataset):
        self._store_fit_params(urm, icm, target_users, target_items, dataset)
        if urm is not None:
            self._computeEligibleUsers(urm)
        if icm is not None:
            self._computeEligibleFeatures(icm)

        # Import compiled module
        from src.ML.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(urm,
                                                 icm,
                                                 self.eligibleUsers,
                                                 self.eligibleFeatures,
                                                 self.learning_rate,
                                                 self.urmSamplingChances,
                                                 self.icmSamplingChances,
                                                 self.epochMultiplier,
                                                 self.topK,
                                                 self.sgd_mode)
        self._BPROpt()

    def set_evaluation_every(self, epochs, evaluator):
        self.evaluate_every_n_epochs = epochs
        self.evaluator = evaluator

    def _store_fit_params(self, urm, icm, target_users, target_items, dataset):
        # Store input parameters
        self.urm = urm
        self.icm = icm
        self.dataset = dataset
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)
        self.n_users = urm.shape[0]
        self.n_features = icm.shape[0]
        self.n_items = urm.shape[1]

    def _computeEligibleUsers(self, urm):
        self.eligibleUsers = []

        for user_id in range(self.n_users):
            start_pos = urm.indptr[user_id]
            end_pos = urm.indptr[user_id + 1]

            if len(urm.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)

    def _computeEligibleFeatures(self, icm):
        self.eligibleFeatures = []

        for user_id in range(self.n_features):
            start_pos = icm.indptr[user_id]
            end_pos = icm.indptr[user_id + 1]

            if len(icm.indices[start_pos:end_pos]) > 0:
                self.eligibleFeatures.append(user_id)

        self.eligibleFeatures = np.array(self.eligibleFeatures, dtype=np.int64)

    def _BPROpt(self):
        start_time_train = time.time()
        for currentEpoch in range(self.epochs):
            start_time_epoch = time.time()
            # Run BPR-Opt iteration
            self._epochIteration()
            # If n_epochs have passed, evaluate model performace
            if currentEpoch > 0 and \
                    (self.evaluate_every_n_epochs is not None) and \
                    (currentEpoch % self.evaluate_every_n_epochs == 0):
                self._evaluateRecommendations()

            print("Epoch {} of {} complete in {:.2f} minutes".format(
                currentEpoch, self.epochs,
                float(time.time() - start_time_epoch) / 60))

        print("Fit completed in {:.2f} minutes".format(
            float(time.time() - start_time_train) / 60))
        sys.stdout.flush()

    def _epochIteration(self):
        S = self.cythonEpoch.epochIteration_Cython()
        self.W = S.transpose()

    def _updateSimilarityMatrix(self):
        if self.topK is not False:
            self.W = top_k_filtering(self.W, topK=self.topK)

    def _evaluateRecommendations(self, at=5):
        print("Evaluating recommendations")
        # Compute prediction matrix (n_target_users X n_items)
        urm_red = self.urm[[self.dataset.get_playlist_index_from_id(x)
                            for x in self.pl_id_list]]
        w_red = self.W[:, [self.dataset.get_track_index_from_id(x)
                           for x in self.tr_id_list]]
        self.R_hat = urm_red.dot(w_red).tocsr()
        print('R_hat evaluated...')
        urm_red = urm_red[:, [self.dataset.get_track_index_from_id(x)
                              for x in self.tr_id_list]]
        # Clean R_hat from already rated entries
        self.R_hat[urm_red.nonzero()] = 0
        self.R_hat.eliminate_zeros()

        # returns a dictionary of
        # 'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
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
        return self.evaluator.evaluate_fold(recs)


def _requireCompilation():
    import os
    moduleSubFolder = 'src/ML/Cython/'
    pyx_source = moduleSubFolder + 'SLIM_BPR_Cython_Epoch.pyx'
    c_source = moduleSubFolder + 'SLIM_BPR_Cython_Epoch.c'
    if os.path.isfile(c_source):
        if os.path.getmtime(pyx_source) < os.path.getmtime(c_source):
            return False
    return True


def _runCompileScript():
    import subprocess
    compiledModuleSubfolder = "/src/ML/Cython"
    fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

    for fileToCompile in fileToCompile_list:
        command = ['python',
                   'compileCython.py',
                   fileToCompile,
                   'build_ext',
                   '--inplace']

        subprocess.check_output(
            ' '.join(command), shell=True,
            cwd=os.getcwd() + compiledModuleSubfolder)

        try:
            command = ['cython',
                       fileToCompile,
                       '-a']
            subprocess.check_output(
                ' '.join(command), shell=True,
                cwd=os.getcwd() + compiledModuleSubfolder)
        except Exception:
            pass

        print("Compiled module saved in subfolder: {}".format(
              compiledModuleSubfolder))


if __name__ == '__main__':
    from src.utils.loader import *
    from src.utils.evaluator import *

    import os

    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())

    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    test_urm = ev.get_test_matrix(0, ds)
    icm = ds.build_icm()

    recommender = BPRSLIM(epochs=500,
                          epochMultiplier=0.5,
                          sgd_mode='adagrad',
                          learning_rate=5e-08,
                          topK=300,
                          urmSamplingChances=1 / 5,
                          icmSamplingChances=4 / 5)
    recommender.set_evaluation_every(1, ev)
    recommender.fit(urm.tocsr(),
                    icm.tocsr(),
                    tg_playlist,
                    tg_tracks,
                    ds)

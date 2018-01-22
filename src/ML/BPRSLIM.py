import scipy.sparse as sps
import numpy as np
import time
import sys
import os

from src.utils.matrix_utils import top_k_filtering, writeSubmission


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
        print('Running fit process.')
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

    def predict(self, at=5):
        self._computeR_hat()
        return self._computeRecommendations(at=at)

    def getParameters(self):
        return self.W.copy()

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
        self._computeR_hat()
        recs = self._computeRecommendations()
        return self.evaluator.evaluate_fold(recs)

    def _computeR_hat(self):
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

    def _computeRecommendations(self, at=5):
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
        return recs


class ParallelizedBPRSLIM():
    """docstring for Parallelized"""
    def __init__(self,
                 epochs=500,
                 epochMultiplier=1.0,
                 sgd_mode='sgd',
                 learning_rate=5e-08,
                 topK=300,
                 urmSamplingChances=0.5,
                 icmSamplingChances=0.5):
        import multiprocessing as mp

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

        self.n_tasks = mp.cpu_count()
        self.max_threads = mp.cpu_count()

    def fit(self, urm, icm, target_users, target_items, dataset):
        import multiprocessing as mp
        self.urm = urm
        self.icm = icm
        self.pl_id_list = target_users
        self.tr_id_list = target_items
        self.dataset = dataset

        for i in range(0, self.n_tasks, self.max_threads):
            print('[{:d} / {:d}] Running parallel fit process'
                  'on {:d} threads...'.format(
                      i, int(self.n_tasks // self.max_threads) + 1,
                      self.max_threads))
            jobs = []
            for _ in range(self.max_threads):
                jobs.append([BPRSLIM(self.epochs,
                                     self.epochMultiplier,
                                     self.sgd_mode,
                                     self.learning_rate,
                                     self.topK,
                                     self.urmSamplingChances,
                                     self.icmSamplingChances),
                             urm,
                             icm,
                             target_users,
                             target_items,
                             dataset])

            with mp.Pool(processes=self.max_threads) as pool:
                weights = pool.map(_single_run, jobs)

                print('Summing results...')
                # Average result weights
                for w in weights:
                    if self.W is None:
                        self.W = w
                    else:
                        self.W += w

                print('Results summed.')
        print('Normalizing results...')
        self.W.data /= self.n_tasks

    def set_n_tasks(self, n_tasks):
        self.n_tasks = n_tasks

    def set_max_threads(self, max_threads):
        self.max_threads = max_threads

    def predict(self, at=5):
        self._computeR_hat()
        return self._computeRecommendations(at=at)

    def getParameters(self):
        return self.W.copy()

    def _computeR_hat(self):
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

    def _computeRecommendations(self, at=5):
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
        return recs


def _single_run(params):
    # Unpack parameters
    model = params[0]
    urm = params[1]
    icm = params[2]
    target_users = params[3]
    target_items = params[4]
    dataset = params[5]

    model.fit(urm, icm, target_users, target_items, dataset)
    return model.getParameters()


def _requireCompilation():
    moduleSubFolder = 'src/ML/Cython/'
    pyx_source = moduleSubFolder + 'SLIM_BPR_Cython_Epoch.pyx'
    c_source = moduleSubFolder + 'SLIM_BPR_Cython_Epoch.c'
    if os.path.isfile(c_source):
        if os.path.getmtime(pyx_source) < os.path.getmtime(c_source):
            return False
    return True


def _runCompileScript():
    import subprocess
    import os
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


def _testAverage(n_training):
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 1, 1, 1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())

    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    # test_urm = ev.get_test_matrix(0, ds)
    icm = ds.build_icm()
    print('Preprocessing the ICM...')
    icm = _prepareICM(icm, ds)

    recommender = ParallelizedBPRSLIM(epochs=300,
                                      epochMultiplier=0.5,
                                      sgd_mode='rmsprop',
                                      learning_rate=5e-08,
                                      topK=300,
                                      urmSamplingChances=1 / 5,
                                      icmSamplingChances=4 / 5)
    recommender.set_n_tasks(n_training)
    recommender.set_max_threads(1)
    recommender.fit(urm.tocsr(),
                    icm.tocsr(),
                    list(tg_playlist),
                    list(tg_tracks),
                    ds)
    recs = recommender.predict()
    ev.evaluate_fold(recs)


def _prepareICM(icm, dataset):
    from sklearn.feature_extraction.text import TfidfTransformer
    ds = dataset
    tags = ds.build_tags_matrix()
    aggr_tags = aggregate_features(tags, 3, 10)
    extended_icm = sps.vstack((icm, aggr_tags), format='csr')
    # transformer = TfidfTransformer(norm='l1', use_idf=True,
    #                                smooth_idf=True, sublinear_tf=False)
    # # Compute TF-IDF. Returns a (n_items, n_features) matrix
    # tfidf = transformer.fit_transform(extended_icm.transpose())
    # # Keep the top-k features for each item
    # tfidf = top_k_filtering(tfidf, topK=20).tocsr().transpose()
    # tfidf[tfidf.nonzero()] = 1
    return extended_icm


def _testSingleRun():
    ds = Dataset(load_tags=True, filter_tag=True)
    # ds.set_track_attr_weights_2(1, 0.9, 0.2, 0.2, 0.2, 0.1, 0.8, 0.1, 0.1)
    ds.set_track_attr_weights(1, 1, 0, 0, 1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())

    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    print('Building the ICM...')
    icm = ds.build_icm()
    print('Preprocessing the ICM...')
    icm = _prepareICM(icm, ds)

    recommender = BPRSLIM(epochs=100,
                          epochMultiplier=1.0,
                          sgd_mode='momentum',
                          learning_rate=5e-02,
                          topK=300,
                          urmSamplingChances=1 / 5,
                          icmSamplingChances=4 / 5)
    recommender.set_evaluation_every(1, ev)
    recommender.fit(urm.tocsr(),
                    icm.tocsr(),
                    tg_playlist,
                    tg_tracks,
                    ds)
    recs = recommender.predict()
    writeSubmission('submission_cslim_bpr.csv', recs, tg_playlist)


def _fullRun():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0, 0, 1)

    urm = ds.build_train_matrix()
    icm = ds.build_icm().tocsr()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())

    print('Preprocessing the ICM...')
    icm = _prepareICM(icm, ds)
    recommender = BPRSLIM(epochs=300,
                          epochMultiplier=1.0,
                          sgd_mode='rmsprop',
                          learning_rate=5e-08,
                          topK=300,
                          urmSamplingChances=1 / 5,
                          icmSamplingChances=4 / 5)
    recommender.fit(urm.tocsr(),
                    icm.tocsr(),
                    tg_playlist,
                    tg_tracks,
                    ds)
    recs = recommender.predict()
    writeSubmission('submission_cslim_bpr.csv', recs, tg_playlist)


if __name__ == '__main__':
    from src.utils.loader import *
    from src.utils.evaluator import *

    import os

    argument = sys.argv[1]

    if argument == '--testAverage':
        _testAverage(int(sys.argv[2]))
    elif argument == '--testFold':
        _testSingleRun()
    elif argument == '--fullRun':
        _fullRun()

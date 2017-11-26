import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering, dot_chunked
import subprocess
import os, sys
from sklearn.linear_model import SGDRegressor


class MF_BPR():

    def __init__(self, compile_cython=True):
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None
        self.cythonEpoch = None
        self.icm_t = None

        if compile_cython:
            self.runCompilationScript()
        pass

    def fit(self, urm, dataset, tg_playlist, tg_tracks, no_components=300, n_epochs=5, user_reg=1e-1, item_reg=1e-2, l_rate=1e-1, epoch_multiplier=5):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        self.eligibleUsers = []
        urm = urm.tocsr()
        for user_id in range(urm.shape[0]):

            start_pos = urm.indptr[user_id]
            end_pos = urm.indptr[user_id + 1]

            numUserInteractions = len(urm.indices[start_pos:end_pos])

            if numUserInteractions > 0:
                self.eligibleUsers.append(user_id)

        # self.eligibleUsers contains the userID having at least one positive interaction and one item non observed
        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)

        if self.cythonEpoch is None:
            from src.MF.MF_BPR.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch
            self.cythonEpoch = MF_BPR_Cython_Epoch(urm.tocsr(),
                                                   self.eligibleUsers,
                                                   num_factors=no_components,
                                                   learning_rate=l_rate,
                                                   batch_size=1,
                                                   sgd_mode='rmsprop',
                                                   user_reg=user_reg,
                                                   positive_reg=item_reg,
                                                   negative_reg=item_reg,
                                                   epoch_multiplier=epoch_multiplier)

        # start learning
        for i in range(n_epochs):
            self.cythonEpoch.epochIteration_Cython()

        print("Training finished")

        # get W and H to make predictions
        # UxF
        W = self.cythonEpoch.get_W()[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        # IxF
        H = self.cythonEpoch.get_H()[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        self.R_hat = np.dot(W, H.transpose())
        self.R_hat = csr_matrix(top_k_filtering(self.R_hat, 10))

        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]]

        self.R_hat[urm_cleaned.nonzero()] = 0
        self.R_hat.eliminate_zeros()
        print("R_hat done")

    def predict(self, at=5):
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

    def runCompilationScript(self):

        # Run compile script setting the working directory to
        # ensure the compiled
        # file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/src/MF/MF_BPR"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
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


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        mf = MF_BPR()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        for epoch in range(50):
            mf.fit(urm, ds, list(tg_playlist), list(tg_tracks))
            recs = mf.predict()
            ev.evaluate_fold(recs)
        # ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

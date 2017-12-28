import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.BaseRecommender import BaseRecommender
from src.CBF.CBF_MF import *
from src.MF.MF_BPR.MF_BPR import *


# best params: rmsprop, user_reg 1e-1 item_reg 1e-2 l_rate 5e-2 (or 1e-2) epochMult = 5, n_components = 500
#  no_components=100, n_epochs=2, user_reg=1e-2, item_reg=1e-3, l_rate=1e-2, epoch_multiplier=5 0.045 after 20 epochs
class MF_BPR_KNN(BaseRecommender):

    def __init__(self, r_hat_aug=None):
        self.r_hat_aug = r_hat_aug
        pass

    def fit(self, urm, tg_playlist, tg_tracks, dataset, n_epochs=2, no_components=500, epoch_multiplier=2, l_rate=1e-2):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        self.dataset = dataset

        if self.r_hat_aug is None:
            cbf = ContentBasedFiltering()
            cbf.fit(urm, tg_playlist,
                    tg_tracks,
                    ds)

            # get R_hat
            self.r_hat_aug = cbf.getR_hat()

        # put the predictions inside the urm for mf
        self.mf = MF_BPR()

        # call fit on mf bpr
        # MAP@5: 0.08256503053607782 with 500 factors after 10 epochs
        self.mf.fit(self.r_hat_aug, dataset, list(tg_playlist), list(tg_tracks), n_epochs=2,
                    no_components=500, epoch_multiplier=2, l_rate=1e-2)

        # save r-hat
        self.R_hat = self.mf.getR_hat_knn(urm.tocsr())

    def predict(self, at=5):
        R_hat = self.R_hat
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

    def getR_hat(self):
        return self.R_hat

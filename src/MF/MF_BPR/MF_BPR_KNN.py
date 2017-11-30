import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering, dot_chunked
import subprocess
from src.utils.BaseRecommender import BaseRecommender
from src.CBF.CBF_MF import *
from src.MF.MF_BPR.MF_BPR import *


# best params: rmsprop, user_reg 1e-1 item_reg 1e-2 l_rate 5e-2 (or 1e-2) epochMult = 5, n_components = 500
#  no_components=100, n_epochs=2, user_reg=1e-2, item_reg=1e-3, l_rate=1e-2, epoch_multiplier=5 0.045 after 20 epochs
class MF_BPR_KNN(BaseRecommender):

    def __init__(self, r_hat_aug=None): 
        self.r_hat_aug = r_hat_aug
        pass

    def fit(self, urm, dataset, tg_playlist, tg_tracks, n_epochs=2, no_components=500, epoch_multiplier=2, l_rate=1e-2):
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
        self.mf.fit(self.r_hat_aug, ds, list(tg_playlist), list(tg_tracks), n_epochs=2,
                    no_components=500, epoch_multiplier=2, l_rate=1e-2)

        # save r-hat
        self.R_hat = self.mf.getR_hat_knn()

    def getR_hat(self):
        return self.R_hat

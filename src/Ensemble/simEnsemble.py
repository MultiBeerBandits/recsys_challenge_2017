import implicit
from scipy.sparse import *
import numpy as np
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering, compute_cosine
from src.MF.iALS import IALS
from src.ML.CSLIM_parallel import SLIM


class SimEnsemble():
    """Results:
    CBF: 0.0894016109977729
    IBF: 0.336234124300068
    CSLIM: 0.40035618284750263
    """

    def __init__(self):

        # reference to tg playlist and tg tracks
        self.pl_id_list = None
        self.tr_id_list = None
        self.urm = None
        self.dataset = None
        # list of similarities
        self.similarities = []

    def fit(self, urm, tg_playlist, tg_tracks, dataset):
        self.pl_id_list = list(tg_playlist)
        self.tr_id_list = list(tg_tracks)
        self.dataset = dataset
        self.urm = urm

        # Build slim similarity
        slim = SLIM()
        slim.fit(urm, self.pl_id_list, self.tr_id_list, dataset)
        S_cslim = slim.getW()
        S_cslim = S_cslim[:, [dataset.get_track_index_from_id(x) for x in self.tr_id_list]].transpose()

        # Build content based similarity
        icm = dataset.build_icm()
        S_cbf = compute_cosine(icm.transpose()[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]], icm, k_filtering=200, shrinkage=10)

        # Build collaborative similarity
        S_cf = compute_cosine(urm.transpose()[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]], urm, k_filtering=200, shrinkage=10)

        # Build similarity from implicit model
        # ials = IALS(500, 50, 1e-4, 800)
        # ials.fit(urm, tg_playlist, tg_tracks, dataset)
        # item_factors = ials.model.item_factors
        # S_ials = compute_cosine(item_factors[[dataset.get_track_index_from_id(x) for x in self.tr_id_list]], item_factors.transpose(), k_filtering=200, shrinkage=10)

        # append all similarities
        self.similarities.append(S_cbf)
        self.similarities.append(S_cf)
        self.similarities.append(S_cslim)

    def mix(self, params):
        """
        takes an array of models all with R_hat as atttribute
        and mixes them using params
        params: array of attributes
        """
        urm = self.urm
        R_hat_mixed = lil_matrix(
            (len(self.pl_id_list), len(self.tr_id_list)))
        S_mixed = lil_matrix((self.similarities[0].shape[0], self.similarities[0].shape[1]))
        for i in range(len(self.similarities)):
            S_mixed += self.similarities[i].multiply(params[i])
        # normalize S_mixed
        s_norm = S_mixed.sum(axis=1)
        # normalize s
        S_mixed = S_mixed.multiply(csr_matrix(np.reciprocal(s_norm)))
        R_hat_mixed = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]].dot(S_mixed.transpose()).tocsr()
        # clean
        urm_cleaned = urm[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
        urm_cleaned = urm_cleaned[:, [self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        R_hat_mixed[urm_cleaned.nonzero()] = 0
        R_hat_mixed.eliminate_zeros()
        R_hat_mixed = top_k_filtering(R_hat_mixed, topK=20)
        return R_hat_mixed.tocsr()

    def predict(self, params, at=5):
        self.R_hat = self.mix(params)
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

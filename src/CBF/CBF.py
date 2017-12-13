from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.utils.BaseRecommender import BaseRecommender


# 0.1187796227953781
class ContentBasedFiltering(BaseRecommender):

    def __init__(self, shrinkage=10, k_filtering=100):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

        self.shrinkage = shrinkage
        self.k_filtering = k_filtering

    def fit(self, urm, target_playlist, target_tracks, dataset, urm_weight=0.8):
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
        S = None
        print("CBF started")

        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm_2()
        # icm = dataset.add_tracks_num_rating_to_icm(icm, urm)
        # urm_n = np.reciprocal(urm.sum(axis=1))
        # urm = csr_matrix(urm.multiply(urm_n))
        # rationale behind this. If in a playlist there are 1000 songs the similarity between them is low
        #urm_mod = applyTfIdf(urm, topK=1000)

        tags = dataset.build_tags_matrix()
        tags = applyTfIdf(tags, topK=55)
        tags.data = np.ones_like(tags.data)
        icm = vstack([icm, tags], format='csr')
        icm = applyTfIdf(icm, norm=None)
        icm = dataset.add_playlist_to_icm(icm, urm, urm_weight)
        # build user content matrix
        # ucm = dataset.build_ucm()

        # # build item user-feature matrix: UFxI
        # iucm = ucm.dot(urm)

        # iucm_norm = urm.sum(axis=0)
        # iucm_norm[iucm_norm == 0] = 1
        # iucm_norm = np.reciprocal(iucm_norm)
        # iucm = csr_matrix(iucm.multiply(iucm_norm))
        # iucm = top_k_filtering(iucm.transpose(), 100).transpose()
        # icm = vstack([icm.multiply(2), iucm], format='csr')
        # applytfidf
        # icm = TfidfTransformer(norm='l1').fit_transform(icm.transpose()).transpose()
        S = compute_cosine(icm.transpose()[[dataset.get_track_index_from_id(x)
                                            for x in self.tr_id_list]],
                           icm,
                           k_filtering=self.k_filtering,
                           shrinkage=self.shrinkage,
                           chunksize=1000)
        s_norm = S.sum(axis=1)

        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))

        # compute ratings
        print("Similarity matrix ready!")
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        self.S = S.transpose()

        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose().tocsc()).tocsr()
        print("R_hat done")
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        self.R_hat = top_k_filtering(R_hat, 20)

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


def applyTfIdf(matrix, topK=False, norm='l1'):
    transf = TfidfTransformer(norm=norm)
    tfidf = transf.fit_transform(matrix.transpose())
    if topK:
        print("Doing topk")
        tfidf = top_k_filtering(tfidf, topK)
    return tfidf.transpose()

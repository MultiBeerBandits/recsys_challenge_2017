from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from src.utils.feature_weighting import *
import implicit
import src.utils.matrix_utils as utils


class ContentBasedFiltering(object):

    def __init__(self):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

    def fit(self, urm, target_playlist, target_tracks, dataset, shrinkage=50, k_filtering=100):
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
        print("CBF started")
        # Compute URM matrix factorization and predict missing ratings
        print('Implicit matrix factorization on URM...')
        ials = implicit.als.AlternatingLeastSquares(factors=10)
        ials.fit(urm.transpose().multiply(40))
        # Get latent factors
        user_factors = ials.user_factors
        item_factors = ials.item_factors
        print('Estimating URM multiplying latent factors...')
        urm_hat = utils.dot_chunked(user_factors,
                                    item_factors.transpose(),
                                    topK=500,
                                    chunksize=1000)
        urm_hat = lil_matrix(urm_hat)
        urm_hat[urm.nonzero()] = 1
        urm_hat = csr_matrix(urm_hat)
        # get ICM from dataset, assume it already cleaned
        icm = dataset.add_playlist_to_icm(dataset.build_icm(),
                                          urm_hat,
                                          0.25).tocsr()
        # Get tags feature matrix and build the aggregated weighted matrix
        # tags = dataset.build_tags_matrix()
        # tags = build_aggregated_feature_space(tags, n_features=3, topK=10)
        # tags = tags.multiply(0.5)  # Weight tags
        # Stack the ICM on top of aggregated weighted tags features
        # icm = vstack((icm, tags))
        print("SHAPE of ICM: ", icm.shape)
        # apply tfidf
        # transformer = TfidfTransformer()
        # icm = transformer.fit_transform(icm.transpose()).transpose()
        # calculate similarity between items:
        # S_ij=(sum for k belonging to attributes t_ik*t_jk)/norm_i * norm_k
        # first calculate norm
        # sum over rows (obtaining a row vector)

        # Compute cosine similarity matrix on ICM
        icm_t = icm.transpose()[[dataset.get_track_index_from_id(x)
                                 for x in self.tr_id_list]]
        # S is a (n_target_tracks, n_tracks)
        S = utils.compute_cosine(icm_t,
                                 icm,
                                 k_filtering=k_filtering,
                                 shrinkage=shrinkage)
        print("Similarity matrix ready, let's normalize it!")
        # zero out diagonal
        # in the diagonal there is the sim between i and i (1)
        # maybe it's better to have a lil matrix here
        # S.setdiag(0)
        # S.eliminate_zeros()
        # keep only target rows of URM and target columns
        urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        # compute ratings
        R_hat = urm_cleaned.dot(S.transpose().tocsc()).tocsr()
        print("R_hat done")
        # apply mask for eliminating already rated items
        urm_cleaned = urm_cleaned[:, [dataset.get_track_index_from_id(x)
                                      for x in self.tr_id_list]]
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        # eliminate playlist that are not target, already done, to check
        # R_hat = R_hat[:, [dataset.get_track_index_from_id(
        #    x) for x in self.tr_id_list]]
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

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

    def get_model(self):
        """
        Returns the complete R_hat
        """
        return self.R_hat.copy()

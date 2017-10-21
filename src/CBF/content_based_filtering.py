from src.utils.loader import *
from scipy.sparse import *
import scipy
import numpy as np
import numpy.linalg as la
from sklearn.feature_extraction.text import TfidfTransformer
import implicit


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        norm += 1e-6
        # compute the number of non-zeros in each column
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(X.indptr)
        # then normalize the values in each column
        X.data /= np.repeat(norm, col_nnz)
        print("Normalized")

        # 2) compute the cosine similarity using the dot-product
        dist = X * X.T
        print("Computed")
        
        # zero out diagonal values
        dist = dist - dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")
        
        # and apply the shrinkage
        if self.shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")    
        
        return dist

    def apply_shrinkage(self, X, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind * X_ind.T
        # remove the diagonal
        co_counts = co_counts - dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


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
        S = None
        print("CBF started")
        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm()
        print("SHAPE of ICM (pre-als): ", icm.shape)
        als = implicit.als.AlternatingLeastSquares(factors=300)
        als.fit(icm * 40)
        print('Implicit ALS fitted', als.item_factors.shape, als.user_factors.shape)
        icm = csr_matrix(als.user_factors)
        print("SHAPE of ICM (post-als): ", icm.shape)
        # apply tfidf
        # transformer = TfidfTransformer()
        # icm = transformer.fit_transform(icm.transpose()).transpose()
        # calculate similarity between items:
        # S_ij=(sum for k belonging to attributes t_ik*t_jk)/norm_i * norm_k
        # first calculate norm
        # sum over rows (obtaining a row vector)
        # print("Calculating norm")
        # norm = la.norm(icm.todense(), axis=0)
        # print("Calculated norm")
        # norm[(norm == 0)] = 1
        # # normalize
        # icm = icm.multiply(csr_matrix(np.reciprocal(norm)))
        # icm_t = icm.transpose()
        # # clean the transposed matrix, we do not need tracks not target
        # icm_t = icm_t[[dataset.get_track_index_from_id(x)
        #                for x in self.tr_id_list]]
        # chunksize = 1000
        # mat_len = icm_t.shape[0]
        # for chunk in range(0, mat_len, chunksize):
        #     if chunk + chunksize > mat_len:
        #         end = mat_len
        #     else:
        #         end = chunk + chunksize
        #     print(('Building cosine similarity matrix for [' +
        #            str(chunk) + ', ' + str(end) + ') ...'))
        #     # First compute similarity
        #     S_prime = icm_t[chunk:end].tocsr().dot(icm)
        #     print("S_prime prime built.")
        #     # compute common features
        #     icm_t_ones = icm_t[chunk:end]
        #     icm_t_ones[icm_t_ones.nonzero()] = 1
        #     icm_ones = icm.copy()
        #     icm_ones[icm_ones.nonzero()] = 1
        #     S_num = icm_t_ones.dot(icm_ones)
        #     S_den = S_num.copy()
        #     S_den.data += shrinkage
        #     S_den.data = np.reciprocal(S_den.data)
        #     S_prime = S_prime.multiply(S_num).multiply(S_den)
        #     print("S_prime applied shrinkage")
        #     # Top-K filtering.
        #     # We only keep the top K similarity weights to avoid considering many
        #     # barely-relevant neighbors
        #     for row_i in range(0, S_prime.shape[0]):
        #         row = S_prime.data[S_prime.indptr[row_i]:S_prime.indptr[row_i + 1]]

        #         sorted_idx = row.argsort()[:-k_filtering]
        #         row[sorted_idx] = 0

        #     print("S_prime filtered")
        #     S_prime.eliminate_zeros()
        #     if S is None:
        #         S = S_prime
        #     else:
        #         # stack matrices vertically
        #         S = vstack([S, S_prime], format="csr")
        cos = Cosine(shrinkage=100)
        S = cos.compute(icm)
        S = check_matrix(S, 'csr') # nearly 10 times faster
        print("Converted to csr")
        for row_i in range(0, S.shape[0]):
            row = S.data[S.indptr[row_i]:S.indptr[row_i + 1]]

            sorted_idx = row.argsort()[:-k_filtering]
            row[sorted_idx] = 0
        S.eliminate_zeros()
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
        #R_hat = R_hat[:, [dataset.get_track_index_from_id(
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

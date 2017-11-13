import implicit
from scipy.sparse import *
import numpy as np
from src.utils.loader import *
from src.utils.evaluator import *
import numpy.linalg as la
import scipy.sparse.linalg as sLA

class IALS():
    """docstring for NMF"""

    def __init__(self, urm, features, learning_steps, reg, confidence):
        # Number of latent factors
        self.features = features

        # Number of learning iterations
        self.learning_steps = learning_steps

        # Store a reference to URM
        self.urm = urm

        # regularization weight
        self.reg = reg

        # model for making recommendation
        self.model = None

        # confidence, gets multiplied by urm
        self.confidence = confidence

        # reference to tg playlist and tg tracks
        self.pl_id_list = None
        self.tr_id_list = None

    def fit(self, tg_playlist, tg_tracks, dataset, shrinkage=50, k_filtering=200, test_dict={}):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        # initialize a model
        self.model = implicit.als.AlternatingLeastSquares(factors=self.features, iterations=self.learning_steps)

        # train the model on a sparse matrix of item/user/confidence weights
        self.model.fit(self.urm.transpose().multiply(self.confidence))

        print(self.model.item_factors.shape)
        print(self.model.user_factors.shape)

        icm = dataset.add_playlist_to_icm(dataset.build_icm(), urm, 0.4)
        icm = vstack([icm, csr_matrix(self.model.item_factors).transpose()])
        # icm = dataset.add_playlist_attr_to_icm(icm, test_dict)
        # icm = get_icm_weighted_chi2(urm, dataset.build_icm())
        print("SHAPE of ICM: ", icm.shape)
        # apply tfidf
        # transformer = TfidfTransformer()
        # icm = transformer.fit_transform(icm.transpose()).transpose()
        # calculate similarity between items:
        # S_ij=(sum for k belonging to attributes t_ik*t_jk)/norm_i * norm_k
        # first calculate norm
        # sum over rows (obtaining a row vector)
        print("Calculating norm")
        norm = sLA.norm(icm, axis=0)
        print("Calculated norm")
        norm[(norm == 0)] = 1
        print("Normalization done")
        # normalize
        icm = icm.multiply(csr_matrix(np.reciprocal(norm)))
        print("multiplied")
        icm_t = icm.transpose()
        # clean the transposed matrix, we do not need tracks not target
        icm_t = icm_t[[dataset.get_track_index_from_id(x)
                       for x in self.tr_id_list]]
        icm_ones = icm.copy()
        print("Copied")
        icm_ones.data = np.ones_like(icm_ones.data)
        chunksize = 1000
        mat_len = icm_t.shape[0]
        S = None
        for chunk in range(0, mat_len, chunksize):
            if chunk + chunksize > mat_len:
                end = mat_len
            else:
                end = chunk + chunksize
            print(('Building cosine similarity matrix for [' +
                   str(chunk) + ', ' + str(end) + ') ...'))
            # First compute similarity
            S_prime = icm_t[chunk:end].tocsr().dot(icm)
            print("S_prime prime built.")
            # compute common features
            icm_t_ones = icm_t[chunk:end]
            icm_t_ones[icm_t_ones.nonzero()] = 1
            S_num = icm_t_ones.dot(icm_ones)
            S_den = S_num.copy()
            S_den.data += shrinkage
            S_den.data = np.reciprocal(S_den.data)
            S_prime = S_prime.multiply(S_num).multiply(S_den)
            print("S_prime applied shrinkage")
            # Top-K filtering.
            # We only keep the top K similarity weights to avoid considering many
            # barely-relevant neighbors
            for row_i in range(0, S_prime.shape[0]):
                row = S_prime.data[S_prime.indptr[row_i]:S_prime.indptr[row_i + 1]]

                sorted_idx = row.argsort()[:-k_filtering]
                row[sorted_idx] = 0

            print("S_prime filtered")
            S_prime.eliminate_zeros()
            if S is None:
                S = S_prime
            else:
                # stack matrices vertically
                S = vstack([S, S_prime], format="csr")
        print("Similarity matrix ready, let's normalize it!")
        # zero out diagonal
        # in the diagonal there is the sim between i and i (1)
        # maybe it's better to have a lil matrix here
        # S.setdiag(0)
        # S.eliminate_zeros()
        # keep only target rows of URM and target columns
        urm_cleaned = self.urm[[dataset.get_playlist_index_from_id(x)
                           for x in self.pl_id_list]]
        s_norm = S.sum(axis=1)
        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        self.S = S.transpose()
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

    def predict(self, target_playlist, target_tracks, dataset, at=5):
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


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ials = IALS(urm, 50, 20, 1e-6, 50)
        ials.fit(list(tg_playlist), list(tg_tracks), ds)
        recs = ials.predict(list(tg_playlist), list(tg_tracks), ds)
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

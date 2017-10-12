from loader_v2 import *
from scipy.sparse import *
import evaluator


class xSquared():

    def __init__(self):
        self.pl_id_list = []
        self.tr_id_list = []
        self.xsq = None
        self.icm = None
        self.urm = None
        self.dataset = None

    def fit(self, urm, target_playlist, target_tracks, dataset, k_feature=100):
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.dataset = dataset
        S = None
        print("target playlist ", len(self.pl_id_list))
        print("target tracks ", len(self.tr_id_list))
        # get ICM from dataset
        self.icm = dataset.build_icm()
        # clean icm
        self.icm[self.icm.nonzero()] = 1

        self.urm = urm[[dataset.get_playlist_index_from_id(
            x) for x in self.pl_id_list]]

        xsq = self.urm.dot(self.icm.transpose())

        # sort feature and keep only top k_feature
        for row_id in range(0, xsq.shape[0]):
            row = xsq.data[xsq.indptr[row_id]:xsq.indptr[row_id + 1]]
            sorted_idx = row.argsort()[:-k_feature]
            row[sorted_idx] = 0

        xsq.eliminate_zeros()
        self.xsq = xsq
        print("Fit done")

    def predict(self, at=5, shrinkage=130, k_filtering=95):
        recs = {}
        # for each target user compute its item similarity matrix
        # keep top k_filtering for each item
        for pl in range(0, self.xsq.shape[0]):
            pl_id = self.pl_id_list[pl]
            recs[pl_id] = self.predict_one(pl_id)
        return recs

    def predict_one(self, pl_id, at=5, shrinkage=130, k_filtering=95):
        """
        Returns all the at prediction for playlist_id
        """
        recs = {}
        pl = self.pl_id_list.index(pl_id)
        # for each target user compute its item similarity matrix
        # keep top k_filtering for each item
        # extract feature weight of playlist pl
        feature_w = self.xsq[pl]
        # notice: indeces tells which feature is not zero
        # so we can use it to discriminate the feature of icm
        icm_w = self.icm[feature_w.indices]
        # icm reweighted is icm * feaure weight
        icm_w = icm_w.multiply(csr_matrix(feature_w.data).transpose())
        norm = np.sqrt(icm_w.sum(axis=0))
        norm[(norm == 0)] = 1
        # normalize
        icm_w = icm_w.multiply(csr_matrix(1 / norm))
        icm_t = icm_w.transpose()[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        sim = icm_t.dot(icm_w)
        # apply shrinkage
        icm_t_ones = icm_t.copy()
        icm_t_ones[icm_t_ones.nonzero()] = 1
        icm_w_ones = icm_w.copy()
        icm_w_ones[icm_w_ones.nonzero()] = 1
        S_num = icm_t_ones
        S_num = S_num.dot(icm_w_ones)
        S_den = S_num.copy()
        S_den.data += shrinkage
        S_den.data = 1 / S_den.data
        sim = sim.multiply(S_num).multiply(S_den)
        # top k filtering
        for row_i in range(0, sim.shape[0]):
            row = sim.data[sim.indptr[row_i]:sim.indptr[row_i + 1]]
            # if there are too much element then filter them keeping only k
            if row.shape[0] > k_filtering:
                # argpartition does not sort the array, returns indeces of top-k element. linear
                non_max_ind = np.argpartition(row, k_filtering)[:-k_filtering]
                # sorted_idx = row.argsort()[:-k_filtering]
                row[non_max_ind] = 0
        sim.eliminate_zeros()
        r_hat_row = self.urm[pl].dot(sim.transpose()).toarray().ravel()
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(r_hat_row.argsort(), axis=0)[0:at]
        print(sorted_row_idx)
        tracks_ids = [self.tr_id_list[x] for x in sorted_row_idx]
        recs[pl_id] = tracks_ids
        print(pl_id, recs[pl_id])
        return recs

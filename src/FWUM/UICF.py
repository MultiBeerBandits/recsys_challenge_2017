from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from sklearn.decomposition import TruncatedSVD


class xSquared():
    """
    Rationale:
    build a user feature matrix:
    UFM = URM * ICM'
    For each user we have how much is present a feature in its playlist
    Then:
    ufm_u = ufm_u / |Iu| where |Iu| is the number of tracks in this playlist
    [ |U| number of user, UF(f) number of user containing that feature]
    is the inverse user frequency of a feature
    The Inverse User Frequency of a feature is low,
    if it occurs in many users’ profiles
    whereas it is high, if the feature occurs in few users profiles
    Weight feature of UFM by the IUF:
    W (u, f ) = F F (u, f ) ∗ I U F (f )
    So it's the UFM multiplied by the row vector of IUF
    Build user similarity:
    S = UFM*UFM' normalized (and maybe shrinked)
    R_hat = S*URM, find best items in the row of a user
    Idea: two user are similar if the like items with similar features
    """

    def __init__(self):
        self.pl_id_list = []
        self.tr_id_list = []
        self.ufm = None
        self.icm = None
        self.urm = None
        self.dataset = None
        self.R_hat = None

    def fit(self, urm, target_playlist, target_tracks, dataset, k_feature=1000, k_similar=1000):
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.dataset = dataset
        print("FWUM started!")
        # get ICM from dataset
        self.icm = dataset.build_icm()
        print("Initial shape of icm ", self.icm.shape)
        # Apply SVD on ICM
        print("Shape of reduced icm: ", self.icm.shape)
        self.urm = urm[[dataset.get_playlist_index_from_id(x) for x in target_playlist]]
        # CONTENT BASED USER PROFILE
        # build the user feature matrix
        self.ufm = self.urm.dot(self.icm.transpose())
        print("UFM Done!")
        print(self.ufm.shape)

        # Iu contains for each user the number of tracks rated
        Iu = self.urm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        print(Iu.shape)
        # multiply the ufm by Iu. Normalize UFM
        self.ufm = csr_matrix(self.ufm.multiply(Iu))
        # clean stuff
        self.icm = self.icm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        print("User feature matrix done")

        # Throw away useless feature
        for row_i in range(0, self.ufm.shape[0]):
                row = self.ufm.data[self.ufm.indptr[row_i]:self.ufm.indptr[row_i + 1]]
                if row.shape[0] > k_feature:
                    sorted_idx = np.argpartition(row, row.shape[0] - k_feature)[:-k_feature]
                    row[sorted_idx] = 0
        self.ufm.eliminate_zeros()
        # Feature weighting step
        # put to one each element in ufm
        # ufm_ones = ufm.copy()
        # ufm_ones[ufm.nonzero()] = 1
        # # first build IUF(f)
        # uf = ufm_ones.sum(axis=0)
        # uf_copy = uf.copy()
        # uf_copy[uf_copy == 0] = 1
        # iuf = np.reciprocal(uf_copy)
        # iuf[uf == 0] = 0
        # iuf = csr_matrix(iuf)
        # iuf.data = float(dataset.playlists_number) * iuf.data
        # iuf.data = np.log(iuf.data)
        # self.ufm = self.ufm.multiply(iuf)

        # NEIGHBOR FORMATION
        # normalize matrix
        print(self.ufm.shape)
        norm = sLA.norm(self.ufm, axis=1)
        norm[norm == 0] = 1
        # convert norm to csr
        # csr matrix builds a row vector. transpose it
        norm = np.reciprocal(norm).transpose()
        print("Normalizing")
        self.ufm = self.ufm.multiply(csr_matrix(np.reshape(norm, (-1, 1))))
        icm_norm = sLA.norm(self.icm, axis=0)
        icm_norm[icm_norm == 0] = 1
        icm_norm = np.reciprocal(icm_norm)
        print("Normalizing icm")
        icm_normalized = self.icm.multiply(csr_matrix(icm_norm))
        S = self.ufm.dot(self.icm).todense()
        # apply shrinkage factor:
        # Let I_uv be the set of attributes in common of item i and j
        # Let H be the shrinkage factor
        #   Multiply element-wise for the matrix of I_uv (ie: the sim matrix)
        #   Divide element-wise for the matrix of I_uv incremented by H
        # Obtaining I_uv / I_uv + H
        # Rationale:
        # if I_uv is high H has no importante, otherwise has importance
        # shr_num = S.copy()
        # shr_num[shr_num.nonzero()] = 1
        # shr_den = shr_num.copy()
        # shr_den.data += shrinkage
        # shr_den.data = np.reciprocal(shr_den.data)
        # S = S.multiply(shr_num)
        # S = csr_matrix(S.multiply(shr_den))
        print("similarity done")
        # eliminate non target users
        # keep only top k similar user for each row
        indices = np.argpartition(S, S.shape[1] - k_similar, axis=1)[:, :-k_similar] # keep all rows but until k columns
        for i in range(S.shape[0]):
            S[i, indices[i]] = 0
        S = csr_matrix(S)
        # normalize s
        # R_hat computation
        self.R_hat = S
        # clean urm
        self.urm = self.urm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        # put to zero already rated elements
        self.R_hat[self.urm.nonzero()] = 0
        self.R_hat.eliminate_zeros()
        print("R_hat done")

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
        # normalize feature
        feature_w.data = feature_w.data / feature_w.data.max(axis=0)
        print(feature_w.data)
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
        # calculate similarity
        print("Calculating similarity")
        sim = icm_t.dot(icm_w)
        print("Applying shrinkage")
        # apply shrinkage
        # icm_t_ones = self.icm[feature_w.indices].transpose()
        # icm_w_ones = icm_t_ones.transpose()
        # S_num = icm_t_ones
        # S_num = S_num * icm_w_ones
        # S_num = lil_matrix(S_num).setdiag(0)
        # S_num = S_num.to_csr()
        # S_den = S_num.to_csr()
        # S_den.data += shrinkage
        # S_den.data = 1 / S_den.data
        # S_num.data *= S_den.data
        # remove rows from S_num
        # S_num = S_num[[self.dataset.get_track_index_from_id(x) for x in self.tr_id_list]]
        # normalize each row
        s_norm = sim.sum(axis=1)
        s_norm[s_norm == 0] = 1
        # normalize s
        sim = sim.multiply(csr_matrix(1 / s_norm))
        print("Before k filtering on similarity element wise")
        # sim = sim.multiply(S_num)
        # top k filtering on similarity
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
        # clean out already rated item of this user
        r_hat_row[self.urm[pl].nonzero()[0]] = 0
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(r_hat_row.argsort(), axis=0)[0:at]
        print(sorted_row_idx)
        tracks_ids = [self.tr_id_list[x] for x in sorted_row_idx]
        recs[pl_id] = tracks_ids
        print(pl_id, recs[pl_id])
        return recs


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=False)
    ds.set_track_attr_weights(1, 1, 0.1, 0.1, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    xbf = xSquared()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        xbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = xbf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)

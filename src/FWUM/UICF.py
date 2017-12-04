from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from sklearn.decomposition import TruncatedSVD
from src.utils.matrix_utils import compute_cosine, top_k_filtering

# MAP@5: 0.08292090778858459
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
        icm = dataset.build_icm()
        self.urm = urm
        # CONTENT BASED USER PROFILE
        ucm_red = dataset.build_ucm()
        # build the user feature matrix
        # FxUt
        ufm = urm.dot(icm.transpose())[[dataset.get_playlist_index_from_id(x) for x in target_playlist]].transpose()
        print("Start filtering")

        ufm = self.filter_by_topic(ufm, dataset).transpose()
        # Iu contains for each user the number of tracks rated
        Iu = urm[[dataset.get_playlist_index_from_id(x) for x in target_playlist]].sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        print("UFM ready")
        ufm = ufm.multiply(Iu).transpose()
        ucm = vstack([ufm, ucm_red[:,[dataset.get_playlist_index_from_id(x) for x in target_playlist]]], format='csr')
        print("UCM ready")
        ## User Based content profile
        # uFxI
        iucm = ucm_red.dot(urm)[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        i_sum = urm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]].sum(axis=0)
        # save from divide by zero!
        i_sum[i_sum == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        i_sum = np.reciprocal(i_sum)
        # multiply the ufm by Iu. Normalize UFM
        iucm = csr_matrix(iucm.multiply(i_sum))
        # Add playlist to icm
        print("SHAPE of ICM: ", icm.shape)
        # filter iucm
        owner = dataset.build_owner_matrix(iucm)
        owner = top_k_filtering(owner, 50)

        title = dataset.build_title_matrix(iucm)
        title = top_k_filtering(title, 50)

        created_at = dataset.build_created_at_matrix(iucm)
        created_at = top_k_filtering(created_at, 10)

        duration = dataset.build_pl_duration_matrix(iucm)
        duration = top_k_filtering(duration, 10)

        numtracks = dataset.build_numtracks_matrix(iucm)
        numtracks = top_k_filtering(numtracks, 10)
        icm = vstack([icm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]], title, owner, created_at, duration, numtracks], format='csr')

        print("UFM and ICM Done!")
        # NEIGHBOR FORMATION
        # normalize matrix
        R_hat_1 = compute_cosine(ucm.transpose(), icm, k_filtering=500, shrinkage=10)

        # R_hat computation
        self.R_hat = R_hat_1.tocsr()
        # self.R_hat = self.R_hat[[dataset.get_playlist_index_from_id(x) for x in target_playlist]]
        # self.R_hat = self.R_hat[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        # restore original ratings
        # self.R_hat[self.urm.nonzero()] = 1
        # clean urm
        self.urm = self.urm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        self.urm = self.urm[[dataset.get_playlist_index_from_id(x) for x in target_playlist]]
        # put to zero already rated elements
        self.R_hat[self.urm.nonzero()] = 0
        self.R_hat.eliminate_zeros()
        print("R_hat done")
        print("Shape:", self.R_hat.shape)

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

    def filter_by_topic(self, matrix, dataset):
        # filter ufm
        artist = dataset.build_artist_matrix(matrix)
        artist = top_k_filtering(artist, 50)

        album = dataset.build_album_matrix(matrix)
        album = top_k_filtering(album, 50)

        tags = dataset.build_tag_matrix(matrix)
        tags = top_k_filtering(tags, 50)

        duration = dataset.build_duration_matrix(matrix)
        duration = top_k_filtering(duration, 10)

        playcount = dataset.build_playcount_matrix(matrix)
        playcount = top_k_filtering(playcount, 10)

        return vstack([artist, album, tags, duration, playcount], format='csr')


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.2, 0, 0.5, 0.05, 0.05)
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

from src.utils.loader import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import TruncatedSVD
from src.utils.BaseRecommender import BaseRecommender
from src.utils.matrix_utils import dot_chunked_single


class xSquared(BaseRecommender):
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

    def __init__(self, k_similar=500):
        self.pl_id_list = []
        self.tr_id_list = []
        self.ufm = None
        self.icm = None
        self.urm = None
        self.dataset = None
        self.R_hat = None
        self.k_similar = k_similar

    def fit(self, urm, target_playlist, target_tracks, dataset):
        # initialization
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.dataset = dataset
        self.urm = urm

        print("FWUM started!")

        # get ICM from dataset
        self.icm = dataset.build_icm_2()

        # get ucm from ds
        self.ucm = dataset.build_ucm()

        # CONTENT BASED USER PROFILE
        # build the user feature matrix
        ufm = self.urm.dot(self.icm.transpose())

        # Iu contains for each user the number of tracks rated
        Iu = urm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        ufm = np.multiply(ufm, Iu)
        self.ufm = ufm
        print("User feature matrix built done")

        # build owner rating matrix
        orm = dataset.build_owner_item_matrix(self.icm, urm)

        # build owner feature matrix
        # for each owner the average of the feature of its tracks
        ofm = orm.dot(icm.transpose()).transpose()

        # usual normalization
        Iu = orm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        print("OFM ready")
        ofm = ofm.multiply(Iu).transpose()

        # put together the user profile and the owner profile
        # by doing a weighted average
        ufm = ufm.multiply(0.8) + ofm.multiply(0.2)

        # now stack ucm and ufm
        ucm_ext = vstack([ufm,
                          self.ucm],
                         format='csr')
        print("UCM ready")

        # now build the item user content matrix
        iucm = ucm.dot(urm) # User Feature x Items

        # usual normalization
        Iu = urm.sum(axis=0)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        print("OFM ready")
        iucm = iucm.multiply(Iu)

        # now stack icm and user content matrix
        icm_ext = vstack([self.icm,
                          iucm],
                         format='csr')

        ucm_ext = ucm_ext[:, [dataset.get_platlist_index_from_id(x) for x in target_playlist]]
        icm_ext = icm_ext[:, [dataset.get_platlist_index_from_id(x) for x in target_tracks]]

        # now compute the dot product
        R_hat = dot_chunked_single(ucm_ext.transpose(), icm_ext, topK=500)

        # clean urm
        urm = urm[:, [dataset.get_track_index_from_id(x) for x in target_tracks]]
        urm = urm[[dataset.get_playlist_index_from_id(x) for x in target_playlist]]
        # put to zero already rated elements
        R_hat[urm.nonzero()] = 0
        R_hat.eliminate_zeros()
        print("R_hat done")
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

    def getR_hat(self):
        return self.R_hat

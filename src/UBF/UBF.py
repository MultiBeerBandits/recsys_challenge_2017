from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sLA
from sklearn.feature_extraction.text import TfidfTransformer
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, applyTfIdf
from src.utils.BaseRecommender import BaseRecommender



class UserBasedFiltering(BaseRecommender):

    """
    0.06352710548141427
    0.06713337857196691 with tfidf
    """
    def __init__(self, shrinkage=50, k_filtering=200):
        # final matrix of predictions
        self.R_hat = None

        # for keeping reference between playlist and row index
        self.pl_id_list = []
        # for keeping reference between tracks and column index
        self.tr_id_list = []

        self.shrinkage = shrinkage
        self.k_filtering = k_filtering

    def fit(self, urm, target_playlist, target_tracks, dataset):
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
        print("UBF started")

        # get UCM from dataset
        ucm = dataset.build_ucm()

        # add user ratings to ucm
        #ucm = vstack([ucm, urm.transpose()])

        #ucm = applyTfIdf(ucm)

        # build user profile from urm and icm
        icm = dataset.build_icm_2()
        tags = dataset.build_tags_matrix()
        tags = applyTfIdf(tags, topK=55)
        icm = vstack([icm, tags], format='csr')
        ufm = urm.dot(icm.transpose())

        # # Iu contains for each user the number of tracks rated
        Iu = urm.sum(axis=1)
        # save from divide by zero!
        Iu[Iu == 0] = 1
        # since we have to divide the ufm get the reciprocal of this vector
        Iu = np.reciprocal(Iu)
        # multiply the ufm by Iu. Normalize UFM
        ufm = ufm.multiply(Iu).transpose()

        ucm = vstack([ucm.multiply(5), urm.transpose().multiply(5), ufm], format='csr')

        # compute cosine similarity between users
        S = compute_cosine(ucm.transpose()[[dataset.get_playlist_index_from_id(x)
                                            for x in self.pl_id_list]],
                           ucm,
                           k_filtering=self.k_filtering,
                           shrinkage=self.shrinkage)
        s_norm = S.sum(axis=1)
        s_norm[s_norm == 0] = 1
        # normalize s
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        # compute ratings
        print("Similarity matrix ready!")
        urm_cleaned = urm[:, [dataset.get_track_index_from_id(x)
                              for x in self.tr_id_list]]

        R_hat = S.dot(urm_cleaned)

        # clean from already rated items
        print("R_hat done")
        urm_cleaned = urm_cleaned[[dataset.get_playlist_index_from_id(x)
                                   for x in self.pl_id_list]]
        R_hat[urm_cleaned.nonzero()] = 0
        R_hat.eliminate_zeros()
        # eliminate playlist that are not target, already done, to check
        # R_hat = R_hat[:, [dataset.get_track_index_from_id(
        #    x) for x in self.tr_id_list]]
        print("Shape of final matrix: ", R_hat.shape)
        self.R_hat = R_hat

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


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=False, weight_tag=False)
    ds.set_track_attr_weights_2(1, 1, 0.1, 0.1, 0, num_rating_weight=0, inferred_album=1, inferred_duration=0, inferred_playcount=0)
    ds.set_playlist_attr_weights(0.2, 0.2, 0.2, 0, 0)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    ubf = UserBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ubf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = ubf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)

from src.utils.loader import *
from scipy.sparse import *
import numpy as np
from src.utils.feature_weighting import *
from src.utils.matrix_utils import compute_cosine, top_k_filtering, yadistance
from src.utils.BaseRecommender import BaseRecommender


class ContentBasedFiltering(BaseRecommender):

    """
    Good conf: tag aggr 3,10; tfidf l1 norm over all matrix
    MAP@5  0.11772497678137457 with 10 shrinkage,
                                    100 k_filtering and other as before
    MAP@5  0.12039006297936491 urm weight 0.7
    MAP@5  0.12109109578826009 without playcount and duration

    Current best:
    CBF (album 1.0, artists 1.0, no duration/playcount)
        + URM 0.8
        + TOP-55 (TFIDF (tags 1.0))
        MAP@5 0.11897304011860126
        Public leaderboard: 0.09616


    """

    def __init__(self, shrinkage=10, k_filtering=100):
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
        urm = urm.tocsr()
        print("CBF started")
        # get ICM from dataset, assume it already cleaned
        icm = dataset.build_icm()

        # Build the tag matrix, apply TFIDF
        print("Build tags matrix and apply TFIDF...")
        icm_tag = dataset.build_tags_matrix()
        tags = applyTFIDF(icm_tag)

        # Before stacking tags with the rest of the ICM, we keep only
        # the top K tags for each item. This way we try to reduce the
        # natural noise added by such sparse features.
        tags = top_k_filtering(tags.transpose(), topK=55).transpose()

        # User augmented UCM
        # print("Building User augmented ICM")
        # ucm = dataset.build_ucm()
        # ua_icm = user_augmented_icm(urm, ucm)
        # ua_icm = top_k_filtering(ua_icm.transpose(), topK=55).transpose()

        # stack all
        icm = vstack([icm, tags, urm * 0.8], format='csr')
        # icm = vstack([icm, tags, applyTFIDF(urm)], format='csr')

        S = compute_cosine(icm.transpose(),
                           icm,
                           k_filtering=self.k_filtering,
                           shrinkage=self.shrinkage,
                           n_threads=4,
                           chunksize=1000)
        s_norm = S.sum(axis=1)

        # Normalize S
        S = S.multiply(csr_matrix(np.reciprocal(s_norm)))
        print("Similarity matrix ready!")

        self.S = S.transpose()

        # Compute ratings
        R_hat = urm.dot(S.transpose().tocsc()).tocsr()
        print("R_hat done")
        R_hat[urm.nonzero()] = 0
        R_hat.eliminate_zeros()
        R_hat = top_k_filtering(R_hat, topK=5)
        # Remove the entries in R_hat that are already present in the URM
        R_hat[urm.nonzero()] = 1


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


def applyTFIDF(matrix):
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(norm='l1', use_idf=True,
                                   smooth_idf=True, sublinear_tf=False)
    tfidf = transformer.fit_transform(matrix.transpose())
    return tfidf.transpose()


def produceCsv():
    # export csv
    dataset = Dataset(load_tags=True,
                      filter_tag=False,
                      weight_tag=False)
    dataset.set_track_attr_weights_2(1.0, 1.0, 0.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0)
    cbf_exporter = ContentBasedFiltering()
    urm = dataset.build_train_matrix()
    tg_playlist = list(dataset.target_playlists.keys())
    tg_tracks = list(dataset.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cbf_exporter.fit(urm,
                     tg_playlist,
                     tg_tracks,
                     dataset)
    recs = cbf_exporter.predict()
    with open('submission_cbf.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})


def evaluateMap():
    from src.utils.evaluator import Evaluator
    dataset = Dataset(load_tags=True,
                      filter_tag=False,
                      weight_tag=False)
    dataset.set_track_attr_weights_2(2.0, 2.0, 0.0, 0.0, 0.0,
                                     2.0, 2.0, 0.0, 0.0)
    # seed = 0xcafebabe
    # print("Evaluating with initial seed: {}".format(seed))
    ev = Evaluator(seed=False)
    ev.cross_validation(5, dataset.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(dataset)
        cbf.fit(urm,
                list(tg_playlist),
                list(tg_tracks),
                dataset)
        recs = cbf.predict()
        ev.evaluate_fold(recs)

    map_at_five = ev.get_mean_map()
    print("MAP@5 ", map_at_five)


def crossValidation():
    from src.utils.evaluator import Evaluator
    pass


if __name__ == '__main__':
    import sys
    choice = sys.argv[1]

    if choice == '--map':
        evaluateMap()
    elif choice == '--produceCsv':
        produceCsv()
    else:
        print(("Unknown paramter.\n",
               "Usage: {} [--map | --produceCsv]".format(sys.argv[0])))

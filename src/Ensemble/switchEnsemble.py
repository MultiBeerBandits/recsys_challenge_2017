import implicit
from scipy.sparse import *
import numpy as np
from src.utils.loader import *
from src.utils.evaluator import *
from src.utils.matrix_utils import top_k_filtering, compute_cosine
from src.MF.iALS import IALS
from src.ML.CSLIM_parallel import SLIM


class SwitchEnsemble():

    def __init__(self, models):
        """
        models is an array of models implementing BaseRecommender
        """

        # reference to tg playlist and tg tracks
        self.pl_id_list = None
        self.tr_id_list = None
        self.urm = None
        self.dataset = None
        # list of similarities
        self.models = models

    def fit(self, urm, tg_playlist, tg_tracks, dataset):
        self.pl_id_list = list(tg_playlist)
        self.tr_id_list = list(tg_tracks)
        self.dataset = dataset

        # Fit all models
        for model in self.models:
            model.fit(urm.copy(), tg_playlist, tg_tracks, dataset)

    def predict(self, user_clusters, model_per_cluster, at=5):
        """
        user clusters is an array of length tg_users containing
                        for each user its cluster
        model_per_cluster is an array containing
                        for each cluster the index of its model inside models
        """

        # recs is an array of dicts of all models
        recs = []

        # collect all recs
        for i in range(len(self.models)):
            recs.append(self.models[i].predict())

        # mixed recs is the dict created by mixing all recommendations
        mixed_recs = {}

        for i in range(0, len(self.pl_id_list)):

            # get the cluster of current user
            u_cluster = user_clusters[i]

            # get the best model for this cluster
            model = model_per_cluster[u_cluster]

            # use its recs
            mixed_recs[self.pl_id_list[i]] = recs[model][self.pl_id_list[i]]

        return mixed_recs

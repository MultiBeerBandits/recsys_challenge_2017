"""
In this script, we want to understand what models are performing better (read:
have higher MAP@5) according to some clustering of users. The clustering is
made on the number of different tracks that the user added
"""
# Scientific libs
import numpy as np
import scipy.sparse as sps
import pandas as pd

# Our utils
from src.utils.evaluator import Evaluator
from src.utils.loader import *
from src.utils.matrix_utils import top_k_filtering

# Models
from src.CBF.CBF_tfidf import ContentBasedFiltering
from src.Pop.PopCBF import PopularityCBF
from src.IBF.IBF import ItemBasedFiltering
from src.UBF.UBF2 import UserBasedFiltering
from src.CBF.CBF_MF import ContentBasedFiltering as CBF_AUG
from src.MF.MF_BPR.MF_BPR_KNN import MF_BPR_KNN


def map_per_n_tracks(mAP, urm, n_clusters):
    # sum over rows obtaining how many features are interest of this user
    n_tracks = np.array(urm.sum(axis=1)).squeeze()

    # divide in n_cluster
    # pd cut returns for each item to what cluster it belongs
    clusters = pd.cut(n_tracks, n_clusters, labels=False)

    # Build mAP vector, with mAP for each cluster
    map_per_cluster = np.zeros(n_clusters)
    for i in range(n_clusters):
        map_i = np.mean(np.array(mAP[clusters == i]))
        map_per_cluster[i] = map_i

    return map_per_cluster


def results_cbf(dataset, ev, urm, tg_tracks, tg_playlist, n_clusters):
    print("Training CBF...")
    cbf = ContentBasedFiltering()
    cbf.fit(urm,
            list(tg_playlist),
            list(tg_tracks),
            dataset)
    recs = cbf.predict()
    ev.evaluate_fold(recs)

    map_playlists = ev.get_map_playlists()
    maps = np.array([map_playlists[x] for x in list(tg_playlist)])
    urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                       for x in list(tg_playlist)]]

    print("Computing MAP n_tracks clustering...")
    # Compute MAP@5 per cluster of n_tracks
    map_per_tracks = map_per_n_tracks(maps,
                                      urm_cleaned,
                                      n_clusters)

    return map_per_tracks


def results_ibf(dataset, ev, urm, tg_tracks, tg_playlist, n_clusters):
    print("Training Item-based CF...")
    ibf = ItemBasedFiltering()
    ibf.fit(urm,
            list(tg_playlist),
            list(tg_tracks),
            dataset)
    recs = ibf.predict()
    ev.evaluate_fold(recs)

    map_playlists = ev.get_map_playlists()
    maps = np.array([map_playlists[x] for x in list(tg_playlist)])
    urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                       for x in list(tg_playlist)]]

    print("Computing MAP n_tracks clustering...")
    # Compute MAP@5 per cluster of n_tracks
    map_per_tracks = map_per_n_tracks(maps,
                                      urm_cleaned,
                                      n_clusters)

    return map_per_tracks


def results_ubf(dataset, ev, urm, tg_tracks, tg_playlist, n_clusters):
    print("Training User-based CF...")
    ubf = UserBasedFiltering()
    ubf.fit(urm,
            list(tg_playlist),
            list(tg_tracks),
            dataset)
    recs = ubf.predict()
    ev.evaluate_fold(recs)

    map_playlists = ev.get_map_playlists()
    maps = np.array([map_playlists[x] for x in list(tg_playlist)])
    urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                       for x in list(tg_playlist)]]

    print("Computing MAP n_tracks clustering...")
    # Compute MAP@5 per cluster of n_tracks
    map_per_tracks = map_per_n_tracks(maps,
                                      urm_cleaned,
                                      n_clusters)

    return map_per_tracks


def result_pop_cbf(dataset, ev, urm, tg_tracks, tg_playlist, n_clusters):
    print("Training PopularityCBF...")
    cbf = PopularityCBF()
    cbf.fit(urm,
            list(tg_playlist),
            list(tg_tracks),
            dataset)
    recs = cbf.predict()
    ev.evaluate_fold(recs)

    map_playlists = ev.get_map_playlists()
    maps = np.array([map_playlists[x] for x in list(tg_playlist)])
    urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                       for x in list(tg_playlist)]]

    print("Computing MAP n_tracks clustering...")
    # Compute MAP@5 per cluster of n_tracks
    map_per_tracks = map_per_n_tracks(maps,
                                      urm_cleaned,
                                      n_clusters)

    return map_per_tracks


def results_mf(dataset, ev, urm, tg_tracks, tg_playlist, n_clusters):
    print("Training MF...")
    print("  Calculating CBF_AUG first...")
    cbf_aug = CBF_AUG()
    cbf_aug.fit(urm, list(tg_playlist), list(tg_tracks), dataset)

    # get R_hat
    print("  Using calculated R_hat to train BRP_MF...")
    r_hat_aug = cbf_aug.getR_hat()
    model = MF_BPR_KNN(r_hat_aug)
    model.fit(urm, list(tg_playlist), list(tg_tracks), dataset)
    recs = model.predict()
    ev.evaluate_fold(recs)

    map_playlists = ev.get_map_playlists()
    maps = np.array([map_playlists[x] for x in list(tg_playlist)])
    urm_cleaned = urm[[dataset.get_playlist_index_from_id(x)
                       for x in list(tg_playlist)]]

    print("Computing MAP n_tracks clustering...")
    # Compute MAP@5 per cluster of n_tracks
    map_per_tracks = map_per_n_tracks(maps,
                                      urm_cleaned,
                                      n_clusters)

    return map_per_tracks


def train_models(dataset, ev, n_folds, n_clusters):
    """
    @brief      { function_description }

    @param      dataset     The dataset
    @param      ev          { parameter_description }
    @param      n_folds     The n folds
    @param      n_clusters  The n clusters

    @return     A list whose entries are the id of the model that best
                performed for each cluster. The id mapping is the following
                0 - ContentBasedFiltering
                1 - Item-based Collaborative Filtering
                2 - User-based Collaborative Filtering
                3 - BPR Matrix factorization
                4 - Popularity
    """
    best = np.zeros((4, n_clusters), dtype=int)
    for i in range(n_folds):
        print("Training models for fold {}-th".format(i))
        urm, tg_tracks, tg_playlist = ev.get_fold(dataset)

        maps_cbf = results_cbf(dataset, ev, urm,
                               tg_tracks, tg_playlist, n_clusters)

        maps_ibf = results_ibf(dataset, ev, urm,
                               tg_tracks, tg_playlist, n_clusters)

        maps_ubf = results_ubf(dataset, ev, urm,
                               tg_tracks, tg_playlist, n_clusters)

        # maps_mf = results_mf(dataset, ev, urm,
        #                      tg_tracks, tg_playlist, n_clusters)

        maps_pop = result_pop_cbf(dataset, ev, urm,
                                  tg_tracks, tg_playlist, n_clusters)

        print("Computing best model for each cluster...")
        maps = [maps_cbf, maps_ibf, maps_ubf, maps_pop]

        curr_best = [x.index(max(x)) for x in zip(*maps)]
        for cluster, model in enumerate(curr_best):
            best[model, cluster] += 1
    return best


def top_over_cluster(best_matrix):
    return np.argmax(best_matrix, axis=0)


def main(n_evaluations):
    print("Setting up dataset...")
    # Setup dataset and evaluator
    dataset = Dataset(load_tags=True,
                      filter_tag=False,
                      weight_tag=False)
    dataset.set_track_attr_weights_2(1.0, 1.0, 0.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0)

    # Run mAP evaluation a number of times to reduce the variance
    # of the results
    n_clusters = 20
    best = np.zeros((4, n_clusters), dtype=int)
    for i in range(n_evaluations):
        print("Running cross validation no. {}".format(i))
        print("Setting up the Evaluator...")
        ev = Evaluator(seed=False)
        ev.cross_validation(5, dataset.train_final.copy())
        folding_best = train_models(dataset, ev,
                                    n_folds=5, n_clusters=n_clusters)
        best += folding_best
        top = top_over_cluster(best)
        _print_top(top)


# ------------------------- Helper procedures ------------------------- #
def _build_icm(dataset, urm):
    icm = dataset.build_icm()

    # Build the tag matrix, apply TFIDF
    print("Build tags matrix and apply TFIDF...")
    icm_tag = dataset.build_tags_matrix()
    tags = _applyTFIDF(icm_tag)

    # Before stacking tags with the rest of the ICM, we keep only
    # the top K tags for each item. This way we try to reduce the
    # natural noise added by such sparse features.
    tags = top_k_filtering(tags.transpose(), topK=55).transpose()

    # stack all
    icm = sps.vstack([icm, tags, urm * 0.8], format='csr')
    return icm


def _applyTFIDF(matrix):
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(norm='l1', use_idf=True,
                                   smooth_idf=True, sublinear_tf=False)
    tfidf = transformer.fit_transform(matrix.transpose())
    return tfidf.transpose()


def _index_to_str(index):
    if index == 0:
        return 'ContentBasedFiltering'
    elif index == 1:
        return 'Item-based Collaborative Filtering'
    elif index == 2:
        return 'User-based Collaborative Filtering'
    elif index == 3:
        return 'Popularity'
    elif index == 4:
        return 'BPR Matrix factorization'


def _print_top(top):
    msg = ""
    for i, v in enumerate(top):
        msg += "Cluster {}: {}\n".format(i, _index_to_str(v))
    print(msg)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(("Wrong number of parameters...\n",
               "Usage: {} <n_evaluations>".format(sys.argv[0])))
    elif sys.argv[1].isdecimal():
        main(int(sys.argv[1]))
    else:
        print(("{} is not a decimal...\n".format(sys.argv[1]),
               "Usage: {} <n_evaluations>".format(sys.argv[0])))

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from src.utils.evaluator import Evaluator
from src.utils.loader import *
from src.CBF.CBF_tfidf import ContentBasedFiltering
from src.IBF.IBF import ItemBasedFiltering
from src.utils.matrix_utils import top_k_filtering


def map_per_cluster_features(mAP, urm, icm, n_clusters):
    """
    @brief      { function_description }

    @param      mAP         A (n_users) vector whose entries are the
                            MAP@5 for each user
    @param      urm         A (n_users, n_items) sparse matrix. The URM.
    @param      icm         A (n_features, n_items) sparse matrix. The ICM.
    @param      n_clusters  The number of clusters to make out of the
                            the number of features per item

    @return     A (n_clusters) vector whose entries are the average MAP@5
                of each cluster.
    """
    # Cluster on number of features per item
    features_cluster = _cluster_by_n_features(icm, n_clusters)

    # Build mAP vector, with mAP for each cluster
    map_per_cluster = np.zeros(n_clusters)
    for i in range(n_clusters):
        map_i = _map_per_items(mAP, urm, (features_cluster == i))
        map_per_cluster[i] = map_i

    return map_per_cluster


def results_cbf(dataset, ev, urm, icm, tg_tracks, tg_playlist, n_clusters):
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

    print("Computing MAP per each clustering...")
    # Compute MAP@5 per cluster of n_features
    map_per_features = map_per_cluster_features(maps,
                                                urm_cleaned,
                                                icm,
                                                n_clusters)

    # Compute MAP@5 per cluster of n_artists
    artists_icm = dataset.build_artist_matrix(icm)
    map_per_artists = map_per_cluster_features(maps,
                                               urm_cleaned,
                                               artists_icm,
                                               n_clusters)

    # Compute MAP@5 per cluster of n_albums
    artists_icm = dataset.build_album_matrix(icm)
    map_per_albums = map_per_cluster_features(maps,
                                              urm_cleaned,
                                              artists_icm,
                                              n_clusters)
    return {'n_features': map_per_features,
            'n_artists': map_per_artists,
            'n_albums': map_per_albums}


def results_ibf(dataset, ev, urm, icm, tg_tracks, tg_playlist, n_clusters):
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

    print("Computing MAP per each clustering...")
    # Compute MAP@5 per cluster of n_features
    map_per_features = map_per_cluster_features(maps,
                                                urm_cleaned,
                                                icm,
                                                n_clusters)

    # Compute MAP@5 per cluster of n_artists
    artists_icm = dataset.build_artist_matrix(icm)
    map_per_artists = map_per_cluster_features(maps,
                                               urm_cleaned,
                                               artists_icm,
                                               n_clusters)

    # Compute MAP@5 per cluster of n_albums
    artists_icm = dataset.build_album_matrix(icm)
    map_per_albums = map_per_cluster_features(maps,
                                              urm_cleaned,
                                              artists_icm,
                                              n_clusters)
    return {'n_features': map_per_features,
            'n_artists': map_per_artists,
            'n_albums': map_per_albums}


def train_models(dataset, ev, n_clusters):
    urm, tg_tracks, tg_playlist = ev.get_fold(dataset)

    print("Build ICM...")
    icm = _build_icm(dataset, urm)

    maps_cbf = results_cbf(dataset, ev, urm, icm,
                           tg_tracks, tg_playlist, n_clusters)
    maps_ibf = results_ibf(dataset, ev, urm, icm,
                           tg_tracks, tg_playlist, n_clusters)

    visualize_maps(range(n_clusters), maps_cbf, maps_ibf)


def visualize_maps(x, cbf_maps, ibf_maps):
    plt.axis([0, len(x), 0.02, 0.13])
    plt.grid(True)

    # Plot map per n_features
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.scatter(x, cbf_maps['n_features'],
                c='b', label='CBF')
    ax1.scatter(x, ibf_maps['n_features'],
                c='r', label='Item-based CF')
    ax1.set_title("Clustering on n_features")

    # Plot map per n_albums
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.scatter(x, cbf_maps['n_artists'],
                c='b', label='CBF')
    ax2.scatter(x, ibf_maps['n_artists'],
                c='r', label='Item-based CF')
    ax2.set_title("Clustering on n_artists")

    # Plot map per n_artists
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.scatter(x, cbf_maps['n_albums'],
                c='b', label='CBF')
    ax3.scatter(x, ibf_maps['n_albums'],
                c='r', label='Item-based CF')
    ax3.set_title("Clustering on n_albums")

    # plt.savefig("CBF vs Item-based CF")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


def main():
    print("Setting up dataset and evaluator...")
    # Setup dataset and evaluator
    dataset = Dataset(load_tags=True,
                      filter_tag=False,
                      weight_tag=False)
    dataset.set_track_attr_weights_2(1.0, 1.0, 0.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0)
    ev = Evaluator(seed=False)
    ev.cross_validation(5, dataset.train_final.copy())

    for i in range(0, 5):
        print("Training models for fold {}-th".format(i))
        train_models(dataset, ev, n_clusters=20)


# ------------------------- Helper procedures ------------------------- #
def _cluster_by_n_features(icm, n_clusters):
    # Binarize ICM
    icm[icm.nonzero()] = 1

    # Compute number of features per item
    n_features = np.ravel(icm.sum(axis=0))

    # Cluster on number of features per item
    features_cluster = KMeans(n_clusters).fit_predict(
        np.reshape(n_features, (-1, 1)))

    return features_cluster


def _map_per_items(mAP, urm, item_indices):
    urm_cluster = urm[:, item_indices]
    mAP = np.reshape(mAP, (-1, 1))
    map_cluster = urm_cluster.multiply(mAP)
    # Average all the elements
    # The result is the average mAP of the playlists having items
    # belonging to some cluster
    if map_cluster.nnz > 0:
        return np.mean(map_cluster.data, axis=None)
    else:
        return 0


def _build_icm(dataset, urm):
    icm = dataset.build_icm_2()

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


if __name__ == '__main__':
    main()
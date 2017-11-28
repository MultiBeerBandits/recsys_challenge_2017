from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.UBF.UBF2 import *
from src.IBF.IBF import *
from src.CBF.CBF import ContentBasedFiltering
from src.utils.matrix_utils import cluster_per_n_rating,cluster_per_ucm
from src.utils.plotter import visualize_2d
from src.FWUM.UICF import xSquared
from src.FWUM.UICF2 import UserItemFiltering
from src.MF.iALS import IALS
from src.Pop.popularity import Popularity


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.1, 0.1, 0.1, 0.05)
    ds.set_playlist_attr_weights(0.5, 0.6, 0.6, 0.1, 0.1)
    ev = Evaluator()
    ev.cross_validation(3, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    n_clusters = 10

    # create all the models
    cbf = ContentBasedFiltering()
    ibf = ItemBasedFiltering()
    xbf = xSquared()
    uicf = UserItemFiltering()
    # ials = IALS(500, 50, 1e-4, 800)
    ubf = UserBasedFiltering()
    pop = Popularity()

    models = [ibf, cbf, xbf, uicf, ubf, pop]
    names = ["IBF", "CBF", "XBF", "UICF", "UBF", "POP"]
    # for each cluster the best model
    rating_best_for_cluster = np.zeros(n_clusters)
    ucm_best_for_cluster = np.zeros(n_clusters)
    model_name_for_cluster_rating = ["" for x in range(n_clusters)]
    model_name_for_cluster_ucm = ["" for x in range(n_clusters)]

    # cluster per rating
    rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, n_clusters)

     # cluster per ucm
    ucm_cluster = cluster_per_ucm(urm, tg_playlist, ds, n_clusters)

    for model, name in zip(models, names):
        model.fit(urm, tg_playlist, tg_tracks, ds)
        recs = model.predict()
        ev.evaluate_fold(recs)

        mpc = ev.map_per_cluster(tg_playlist, rating_cluster, n_clusters)
        visualize_2d(range(n_clusters), mpc, "Cluster Ratings", "Map@5", "MAP_per_cluster_ratings_" + name + str(n_clusters))

        mpc_ucm = ev.map_per_cluster(tg_playlist, ucm_cluster, n_clusters)
        visualize_2d(range(n_clusters), mpc_ucm, "Cluster UCM", "Map@5", "MAP_per_cluster_ucm" + name + str(n_clusters))

        # save the best model in lists
        for i in range(n_clusters):
            if rating_best_for_cluster[i] < mpc[i]:
                rating_best_for_cluster[i] = mpc[i]
                model_name_for_cluster_rating[i] = name
            if ucm_best_for_cluster[i] < mpc_ucm[i]:
                ucm_best_for_cluster[i] = mpc_ucm[i]
                model_name_for_cluster_ucm[i] = name

    # write result on file
    with open('cluster_info.log', 'w') as f:
        f.write("N_rating \n")
        f.write(str(rating_best_for_cluster))
        f.write(str(model_name_for_cluster_rating))
        f.write("UCM \n")
        f.write(str(ucm_best_for_cluster))
        f.write(str(model_name_for_cluster_ucm))


if __name__ == '__main__':
    main()

from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.UBF.UBF2 import *
from src.CBF.CBF import ContentBasedFiltering
from src.utils.matrix_utils import cluster_per_n_rating,cluster_per_ucm
from src.utils.plotter import visualize_2d
from src.FWUM.UICF import xSquared
from src.MF.iALS import IALS
from src.Pop.popularity import Popularity


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.1, 0.1, 0.1, 0.05)
    ds.set_playlist_attr_weights(0.5, 0.6, 0.6, 0.1, 0.1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    n_clusters = 20

    # create all the models
    cbf = ContentBasedFiltering()
    xbf = xSquared()
    ials = IALS(500, 50, 1e-4, 800)
    ubf = UserBasedFiltering()
    pop = Popularity()

    models = [cbf, xbf, ials, ubf, pop]
    names = ["CBF", "XBF", "IALS", "UBF", "POP"]
    for model, name in zip(models, names):
        model.fit(urm, tg_playlist, tg_tracks, ds)
        recs = model.predict()
        ev.evaluate_fold(recs)
        rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, n_clusters)
        mpc = ev.map_per_cluster(tg_playlist, rating_cluster, n_clusters)
        visualize_2d(range(n_clusters), mpc, "Cluster Ratings", "Map@5", "MAP_per_cluster_" + name + str(n_clusters))


if __name__ == '__main__':
    main()

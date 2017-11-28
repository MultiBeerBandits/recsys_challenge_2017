from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.IBF.IBF import *
from src.utils.matrix_utils import cluster_per_n_rating
from src.utils.plotter import visualize_2d


def main():
    ds = Dataset()
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    ibf = ItemBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ibf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = ibf.predict()
        ev.evaluate_fold(recs)
        rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, 10)
        mpc = ev.map_per_cluster(tg_playlist, rating_cluster, 10)
        visualize_2d(range(10), mpc, "Cluster N Rating", "Map@5", "MAP per cluster IBF")
    map_at_five = ev.get_mean_map()


if __name__ == '__main__':
    main()

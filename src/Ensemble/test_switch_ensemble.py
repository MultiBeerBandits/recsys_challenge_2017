from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.UBF.UBF2 import *
from src.IBF.IBF import *
from src.CBF.CBF import ContentBasedFiltering
from src.utils.matrix_utils import cluster_per_n_rating, cluster_per_ucm
from src.FWUM.UICF import xSquared
from src.FWUM.UICF2 import UserItemFiltering
from src.MF.iALS import IALS
from src.Pop.popularity import Popularity
from src.Ensemble.switchEnsemble import SwitchEnsemble


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.1, 0.1, 0.1, 0.05)
    ds.set_playlist_attr_weights(0.5, 0.6, 0.6, 0.1, 0.1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    n_clusters = 10

    # create all the models
    cbf = ContentBasedFiltering()
    ibf = ItemBasedFiltering()
    xbf = xSquared()
    uicf = UserItemFiltering()

    models = [ibf, cbf, xbf, uicf]

    switch_ensemble = SwitchEnsemble(models)
    switch_ensemble.fit(urm, tg_playlist, tg_tracks, ds)

    # using ucm cluster
    users_cluster_ucm = cluster_per_ucm(urm, tg_playlist, ds, n_clusters)
    # ['UICF', 'UICF', 'IBF', 'UICF', 'XBF', 'CBF', 'CBF', 'UBF', 'XBF', 'UICF']
    model_per_cluster_ucm = [3, 3, 0, 3, 2, 1, 1, 1, 2, 3]
    recs = switch_ensemble.predict(users_cluster_ucm, model_per_cluster_ucm)
    print("Evaluating ucm")
    ev.evaluate_fold(recs)

    # using n_rating cluster
    rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, n_clusters)
    # ['IBF', 'CBF', 'CBF', 'CBF', 'UICF', 'CBF', 'CBF', 'CBF', 'XBF', 'CBF']
    model_per_cluster_rating = [0, 1, 1, 1, 3, 1, 1, 1, 2, 1]
    recs = switch_ensemble.predict(rating_cluster, model_per_cluster_rating)
    print("Evaluating n_rating")
    ev.evaluate_fold(recs)

    # export, pray god
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    switch_ensemble = SwitchEnsemble(models)
    switch_ensemble.fit(urm, tg_playlist, tg_tracks, ds)

    # using ucm cluster
    users_cluster_ucm = cluster_per_ucm(urm, tg_playlist, ds, n_clusters)
    model_per_cluster_ucm = [3, 3, 0, 3, 2, 1, 1, 1, 2, 3]
    recs = switch_ensemble.predict(users_cluster_ucm, model_per_cluster_ucm)
    with open('submission_switch_ensemble.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})


if __name__ == '__main__':
    main()

from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.UBF.UBF2 import *
from src.utils.matrix_utils import cluster_per_n_rating
from src.utils.plotter import visualize_2d


def main():
    best_map = 0
    best_shrinkage = 0
    best_k_filtering = 0
    for shr in range(10, 150, 10):
        for k_f in range(20, 100, 5):
            ds = Dataset()
            ds.set_track_attr_weights(1, 0.9, 0.1, 0.1, 0.1, 0.05)
            ds.set_playlist_attr_weights(0.5, 0.6, 0.6, 0.1, 0.1)
            ev = Evaluator()
            ev.cross_validation(5, ds.train_final.copy())
            ubf = UserBasedFiltering()
            for i in range(0, 5):
                urm, tg_tracks, tg_playlist = ev.get_fold(ds)
                ubf.fit(urm, tg_playlist, tg_tracks, ds)
                recs = ubf.predict()
                ev.evaluate_fold(recs)
                rating_cluster = cluster_per_n_rating(urm, tg_playlist, ds, 10)
                mpc = ev.map_per_cluster(tg_playlist, rating_cluster, 10)
                visualize_2d(range(10), mpc, "Cluster N Rating", "Map@5", "MAP per cluster UBF")
            map_at_five = ev.get_mean_map()
            print("MAP@5 [Shrinkage = " + str(shr) +  ' K_f ' + str(k_f) + ']', map_at_five)
            # Store best shrinkage factor according to highest MAP
            if map_at_five > best_map:
                best_map = map_at_five
                best_shrinkage = shr
                best_k_filtering = k_f

    print(('Best MAP@5 = ' + str(best_map) +
           ' [Shrinkage factor:' + str(best_shrinkage) + ', K_filtering: ' +
           str(best_k_filtering) + ']'))

    # export csv
    ubf_exporter = UserBasedFiltering()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    ubf_exporter.fit(urm,
                     tg_playlist,
                     tg_tracks,
                     ds,
                     best_shrinkage,
                     best_k_filtering)
    recs = ubf_exporter.predict()
    with open('submission_user_based_filtering.csv', mode='w', newline='') as out:
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

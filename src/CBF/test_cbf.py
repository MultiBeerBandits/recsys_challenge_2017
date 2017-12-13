from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF import *
from itertools import product
from src.utils.matrix_utils import cluster_per_n_rating


def main():
    ds = Dataset(load_tags=True, filter_tag=False, weight_tag=False)
    ds.set_track_attr_weights_2(1, 1, 0.01, 0.01, 0, num_rating_weight=1, inferred_album=1, inferred_duration=0, inferred_playcount=0)
    ds.set_playlist_attr_weights(1, 1, 1, 0, 0)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)
        recs = cbf.predict()
        ev.evaluate_fold(recs)

    map_at_five = ev.get_mean_map()
    print("MAP@5 ", map_at_five)

    # export csv
    cbf_exporter = ContentBasedFiltering()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cbf_exporter.fit(urm,
                     tg_playlist,
                     tg_tracks,
                     ds)
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


if __name__ == '__main__':
    main()

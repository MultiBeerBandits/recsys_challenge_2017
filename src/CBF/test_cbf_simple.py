from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.content_based_filtering import *
from itertools import product


def main():
    best_map = 0
    best_album_w = 0
    best_artist_w = 0
    shr_w = 100
    k_f = 50
    ds = Dataset(load_tags=True, filter_tag=False)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.1)
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
    print(("MAP@5 [shrinkage: " + str(shr_w) +
           " k_filtering: " + str(k_f) +
           " album_w: " + str(alb_w) +
           " artist_w: " + str(art_w) + "]: "), map_at_five)

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

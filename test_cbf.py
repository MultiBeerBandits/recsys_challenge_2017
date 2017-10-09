from loader_v2 import *
from test import get_norm_urm, Recommendation
from scipy.sparse import *
import numpy as np
import os.path
import logging
import evaluator
from cbf import *


def main():
    ds = Dataset()
    evaluator.cross_validation(5, ds.train_final)
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = evaluator.get_fold(ds)
        cbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = cbf.predict()
        evaluator.evaluate_fold(recs)
    map_at_five = evaluator.get_mean_map()
    print("MAP@5", map_at_five)

    # export csv
    cbf_exporter = UserBasedFiltering()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    cbf_exporter.fit(urm, tg_playlist, tg_tracks, ds)
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

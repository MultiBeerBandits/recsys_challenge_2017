from loader_v2 import *
from test import get_norm_urm, Recommendation
from scipy.sparse import *
import numpy as np
import os.path
import logging
import evaluator
from cbf import *
from time import gmtime, strftime


def main():
    # number of values of each attributes
    n_values = 5
    artist_weights = np.linspace(0, 1, n_values)
    album_weights = np.linspace(0, 1, n_values)
    duration_weights = np.linspace(0, 1, n_values)
    playcount_weights = np.linspace(0, 1, n_values)

    # iterate over values
    for art_v in artist_weights:
        for alb_v in album_weights:
            for dur_v in duration_weights:
                for pl_v in playcount_weights:
                    if art_v != 0 or alb_v != 0 or dur_v != 0 or pl_v != 0:
                        weight_list = list((art_v, alb_v, dur_v, pl_v))
                        print("Trying weights: ", weight_list)
                        ds = Dataset()
                        ev = evaluator.Evaluator()
                        ev.cross_validation(5, ds.train_final)
                        cbf = ContentBasedFiltering(weight_list)
                        for i in range(0, 5):
                            urm, tg_tracks, tg_playlist = ev.get_fold(ds)
                            cbf.fit(urm, tg_playlist, tg_tracks, ds)
                            recs = cbf.predict()
                            ev.evaluate_fold(recs)
                        map_at_five = ev.get_mean_map()
                        print("MAP@5", map_at_five, "With weights ", weight_list)
                        # csv_name = './cbf_subm/submission_cbf_' + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '.csv'
                        with open('cbf_cv_info.log', mode='a') as out_file:
                            to_write = ' '.join([str(x) for x in weight_list])
                            to_write = 'MAP@5=' + str(map_at_five) + 'Obtained with ' + to_write
                            out_file.write(to_write)
                            out_file.flush()
                        # export csv
                        # cbf_exporter = ContentBasedFiltering(weight_list)
                        # urm = ds.build_train_matrix()
                        # tg_playlist = list(ds.target_playlists.keys())
                        # tg_tracks = list(ds.target_tracks.keys())
                        # cbf_exporter.fit(urm, tg_playlist, tg_tracks, ds)
                        # recs = cbf_exporter.predict()
                        # with open(csv_name, mode='w', newline='') as out:
                        #    fieldnames = ['playlist_id', 'track_ids']
                        #    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
                        #    writer.writeheader()
                        #    for k in tg_playlist:
                        #        track_ids = ''
                        #        for r in recs[k]:
                        #            track_ids = track_ids + r + ' '
                        #        writer.writerow({'playlist_id': k,
                        #                         'track_ids': track_ids[:-1]})


if __name__ == '__main__':
    main()

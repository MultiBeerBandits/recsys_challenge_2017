from loader_v2 import *
from scipy.sparse import *
import evaluator
from NMF import *


def main():
    ds = Dataset()
    ev = evaluator.Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        nmf = NMF(urm, features=150, learning_steps=10000)
        nmf.fit(1e-6, 1e-6)
        recs = nmf.predict(list(tg_playlist), list(tg_tracks), ds)
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 [Features = 100, Iterations: 10000] ", map_at_five)

    # export csv
    urm = ds.build_train_matrix()
    nmf_exporter = NMF(urm, features=150, learning_steps=10000)
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    nmf_exporter.fit(1e-6, 1e-6)
    recs = nmf_exporter.predict(tg_playlist, tg_tracks, ds)
    with open('submission_nmf.csv', mode='w', newline='') as out:
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

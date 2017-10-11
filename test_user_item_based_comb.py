from loader_v2 import *
from scipy.sparse import *
import evaluator
from user_based import *
from item_based_filtering import *


def main():
    """
    One weight for user-based and one for item based
    """
    best_map = 0
    best_user_w = 0
    best_item_w = 0
    for user_w in np.arange(0.0, 1.0, 0.1):
        item_w = 1 - user_w
        ds = Dataset()
        ev = evaluator.Evaluator()
        ev.cross_validation(5, ds.train_final.copy())
        ubf = UserBasedFiltering()
        ibf = ItemBasedFiltering()
        for i in range(0, 5):
            urm, tg_tracks, tg_playlist = ev.get_fold(ds)
            ubf.fit(urm, tg_playlist, tg_tracks, ds)
            ibf.fit(urm, tg_playlist, tg_tracks, ds)
            ub_model = ubf.get_model()
            ib_model = ibf.get_model()
            recs = combine_ratings(ub_model, ib_model, user_w, item_w, tg_playlist, tg_tracks)
            ev.evaluate_fold(recs)
        map_at_five = ev.get_mean_map()
        print("MAP@5 [User Weight = " + str(user_w) +  ' Item Weight ' + str(item_w) + ']', map_at_five)
        # Store best shrinkage factor according to highest MAP
        if map_at_five > best_map:
            best_map = map_at_five
            best_user_w = user_w
            best_item_w = item_w

    print(('Best MAP@5 = ' + str(best_map) +
           ' [User Weight:' + str(best_user_w) + ', Item Weight: ' +
           str(best_item_w) + ']'))

    # export csv
    ubf = UserBasedFiltering()
    ibf = ItemBasedFiltering()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    ubf.fit(urm,
            tg_playlist,
            tg_tracks,
            ds)
    ibf.fit(urm,
            tg_playlist,
            tg_tracks,
            ds)
    ub_model = ubf.get_model()
    ib_model = ibf.get_model()
    recs = combine_ratings(ub_model, ib_model, best_user_w, best_item_w, tg_playlist, tg_tracks)
    with open('submission_user_item_combination.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})


def combine_ratings(model1, model2, w1, w2, target_playlists, target_tracks, at=5):
    model1.data = model1.data * w1
    model2.data = model2.data * w2
    tg_playlist = list(target_playlists)
    tg_tracks = list(target_tracks)
    R_hat = csr_matrix(model1 + model2)
    recs = {}
    for i in range(0, R_hat.shape[0]):
        pl_id = tg_playlist[i]
        pl_row = R_hat.data[R_hat.indptr[i]:
                            R_hat.indptr[i + 1]]
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
        track_cols = [R_hat.indices[R_hat.indptr[i] + x]
                      for x in sorted_row_idx]
        tracks_ids = [tg_tracks[x] for x in track_cols]
        recs[pl_id] = tracks_ids
    return recs


if __name__ == '__main__':
    main()

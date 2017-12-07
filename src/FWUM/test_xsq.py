from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.FWUM.UICF3 import *
from src.CBF.CBF import *


def main():
    ds = Dataset(load_tags=True, filter_tag=True, weight_tag=False)
    ds.set_track_attr_weights_2(1, 1, 0.2, 0.2, 0.2, num_rating_weight=1, inferred_album=1, inferred_duration=0.2, inferred_playcount=0.2)
    ds.set_playlist_attr_weights(0.2, 0.5, 0.9, 0.01, 0.01)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    xbf = xSquared()
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        xbf.fit(urm, tg_playlist, tg_tracks, ds)
        cbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = xbf.predict()
        ev.evaluate_fold(recs)
        recs = cbf.predict()
        ev.evaluate_fold(recs)

        R_hat_mix = cbf.getR_hat().multiply(xbf.getR_hat())
        recs = predict(R_hat_mix, list(tg_playlist), list(tg_tracks))
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)


def predict(R_hat, pl_id_list, tr_id_list, at=5):
    """
    returns a dictionary of
    'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
    """
    recs = {}
    for i in range(0, R_hat.shape[0]):
        pl_id = pl_id_list[i]
        pl_row = R_hat.data[R_hat.indptr[i]:
                            R_hat.indptr[i + 1]]
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
        track_cols = [R_hat.indices[R_hat.indptr[i] + x]
                      for x in sorted_row_idx]
        tracks_ids = [tr_id_list[x] for x in track_cols]
        recs[pl_id] = tracks_ids
    return recs


if __name__ == '__main__':
    main()
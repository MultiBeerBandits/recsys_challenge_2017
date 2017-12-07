from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.FWUM.UICF3 import *


def main():
    ds = Dataset(load_tags=True, filter_tag=True, weight_tag=False)
    ds.set_track_attr_weights_2(1, 1, 0.2, 0.2, 0.2, num_rating_weight=1, inferred_album=1, inferred_duration=0.2, inferred_playcount=0.2)
    ds.set_playlist_attr_weights(0.2, 0.5, 0.9, 0.01, 0.01)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    xbf = xSquared()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        xbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = xbf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 :", map_at_five)


if __name__ == '__main__':
    main()
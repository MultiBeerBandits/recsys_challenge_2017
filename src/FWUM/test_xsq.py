from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
from src.FWUM.XSquared import *


def main():
    best_map = 0
    best_shrinkage = 0
    best_k_filtering = 0
    ds = Dataset(load_tags=True, filter_tag=True)
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
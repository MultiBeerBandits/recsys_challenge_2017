from loader_v2 import *
from scipy.sparse import *
import evaluator
from content_based_filtering import *


def main():
    best_map = 0
    best_shrinkage = 0
    best_k_filtering = 0
    ds = Dataset()
    ev = evaluator.Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist, tg_tracks, ds)
        recs = ibf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 [Shrinkage = " + str(shr) +  ' K_f ' + str(k_f) + ']', map_at_five)


if __name__ == '__main__':
    main()

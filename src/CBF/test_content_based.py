from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF_MF import *
from src.MF.MF_BPR.MF_BPR import MF_BPR
from itertools import product


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights_2(1, 1, 0, 0, 1, num_rating_weight=0, inferred_album=1, inferred_duration=0, inferred_playcount=0)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    mf = MF_BPR(compile_cython=False)
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)

        # get R_hat
        R_hat = cbf.getR_hat()

        # call fit on mf bpr
        for epoch in range(50):
            mf.fit(R_hat, ds, list(tg_playlist), list(tg_tracks), n_epochs=1, no_components=200, epoch_multiplier=0.2, l_rate=1e-2)
            recs = mf.predict_dot_custom(urm)
            ev.evaluate_fold(recs)

    map_at_five = ev.get_mean_map()
    print(map_at_five)


if __name__ == '__main__':
    main()

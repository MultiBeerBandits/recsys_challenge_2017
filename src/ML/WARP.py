from lightfm import LightFM
import lightfm.evaluation
import numpy as np
from scipy.sparse import *
from src.utils.loader import *
from src.utils.evaluator import *


class WARP():

    def __init__(self):
        self.R_hat = None
        self.pl_id_list = None
        self.tr_id_list = None
        pass

    def fit(self, urm, epochs, dataset, tg_playlist, tg_tracks):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        model = LightFM(loss='warp', random_state=2016)
        # Initialize model.
        model.fit(urm, epochs=0)

        icm = dataset.build_icm()

        model.fit_partial(urm,
                          item_features=icm,
                          epochs=epochs, **{'num_threads': 4})

        tr_indices = [dataset.get_track_index_from_id(x) for x in tg_tracks]

        R_hat = lil_matrix((len(tg_playlist), len(tg_tracks)))
        for u in tg_playlist:
            u_id = dataset.get_playlist_index_from_id(u)
            R_hat[u] = model.predict(u_id, tr_indices, num_threads=4)
            print(R_hat[u])
        self.R_hat = csr_matrix(R_hat)

    def predict(self):
        recs = {}
        for i in range(0, self.R_hat.shape[0]):
            pl_id = self.pl_id_list[i]
            pl_row = self.R_hat.data[self.R_hat.indptr[i]:
                                     self.R_hat.indptr[i + 1]]
            # get top 5 indeces. argsort, flip and get first at-1 items
            sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
            track_cols = [self.R_hat.indices[self.R_hat.indptr[i] + x]
                          for x in sorted_row_idx]
            tracks_ids = [self.tr_id_list[x] for x in track_cols]
            recs[pl_id] = tracks_ids
        return recs


def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-' * spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing - 2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing - 2)
        bottom += '+{}'.format('=' * spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        warp = WARP()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        warp.fit(urm, 20, ds, list(tg_playlist), list(tg_tracks))
        recs = warp.predict()
        ev.evaluate_fold(recs)
        ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

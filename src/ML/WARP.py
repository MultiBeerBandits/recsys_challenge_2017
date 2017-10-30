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

    def fit(self, urm, test_urm, dataset, tg_playlist, tg_tracks, no_components=100, n_epochs=100, item_alpha=1e-4, l_rate=5e-2):
        self.pl_id_list = tg_playlist
        self.tr_id_list = tg_tracks
        model = LightFM(loss='warp', learning_rate=l_rate, random_state=2016, no_components=no_components, item_alpha=item_alpha, max_sampled=100,  learning_schedule='adagrad')
        # Initialize model.
        icm = dataset.build_icm().tocsr()

        # iterarray = range(10, 110, 10)

        # patk_learning_curve(model, urm, test_urm, urm, iterarray, item_features=icm.transpose(), **{'num_threads': 4})

        model.fit_partial(urm,
                          item_features=icm.transpose(),
                          epochs=n_epochs, verbose=True, **{'num_threads': 4})
        print("Training finished")

        tr_indices = np.array(
            [dataset.get_track_index_from_id(x) for x in tg_tracks])

        R_hat = lil_matrix((len(tg_playlist), len(tg_tracks)))
        cont = 0
        for u_id in tg_playlist:
            u_index = dataset.get_playlist_index_from_id(u_id)
            R_hat[cont] = model.predict(
                u_index, tr_indices, item_features=icm.transpose(), num_threads=4)
            if cont % 1000 == 0:
                print("Done: ", cont)
            cont += 1
        self.R_hat = csr_matrix(R_hat)

    def predict(self, at=5):
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
        test_urm = ev.get_test_matrix(i, ds)
        warp.fit(urm, test_urm, ds, list(tg_playlist), list(tg_tracks))
        recs = warp.predict()
        ev.evaluate_fold(recs)
        # ev.print_worst(ds)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)

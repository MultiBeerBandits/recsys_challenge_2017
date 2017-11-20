from src.utils.loader import *
from src.utils.evaluator import *
from scipy.sparse import *
import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
from src.utils.matrix_utils import compute_cosine, top_k_filtering
from src.FWUM.UICF import xSquared
from src.CBF.CBF import ContentBasedFiltering
from src.UBF.UBF import UserBasedFiltering


def mix(models, params, normalize_ratings=False):
    """
    takes an array of models all with R_hat as atttribute and mixes them using params
    params: array of attributes
    """
    R_hat_mixed = lil_matrix((models[0].R_hat.shape[0], models[0].R_hat.shape[1]))
    print(R_hat_mixed.shape)
    for i in range(len(models)):
        if normalize_ratings:
            current_r_hat = max_normalize(models[i].R_hat)
        else:
            current_r_hat = models[i].R_hat
        R_hat_mixed += current_r_hat
    return R_hat_mixed.tocsr()


def max_normalize(X):
    """
    Normalizes X by rows dividing each row by its max
    """
    max_r = X.max(axis=1)
    max_r.data = np.reciprocal(max_r.data)
    return X.multiply(max_r)

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

def fit(params):
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(1, 1, 1, 1, 1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    xbf = xSquared()
    cbf = ContentBasedFiltering()
    ubf = UserBasedFiltering()
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    xbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
    cbf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
    ubf.fit(urm.copy(), tg_playlist, tg_tracks, ds)
    recs_xbf = xbf.predict()
    recs_cbf = cbf.predict()
    recs_ubf = ubf.predict()
    ev.evaluate_fold(recs_xbf)
    ev.evaluate_fold(recs_cbf)
    ev.evaluate_fold(recs_ubf)
    # Mix them all
    models = [xbf, cbf, ubf]
    R_hat_mixed = mix(models, params, normalize_ratings=True)
    recs_mix = predict(R_hat_mixed, list(tg_playlist), list(tg_tracks))
    map_5 = ev.evaluate_fold(recs_mix)

    # multiply both
    R_hat_mult = max_normalize(xbf.R_hat).multiply(max_normalize(cbf.R_hat)).multiply(max_normalize(ubf.R_hat))
    recs_mult = predict(R_hat_mult, list(tg_playlist), list(tg_tracks))
    ev.evaluate_fold(recs_mult)

    print("MAP@5 :", map_5)
    return -map_5

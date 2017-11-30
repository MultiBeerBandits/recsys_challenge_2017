from src.utils.loader import *
from scipy.sparse import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF_MF import *
from src.MF.MF_BPR.MF_BPR import MF_BPR
from itertools import product
from sparsesvd import sparsesvd
from src.utils.matrix_utils import top_k_filtering
from src.ML.BPRSLIM_ext import BPRSLIM


def main():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights_2(1, 1, 0, 0, 1, num_rating_weight=0, inferred_album=1, inferred_duration=0, inferred_playcount=0)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    cbf = ContentBasedFiltering()
    for i in range(0, 5):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf.fit(urm, tg_playlist,
                tg_tracks,
                ds)

        # get R_hat
        R_hat = cbf.getR_hat()
        # MAP@5: 0.03575133830958298 with 500 factors and ornly urm
        # 0.04349393048329915 with 1000 factors
        u, s, v = sparsesvd(R_hat.tocsc(), 2500)
        # numpy.dot(ut.T, numpy.dot(numpy.diag(s), vt))
        R_hat_new = csr_matrix(_worker_dot_chunked(np.dot(u.T[[ds.get_playlist_index_from_id(x) for x in list(tg_playlist)]], np.diag(s)), v[:,[ds.get_track_index_from_id(x) for x in list(tg_tracks)]], topK=100))

        urm_cleaned = urm[[ds.get_playlist_index_from_id(x) for x in list(tg_playlist)]]
        urm_cleaned = urm_cleaned[:, [ds.get_track_index_from_id(x) for x in list(tg_tracks)]]
        R_hat_new[urm_cleaned.nonzero()] = 0

        # test if svd is ok
        recs = predict(R_hat_new, list(tg_playlist), list(tg_tracks))
        ev.evaluate_fold(recs)

        mf = MF_BPR()

        # call fit on mf bpr
        for epoch in range(50):
            # MAP@5: 0.08256503053607782 with 500 factors after 10 epochs
            # MAP@5: 0.08594586443489391 with 500 factors afetr 4 epochs no_components=500, epoch_multiplier=2, l_rate=1e-2
            mf.fit(R_hat, ds, list(tg_playlist), list(tg_tracks), n_epochs=1, no_components=700, epoch_multiplier=2, l_rate=1e-3)
            recs = mf.predict_dot_custom(urm)
            ev.evaluate_fold(recs)

            # MAP@5: 0.09407901681369218 with neighborhood
            # MAP@5: 0.09854105406016736 with neighborhood after 4 epochs
            recs = mf.predict_knn_custom(urm)
            ev.evaluate_fold(recs)

    map_at_five = ev.get_mean_map()
    print(map_at_five)


def mainBPR():
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights_2(1, 1, 0, 0, 1, num_rating_weight=0, inferred_album=1, inferred_duration=0, inferred_playcount=0)
    # ds.set_track_attr_weights(1, 1, 0, 0, 1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())

    urm, tg_tracks, tg_playlist = ev.get_fold(ds)

    cbf = ContentBasedFiltering()
    cbf.fit(urm, tg_playlist,
            tg_tracks,
            ds)

    # get R_hat
    R_hat = cbf.getR_hat()

    print('Building the ICM...')
    ds.set_track_attr_weights_2(1, 1, 1, 1, 1, num_rating_weight=1, inferred_album=1, inferred_duration=1, inferred_playcount=1)


    recommender = BPRSLIM(epochs=100,
                          epochMultiplier=3,
                          sgd_mode='rmsprop',
                          learning_rate=5e-08,
                          topK=300,
                          urmSamplingChances=2 / 5,
                          icmSamplingChances=3 / 5,
                          urm_ext=R_hat)
    recommender.set_evaluation_every(1, ev)
    recommender.fit(urm.tocsr(),
                    tg_playlist,
                    tg_tracks,
                    ds)
    # recs = recommender.predict()
    # writeSubmission('submission_cslim_bpr.csv', recs, tg_playlist)

def predict(R_hat, target_playlist, target_tracks, at=5):
    """
    returns a dictionary of
    'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
    """
    recs = {}
    for i in range(0, R_hat.shape[0]):
        pl_id = target_playlist[i]
        pl_row = R_hat.data[R_hat.indptr[i]:
                                 R_hat.indptr[i + 1]]
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
        track_cols = [R_hat.indices[R_hat.indptr[i] + x]
                      for x in sorted_row_idx]
        tracks_ids = [target_tracks[x] for x in track_cols]
        recs[pl_id] = tracks_ids
    return recs

def _worker_dot_chunked(X, Y, topK, chunksize=1000):
    # Unpack parameters
    result = None
    start = 0
    mat_len = X.shape[0]
    for chunk in range(start, mat_len, chunksize):
        if chunk + chunksize > mat_len:
            end = mat_len
        else:
            end = chunk + chunksize
        print('Computing dot product for chunk [{:d}, {:d})...'
              .format(chunk, end))
        X_chunk = X[chunk:end]
        sub_matrix = X_chunk.dot(Y)
        sub_matrix = csr_matrix(top_k_filtering(sub_matrix, topK))
        if result is None:
            result = sub_matrix
        else:
            result = vstack([result, sub_matrix], format='csr')
    return result


if __name__ == '__main__':
    main()

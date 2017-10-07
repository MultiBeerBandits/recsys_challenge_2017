from loader import *
from test import get_norm_urm, Recommendation
from scipy.sparse import *
import numpy as np
import os.path


def build_recs_dict(dataset):
    recs = {}
    target_p = dataset.target_playlists
    for p_id in target_p.keys():
        recs[p_id] = [Recommendation() for x in range(0, 5)]
    return recs


def main():
    ds = Dataset()
    icm = ds.build_icm('./data/tracks_final.csv')
    print('ICM matrix built...')
    icm_csr = csr_matrix(icm)
    norm = np.sqrt(icm_csr.multiply(icm_csr).sum(0))
    print('  Norm done...')
    icm_bar = icm_csr.multiply(csr_matrix(1 / norm))
    print('  Matrix normalized evaluated...')
    c = icm_bar
    c_t = icm_bar.transpose()
    chunksize = 1000
    mat_len = c_t.shape[0]
    recs = build_recs_dict(ds)
    r_hat_final = lil_matrix((ds.playlists_number, ds.tracks_number))
    for chunk in range(0, mat_len, chunksize):
        if chunk + chunksize > mat_len:
            end = mat_len
        else:
            end = chunk + chunksize
        name = ('sim_' + '{:08d}'.format(chunk) +
                '{:08d}'.format(end) + '.npz')
        print(('Building cosine similarity matrix for [' +
               str(chunk) + ', ' + str(end) + ') ...'))
        if os.path.isfile(name):
            S_prime = load_sparse_matrix(name)
        else:
            sim = c_t[chunk:chunk + chunksize].tocsr().dot(c)
            M_s = ds.build_target_tracks_mask(chunk, end).tocsr()
            S_prime = sim.multiply(M_s).tocsr()
            S_prime.eliminate_zeros()
            save_sparse_matrix(name, S_prime)
        r_bar, user_bias, item_bias = get_norm_urm(ds)
        r_bar = r_bar.tocsr()
        r_hat = r_bar.dot(S_prime.transpose()).tocsr()
        print('  Un-normalized r_hat evaluated...')
        urm = ds.build_train_matrix().tocsr()
        mean_norm = urm.dot(S_prime.transpose()).tocsr()
        print('  Mean-Norm evaluated...')
        r_hat = csr_matrix((r_hat / mean_norm))
        print('  Normalized r_hat evaluated...')
        urm_chunk = urm[:, chunk:end].tocsr()
        r_hat[urm_chunk.nonzero()] = 0
        r_hat.eliminate_zeros()
        print('  Chunked r_hat evaluated...\n',
              '  Concatenating to final r_hat...  ', end='')
        r_hat_final[:, chunk:end] = r_hat
        print('Done!')
    user_counter = 0
    for pl_id in recs.keys():
        if user_counter % 100 == 0:
                print('.', end='', flush=True)
        pl_index = ds.playlist_id_mapper[pl_id]
        pl_row = r_hat_final.data[r_hat_final.indptr[pl_index]:r_hat_final.indptr[pl_index + 1]]
        for i in range(0, pl_row.shape[0]):
            track_index = r_hat_final.indeces[r_hat_final.indptr[pl_index] + i]
            track_id = ds.get_track_id_from_index(track_index)
            candidate = Recommendation()
            candidate.rating = pl_row[i]
            candidate.track_id = track_id
            min_rec = min(recs[pl_id])
            if candidate > min_rec:
                recs[pl_id][recs[pl_id].index(min_rec)] = candidate
        user_counter += 1

    # for u_index in range(0, r_hat.shape[0]):
    #     u_id = ds.playlist_index_mapper[u_index]
    #     if u_id in recs:
    #         if user_counter % 100 == 0:
    #             print('.', end='', flush=True)
    #         for ch in range(0, r_hat.shape[1]):
    #             candidate = Recommendation()
    #             candidate.rating = r_hat[u_index, ch]
    #             candidate.track_id = ds.get_track_id_from_index(chunk + ch)
    #             min_rec = min(recs[u_id])
    #             if candidate > min_rec:
    #                 recs[u_id][recs[u_id].index(min_rec)] = candidate
    #         user_counter += 1
    #     print('\n', end='', flush=True)
    with open('submission.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for pl_id in recs.keys():
            track_ids = ''
            recs[pl_id].sort(reverse=True)
            for rec in recs[pl_id]:
                track_ids = track_ids + rec.track_id + ' '
            writer.writerow({'playlist_id': pl_id,
                             'track_ids': track_ids[:-1]})


if __name__ == '__main__':
    main()

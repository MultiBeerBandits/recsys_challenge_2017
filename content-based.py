from loader import *
from test import get_norm_urm, Recommendation
from scipy.sparse import *
import numpy as np
import os.path
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='cbf.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


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
    logging.info('ICM matrix built...')
    icm_csr = csr_matrix(icm)
    norm = np.sqrt(icm_csr.multiply(icm_csr).sum(0))
    print('  Norm done...')
    logging.info('  Norm done...')
    icm_bar = icm_csr.multiply(csr_matrix(1 / norm))
    print('  Matrix normalized evaluated...')
    logging.info(' Matrix normalized evaluated...')
    c = icm_bar
    c_t = icm_bar.transpose()
    chunksize = 1000
    mat_len = c_t.shape[0]
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
        logging.info('Building cosine similarity matrix for [' +
                     str(chunk) + ', ' + str(end) + ') ...')
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
        logging.info('  Un-normalized r_hat evaluated...')
        urm = ds.build_train_matrix().tocsr()
        mean_norm = csr_matrix(urm.dot(S_prime.transpose()))
        print('  Mean-Norm evaluated...')
        logging.info('  Mean-Norm evaluated...')
        c = r_hat / mean_norm
        c[np.isnan(c)] = 0
        r_hat = csr_matrix(c)
        print('  Normalized r_hat evaluated...')
        logging.info('  Normalized r_hat evaluated...')
        urm_chunk = urm[:, chunk:end].tocsr()
        print(r_hat.shape, urm_chunk.shape)
        r_hat = r_hat.tolil()
        r_hat[urm_chunk.nonzero()] = 0
        r_hat = r_hat.tocsr()
        r_hat.eliminate_zeros()
        print('  Chunked r_hat evaluated...\n',
              ' Concatenating to final r_hat...  ', end='')
        logging.info(('  Chunked r_hat evaluated...\n' +
                      '  Concatenating to final r_hat...  '))
        r_hat_final[:, chunk:end] = r_hat
        r_hat_final = r_hat_final.tocsr()
        for r_id in range(0, r_hat_final.shape[0]):
            row = r_hat_final.data[r_hat_final.indptr[r_id]:
                                   r_hat_final.indptr[r_id + 1]]
            sorted_row_idx = row.argsort()[:-5]
            row[sorted_row_idx] = 0
        r_hat_final.eliminate_zeros()
        print('r_hat_final nnz:', r_hat_final.nnz)
        r_hat_final = r_hat_final.tolil()

    user_counter = 0
    r_hat_final = r_hat_final.tocsr()
    with open('submission.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for pl_id in recs.keys():
            if user_counter % 10 == 0:
                print('.', end='', flush=True)
                logging.info(' ' + str(user_counter))
            pl_index = ds.playlist_id_mapper[pl_id]
            pl_row = r_hat_final.data[r_hat_final.indptr[pl_index]:
                                      r_hat_final.indptr[pl_index + 1]]
            if pl_row.shape[0] != 5:
                print(('Warning: playlist row has ' +
                       pl_row.shape[0] + ' tracks...'))
                logging.info(('Warning: playlist row has ' +
                              pl_row.shape[0] + ' tracks...'))
            top_5 = []
            for i in range(0, pl_row.shape[0]):
                t_index = r_hat_final.indices[r_hat_final.indptr[pl_index] + i]
                track_id = ds.get_track_id_from_index(t_index)
                top_5.append(track_id)
            top_5.sort(reverse=True)
            track_ids = ''
            for r in top_5:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': pl_id,
                             'track_ids': track_ids[:-1]})
            user_counter += 1
        print('\n', end='', flush=True)


if __name__ == '__main__':
    main()

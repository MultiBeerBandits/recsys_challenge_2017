from loader_v2 import *
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

class ContentBasedFiltering(object):

    self.H = 10
    # final matrix of predictions
    self.R_hat = None

    # for keeping reference between playlist and row index
    self.pl_id_list = []
    # for keeping reference between tracks and column index
    self.tr_id_list = []

    self.tr_number = 0

    # dataset object
    self.ds = None

    # list of weights associated to attribute values
    # artist, album, duration, playcount
    self.weights = list()

    def __init__(self, weights):
        self.weights = weights

    def fit(self, urm, target_playlist, target_tracks, ds):
        self.pl_id_list = list(target_playlist)
        self.tr_id_list = list(target_tracks)
        self.tr_number = len(target_tracks)
        self.ds = ds
        icm = ds.build_icm('./data/tracks_final.csv', self.weights)
        print('ICM matrix built...')
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
        r_hat_final = lil_matrix((len(target_playlist), ds.tracks_number))
        urm_target = urm[[ds.get_playlist_index_from_id(x) for x in target_playlist]]
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
            # if os.path.isfile(name):
            #    S_prime = load_sparse_matrix(name)
            # else:
            sim = c_t[chunk:chunk + chunksize].tocsr().dot(c)
            M_s = self._build_target_tracks_mask(chunk, end, ds).tocsr()
            S_prime = sim.multiply(M_s).tocsr()
            S_prime.eliminate_zeros()
            # save_sparse_matrix(name, S_prime)
            r_hat = urm_target.dot(S_prime.transpose()).tocsr()
            print('  Un-normalized r_hat evaluated...')
            logging.info('  Un-normalized r_hat evaluated...')
            mean_norm = urm_target.dot(S_prime.transpose())
            print('  Mean-Norm evaluated...')
            logging.info('  Mean-Norm evaluated...')
            r_hat = r_hat / mean_norm
            r_hat[np.isnan(r_hat)] = 0
            r_hat = csr_matrix(r_hat)
            print('  Normalized r_hat evaluated...')
            logging.info('  Normalized r_hat evaluated...')
            urm_chunk = urm_target[:, chunk:end].tocsr()
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
            r_hat_final = r_hat_final.tolil()
        self.R_hat = r_hat_final

    def predict(self, at=5):
        user_counter = 0
        r_hat_final = self.R_hat.tocsr()
        recs = {}
        for pl_index in range(0, len(self.pl_id_list)):
            if user_counter % 10 == 0:
                print('.', end='', flush=True)
                logging.info(' ' + str(user_counter))
            pl_id = self.pl_id_list[pl_index]
            recs[pl_id] = list()
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
                track_id = self.ds.get_track_id_from_index(t_index)
                top_5.append([track_id, pl_row[i]])
            top_5.sort(key=lambda x: x[1], reverse=True)
            track_ids = ''
            for r in top_5:
                track_ids = track_ids + r[0] + ' '
                recs[pl_id].append(r[0])
        return recs

    def _build_target_tracks_mask(self, start, end, ds):
        """
        Returns a (end-start) X #items lil_matrix whose non-zero
        rows are those of the target tracks
        """
        M = lil_matrix((end - start, 1))
        for i in range(start, end):
            if ds.get_track_id_from_index(i) in self.tr_id_list:
                M[i - start] = 1
        return M

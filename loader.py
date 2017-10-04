import csv
from scipy.sparse import *
import numpy as np
import os.path
import time


class Dataset():
    """
    A Dataset contains useful structures for accessing tracks and playlists
    """

    def __init__(self):
        self.prefix = './data/'
        self.tracks_final = load_csv(self.prefix + 'tracks_final.csv',
                                     'track_id')
        self.playlists_final = load_csv(self.prefix + 'playlists_final.csv',
                                        'playlist_id')
        self.target_playlists = load_csv(self.prefix + 'target_playlists.csv',
                                         'playlist_id')
        self.target_tracks = load_csv(self.prefix + 'target_tracks.csv',
                                      'track_id')
        self.train_final = load_train_final(self.prefix + 'train_final.csv')


def load_train_final(path):
    res = {}
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            res.setdefault(row['playlist_id'], list())
            res[row['playlist_id']].append(row['track_id'])
    return res


def load_csv(path, dict_id):
    """
    :param path is a string of the path of the file to parse
    :param dict_id is a string of the name of key of dataset file (TODO)
    """
    data = {}
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        fields = reader.fieldnames
        for row in reader:
            data[row[dict_id]] = {key: row[key]
                                  for key in fields}
    return data


def build_train_final_matrix(dataset, filename='csr_urm.npz'):
    """
    Builds the user rating matrix from the dataset.
    Before loading check if it is already saved in dataset.data + filename
    """
    path = dataset.prefix + filename
    # if the matrix is already serialized load it
    # timing
    start_time = time.time()
    if os.path.isfile(path):
        print("File found, retrieving urm from it.")
        M = load_sparse_matrix(path)
        # print time
        print("Load from file takes {:.2f} seconds".format(
              time.time() - start_time))
        return M
    rows = max(map(int, dataset.train_final.keys()))
    columns = max([max(map(int, x)) for x in dataset.train_final.values()])
    M = lil_matrix((rows + 1, columns + 1))
    for k, v in dataset.train_final.items():
        for track in v:
            M[k, track] = 1
    print("Build urm takes {:.2f} seconds".format(time.time() - start_time))
    print("Serializing urm matrix to " + path)
    # serialize it to path
    save_sparse_matrix(path, M)
    return M


def save_sparse_matrix(filename, matrix):
    """
    Saves the matrix to the filename, matrix must be a lil matrix
    """
    # convert to a csr matrix since savez needs arrays
    m = csr_matrix(matrix)
    np.savez(filename, data=m.data, indices=m.indices,
             indptr=m.indptr, shape=m.shape)


def load_sparse_matrix(filename):
    """
    Load the matrix contained in the file as csr matrix and convert it to lil
    """
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape']).tolil()

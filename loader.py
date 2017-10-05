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
        self.max = {}
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


def build_tracks_matrix(path):
    """
    Returns the icm matrix encoded as follows: AxI (A is the number of attributes and I is the number of items)
    """
    # attrs is a dict that contains for every attribute its different values
    # used for computing matrix rows (sum of length of sets)
    attrs = {'artist_id': set(), 'album': set(), 'tags': set()}
    # mapper from track id to column index
    track_id_mapper = {}
    # this is the number of columns of the matrix of the matrix
    columns = 0
    if not os.path.isfile('./data/icm_tracks_csr.npz'):
        with open(path, newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t')
            for row in reader:
                attrs['artist_id'].add(row['artist_id'])
                albums = parse_csv_array(row['album'])
                for album in albums:
                    attrs['album'].add(album)
                tags = parse_csv_array(row['tags'])
                for tag in tags:
                    attrs['tags'].add(tag)
                #if int(row['track_id']) > columns:
                #    columns = int(row['track_id'])
                track_id_mapper[row['track_id']] = columns
                columns += 1
        mapper = {'artist_id': {}, 'album': {}, 'tags': {}}
        rows = 0
        for v in attrs['artist_id']:
            mapper['artist_id'][v] = rows
            rows += 1
        for v in attrs['album']:
            mapper['album'][v] = rows
            rows += 1
        for v in attrs['tags']:
            mapper['tags'][v] = rows
            rows += 1
        print("Creating icm :" + str(rows) + ' ' + str(columns))
        icm = lil_matrix((rows, columns))
        with open(path, newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t')
            for row in reader:
                icm[mapper['artist_id'][row['artist_id']],
                    track_id_mapper[row['track_id']]] = 1
                albums = parse_csv_array(row['album'])
                for album in albums:
                    icm[mapper['album'][album], track_id_mapper[row['track_id']]] = 1
                tags = parse_csv_array(row['tags'])
                for tag in tags:
                    icm[mapper['tags'][tag], track_id_mapper[row['track_id']]] = 1
        save_sparse_matrix('./data/icm_tracks_csr.npz', icm)
    else:
        icm = load_sparse_matrix('./data/icm_tracks_csr.npz')
    return icm


def parse_csv_array(arr_str):
    """
    Parse an array string formatted as [el1,el2,el3]
    Returns a list of item, or an empty list if no items
    """
    arr_str = arr_str.replace('[', '')
    arr_str = arr_str.replace(']', '')
    if arr_str == '' or arr_str == 'None':
        return ()
    else:
        return arr_str.split(', ')


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
    m = matrix.tocsr()
    np.savez(filename, data=m.data, indices=m.indices,
             indptr=m.indptr, shape=m.shape)


def load_sparse_matrix(filename):
    """
    Load the matrix contained in the file as csr matrix and convert it to lil
    type str of sparse matrix type
    """
    loader = np.load(filename)
    m = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                   shape=loader['shape']).tolil()
    return m

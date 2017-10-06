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
        # prefix of data folder
        self.prefix = './data/'
        # initialization of user rating matrix
        self.urm = None
        # build tracks mappers
        # track_id_mapper maps tracks id to columns of icm
        # format: {'item_id': column_index}
        # track_index_mapper maps index to track id
        # format: {column index: 'track_id'}
        # track_attr_mapper maps attributes to row index
        # format: {'artist_id': {'artist_key': row_index}}
        self.track_id_mapper, self.track_index_mapper, self.track_attr_mapper = build_tracks_mappers(
            self.prefix + 'tracks_final.csv')
        # build playlist mappers
        # playlist_id_mapper maps playlist id to columns of ucm
        # format: {'item_id': column_index}
        # playlist_index_mapper maps index to playlist id
        # format: {column index: 'playlist_id'}
        # playlist_attr_mapper maps attributes to row index
        # format: {'title': {'title_key': row_index}}
        self.playlist_id_mapper, self.playlist_index_mapper, self.playlist_attr_mapper = build_playlists_mappers(
            self.prefix + 'playlists_final.csv')
        # number of playlists
        self.playlists_number = len(self.playlist_id_mapper.keys())
        # number of tracks
        self.tracks_number = len(self.track_id_mapper.keys())
        self.tracks_final = load_csv(self.prefix + 'tracks_final.csv',
                                     'track_id')
        self.playlists_final = load_csv(self.prefix + 'playlists_final.csv',
                                        'playlist_id')
        self.target_playlists = load_csv(self.prefix + 'target_playlists.csv',
                                         'playlist_id')
        self.target_tracks = load_csv(self.prefix + 'target_tracks.csv',
                                      'track_id')
        self.train_final = load_train_final(self.prefix + 'train_final.csv')

    def build_icm(self, path):
        """
        returns the item content matrix using mappers defined in dataset class
        icm matrix encoded as follows:
        AxI (A is the number of attributes and I is the number of items)
        """
        if not os.path.isfile('./data/icm_tracks_csr.npz'):
            with open(path, newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter='\t')
                for row in reader:
                    # get index of this track
                    track_index = self.track_id_mapper[row['track_id']]
                    # set attributes in icm
                    artist_index = self.attr_mapper['artist_id'][row['artist_id']]
                    icm[artist_index, track_index] = 1
                    albums = parse_csv_array(row['album'])
                    for album in albums:
                        album_index = self.attr_mapper['album'][album]
                        icm[album_index, track_index] = 1
                    tags = parse_csv_array(row['tags'])
                    for tag in tags:
                        tag_index = self.attr_mapper['tags'][tag]
                        icm[tag_index, track_index] = 1
            save_sparse_matrix('./data/icm_tracks_csr.npz', icm)
        else:
            icm = load_sparse_matrix('./data/icm_tracks_csr.npz')
        return icm

    def build_train_matrix(self, filename='csr_urm.npz'):
        """
        Builds the user rating matrix from the dataset.
        Before loading check if it is already saved in dataset.data + filename
        """
        if self.urm is None:
            path = self.prefix + filename
            # if the matrix is already serialized load it
            # timing
            start_time = time.time()
            if os.path.isfile(path):
                print("File found, retrieving urm from it.")
                self.urm = load_sparse_matrix(path)
                # print time
                print("Load from file takes {:.2f} seconds".format(
                      time.time() - start_time))
                return self.urm
            self.urm = lil_matrix((self.playlists_number, self.tracks_number))
            for k, v in self.train_final.items():
                for track in v:
                    row = self.playlist_id_mapper[k]
                    column = self.track_id_mapper[track]
                    self.urm[row, column] = 1
            print("Build urm takes {:.2f} seconds".format(
                time.time() - start_time))
            print("Serializing urm matrix to " + path)
            print(self.urm.shape)
            # serialize it to path
            save_sparse_matrix(path, self.urm)
        return self.urm

    def get_track_id_from_index(self,index):
        return self.track_index_mapper[index]

    def get_track_index_from_id(self,tr_id):
        return self.track_id_mapper[tr_id]


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


def build_tracks_mappers(path):
    """
    Build the mappers of tracks
    """
    # attrs is a dict that contains for every attribute its different values
    # used for mappers
    attrs = {'artist_id': set(), 'album': set(), 'tags': set()}
    # mapper from track id to column index. key is track id value is column
    track_id_mapper = {}
    # mapper from index to track id. key is column, value is id
    track_index_mapper = {}
    # this is the number of columns of the matrix of the matrix
    track_index = 0
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
            track_id_mapper[row['track_id']] = track_index
            track_index_mapper[track_index] = row['track_id']
            track_index += 1
    mapper = {'artist_id': {}, 'album': {}, 'tags': {}}
    attr_index = 0
    for v in attrs['artist_id']:
        mapper['artist_id'][v] = attr_index
        attr_index += 1
    for v in attrs['album']:
        mapper['album'][v] = attr_index
        attr_index += 1
    for v in attrs['tags']:
        mapper['tags'][v] = attr_index
        attr_index += 1
    return track_id_mapper, track_index_mapper, attrs


def build_playlists_mappers(path):
    """
    Builds the mapper for playlist
    """
    attrs = {'title': set(), 'owner': set()}
    # mapper from playlist id to row index
    playlist_id_mapper = {}
    # mapper from row index to playlist id
    playlists_index_mapper = {}
    playlist_index = 0
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            titles = parse_csv_array(row['title'])
            for title in titles:
                attrs['title'].add(title)
            owner = row['owner']
            attrs['owner'].add(owner)
            playlist_id_mapper[row['playlist_id']] = playlist_index
            playlists_index_mapper[playlist_index] = row['playlist_id']
            playlist_index += 1
    mapper = {'title': {}, 'owner': {}}
    attr_index = 0
    for v in attrs['title']:
        mapper['title'][v] = attr_index
        attr_index += 1
    for v in attrs['owner']:
        mapper['owner'][v] = attr_index
        attr_index += 1
    return playlist_id_mapper, playlists_index_mapper, attrs


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

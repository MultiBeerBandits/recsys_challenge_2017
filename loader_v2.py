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
        # Initialize clusters for duration and playcount
        self.duration_intervals = 10
        self.playcount_intervals = 10
        # build tracks mappers
        # track_id_mapper maps tracks id to columns of icm
        # format: {'item_id': column_index}
        # track_index_mapper maps index to track id
        # format: {column index: 'track_id'}
        # track_attr_mapper maps attributes to row index
        # format: {'artist_id': {'artist_key': row_index}}
        self.track_id_mapper, self.track_index_mapper, self.track_attr_mapper, self.attrs_number = build_tracks_mappers(
            self.prefix + 'tracks_final.csv', self)
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

        # weights of attributes
        self.artist_weight = 1
        self.album_weight = 1
        self.duration_weight = 0.2
        self.playcount_weight = 0.2

    def build_icm(self, path):
        """
        returns the item content matrix using mappers defined in dataset class
        icm matrix encoded as follows:
        AxI (A is the number of attributes and I is the number of items)
        """
        if not os.path.isfile('./data/icm_tracks_csr.npz'):
            icm = lil_matrix((self.attrs_number, self.tracks_number))
            with open(path, newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter='\t')
                for row in reader:
                    # get index of this track
                    track_index = self.track_id_mapper[row['track_id']]
                    # set attributes in icm
                    artist_id = row['artist_id']
                    artist_index = self.track_attr_mapper['artist_id'][row['artist_id']]
                    icm[artist_index, track_index] = self.artist_weight
                    # albums
                    albums = parse_csv_array(row['album'])
                    for album in albums:
                        album_index = self.track_attr_mapper['album'][album]
                        icm[album_index, track_index] = self.album_weight
                    # duration
                    duration = row['duration']
                    if duration is not None and duration != '' and float(duration) != -1:
                        duration = float(duration) - self.min_duration
                        duration_offset = min(int(duration / self.duration_int), self.duration_intervals - 1)
                        duration_index = self.track_attr_mapper['duration'] + \
                            duration_offset
                        icm[duration_index, track_index] = self.duration_weight
                    # playcount
                    playcount = row['playcount']
                    if playcount is not None and playcount != '' and float(playcount) != -1:
                        playcount = float(playcount) - self.min_playcount
                        playcount_offset = min(int(playcount / self.playcount_int), self.playcount_intervals - 1)
                        playcount_index = self.track_attr_mapper['playcount'] + \
                            playcount_offset
                        icm[playcount_index, track_index] = self.playcount_weight
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

    def get_track_id_from_index(self, index):
        return self.track_index_mapper[index]

    def get_track_index_from_id(self, tr_id):
        return self.track_id_mapper[tr_id]

    def get_playlist_id_from_index(self, index):
        return self.playlist_index_mapper[index]

    def get_playlist_index_from_id(self, pl_id):
        return self.playlist_id_mapper[pl_id]

    def build_target_tracks_mask(self, start, end):
        """
        Returns a (end-start) X #items lil_matrix whose non-zero
        rows are those of the target tracks"""
        M = lil_matrix((end - start, self.tracks_number))
        for i in range(start, end):
            if self.get_track_id_from_index(i) in self.target_tracks:
                M[i - start] = 1
        return M


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


def build_tracks_mappers(path, dataset):
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
    # duration and playcount attributes
    min_playcount = 10
    max_playcount = 0
    min_duration = 224000
    max_duration = 224000
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
            # duration
            # duration is -1 if not present
            if row['duration'] is not None and row['duration'] != '':
                duration = float(row['duration'])
                # threshold for max and cleaning
                if duration != -1 and duration / 1000 < 700:
                    if duration < min_duration:
                        min_duration = duration
                    if duration > max_duration:
                        max_duration = duration
            if row['playcount'] is not None and row['playcount'] != '':
                playcount = float(row['playcount'])
                # threshold for max
                if playcount < 5000:
                    if playcount < min_playcount:
                        min_playcount = playcount
                    if playcount > max_playcount:
                        max_playcount = playcount
            track_index += 1
    # is a dictionary of dictionary
    # for each attrbute a dictionary of keys of attribute and their index
    mapper = {'artist_id': {}, 'album': {}, 'duration': 0, 'playcount': 0}
    attr_index = 0
    for v in attrs['artist_id']:
        mapper['artist_id'][v] = attr_index
        attr_index += 1
    for v in attrs['album']:
        mapper['album'][v] = attr_index
        attr_index += 1
    # compute ranges
    dataset.duration_int = (max_duration - min_duration) / \
        dataset.duration_intervals
    dataset.playcount_int = (
        max_playcount - min_playcount) / dataset.playcount_intervals
    # set min duration and min playcount
    dataset.min_duration = min_duration
    dataset.min_playcount = min_playcount
    # set index of duration and playcount
    mapper['duration'] = attr_index
    mapper['playcount'] = attr_index + dataset.duration_intervals

    attr_index += dataset.duration_intervals + dataset.playcount_intervals + 1

    return track_id_mapper, track_index_mapper, mapper, attr_index


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

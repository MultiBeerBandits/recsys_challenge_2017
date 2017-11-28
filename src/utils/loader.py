import csv
from scipy.sparse import *
import numpy as np
import os.path
import time
import pandas as pd
# Cluster stuff
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from operator import indexOf


class Dataset():
    """
    A Dataset contains useful structures for accessing tracks and playlists
    """

    def __init__(self, load_tags=False, filter_tag=False):
        # Load_tags is true if need to load tags
        # prefix of data folder
        self.prefix = './data/'
        self.load_tags = load_tags
        # initialization of user rating matrix
        self.urm = None
        # Initialize clusters for duration and playcount
        # selected by studying the error of k-means
        self.duration_intervals = 20
        self.playcount_intervals = 20
        self.pop_threshold = 5
        # for created_at of playlists
        self.created_at_intervals = 30
        self.playlist_duration_intervals = 30
        self.playlist_numtracks_intervals = 30
        # for numbers of cluster of ratings
        self.playlist_num_rating_cluster_size = 25
        self.tracks_num_rating_cluster_size = 25
        # build tracks mappers
        # track_id_mapper maps tracks id to columns of icm
        # format: {'item_id': column_index}
        # track_index_mapper maps index to track id
        # format: {column index: 'track_id'}
        # track_attr_mapper maps attributes to row index
        # format: {'artist_id': {'artist_key': row_index}}
        # tag counter maps tags to its frequency normalized
        #self.track_id_mapper, self.track_index_mapper, self.track_attr_mapper, self.attrs_number, self.tag_counter = build_tracks_mappers_clusters(
        #    self.prefix + 'tracks_final.csv', self, load_tags, filter_tag)
        # extended version
        self.track_id_mapper, self.track_index_mapper, self.track_attr_mapper, self.attrs_number, self.tag_counter, self.album_artist_counter, self.album_artist = build_tracks_mappers_clusters_ext(
            self.prefix + 'tracks_final.csv', self, load_tags, filter_tag)
        # build playlist mappers
        # playlist_id_mapper maps playlist id to columns of ucm
        # format: {'item_id': column_index}
        # playlist_index_mapper maps index to playlist id
        # format: {column index: 'playlist_id'}
        # playlist_attr_mapper maps attributes to row index
        # format: {'title': {'title_key': row_index}}
        self.playlist_id_mapper, self.playlist_index_mapper, self.playlist_attr_mapper, self.playlist_attrs_number = build_playlists_mappers(
            self.prefix + 'playlists_final.csv', self)
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
        # Train final is a dict with pl_key tracks
        self.train_final = load_train_final(self.prefix + 'train_final.csv')

        # weights of attributes of tracks
        self.artist_weight = 1
        self.album_weight = 1
        self.duration_weight = 1
        self.playcount_weight = 1
        self.tags_weight = 1
        self.track_num_rating_weight = 1

        # weights of attributes of playlist
        self.created_at_weight = 1
        self.owner_weight = 1
        self.title_weight = 1
        self.playlist_duration_weight = 1
        self.playlist_numtracks_weight = 1
        self.playlist_num_rating_weight = 1

    def set_track_attr_weights(self, art_w, alb_w, dur_w, playcount_w, tags_w, num_rating_weight=1, artist_album_weight=0.9):
        self.artist_weight = art_w
        self.album_weight = alb_w
        self.artist_album_weight = artist_album_weight
        self.duration_weight = dur_w
        self.playcount_weight = playcount_w
        self.tags_weight = tags_w
        self.track_num_rating_weight = num_rating_weight

    def set_playlist_attr_weights(self, created_at_weight, owner_weight, title_weight, duration_weight, numtracks_weight=1):
        self.created_at_weight = created_at_weight
        self.owner_weight = owner_weight
        self.title_weight = title_weight
        self.playlist_duration_weight = duration_weight
        self.playlist_numtracks_weight = numtracks_weight

    def build_icm(self, path='./data/tracks_final.csv'):
        """
        returns the item content matrix using mappers defined in dataset class
        icm matrix encoded as follows:
        AxI (A is the number of attributes and I is the number of items)
        """
        icm = lil_matrix((self.attrs_number, self.tracks_number))
        duration_index = 0
        playcount_index = 0
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
                # load tags only if specified
                if self.load_tags:
                    # tags
                    tags = parse_csv_array(row['tags'])
                    for tag in tags:
                        if tag in self.track_attr_mapper['tags']:
                            tag_index = self.track_attr_mapper['tags'][tag]
                            tag_weight = self.tag_counter[tag] * \
                                self.tags_weight
                            icm[tag_index, track_index] = tag_weight
                # duration
                duration = row['duration']
                if duration is not None and duration != '' and float(duration) != -1:
                    # duration = float(duration) - self.min_duration
                    # duration_offset = min(
                    #     int(duration / self.duration_int), self.duration_intervals - 1)
                    duration = float(duration)
                    # duration_offset = np.digitize(duration, self.duration_bins) - 1
                    # get the cluster
                    duration_offset = self.duration_cluster[duration_index]
                    duration_index = self.track_attr_mapper['duration'] + \
                        duration_offset
                    icm[duration_index, track_index] = self.duration_weight
                    # go ahead with duration index
                    duration_index += 1
                # playcount
                playcount = row['playcount']
                if playcount is not None and playcount != '' and float(playcount) != -1:
                    # playcount = float(playcount) - self.min_playcount
                    # playcount_offset = min(
                    #     int(playcount / self.playcount_int), self.playcount_intervals - 1)
                    playcount = float(playcount)
                    # playcount_offset = np.digitize(playcount, self.playcount_bins) - 1
                    # get the cluster
                    playcount_offset = self.playcount_cluster[playcount_index]
                    playcount_index = self.track_attr_mapper['playcount'] + \
                        playcount_offset
                    icm[playcount_index, track_index] = self.playcount_weight
                    playcount_index += 1
        return csr_matrix(icm)

    def build_icm_2(self, path='./data/tracks_final.csv'):
        """
        returns the item content matrix using mappers defined in dataset class
        icm matrix encoded as follows:
        AxI (A is the number of attributes and I is the number of items)
        """
        icm = lil_matrix((self.attrs_number, self.tracks_number))
        duration_index = 0
        playcount_index = 0
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
                if len(albums) == 0:
                    # set the album with more tracks of that artist
                    # if artist_id in self.album_artist.keys():
                    #     album = self.album_artist[artist_id][np.argmax(self.album_artist_counter[artist_id])]
                    #     album_index = self.track_attr_mapper['album'][album]
                    #     icm[album_index, track_index] = self.album_weight
                    # add the None album of that artist
                    album_index = self.track_attr_mapper['album'][artist_id]
                    icm[album_index, track_index] = self.album_weight
                for album in albums:
                    album_index = self.track_attr_mapper['album'][album]
                    icm[album_index, track_index] = self.album_weight
                # load tags only if specified
                if self.load_tags:
                    # tags
                    tags = parse_csv_array(row['tags'])
                    for tag in tags:
                        if tag in self.track_attr_mapper['tags']:
                            tag_index = self.track_attr_mapper['tags'][tag]
                            tag_weight = self.tag_counter[tag] * \
                                self.tags_weight
                            icm[tag_index, track_index] = tag_weight
                # duration
                duration = row['duration']
                if duration is not None and duration != '' and float(duration) != -1:
                    # duration = float(duration) - self.min_duration
                    # duration_offset = min(
                    #     int(duration / self.duration_int), self.duration_intervals - 1)
                    duration = float(duration)
                    # duration_offset = np.digitize(duration, self.duration_bins) - 1
                    # get the cluster
                    duration_offset = self.duration_cluster[duration_index]
                    duration_index = self.track_attr_mapper['duration'] + \
                        duration_offset
                    icm[duration_index, track_index] = self.duration_weight
                    # go ahead with duration index
                    duration_index += 1
                # playcount
                playcount = row['playcount']
                if playcount is not None and playcount != '' and float(playcount) != -1:
                    # playcount = float(playcount) - self.min_playcount
                    # playcount_offset = min(
                    #     int(playcount / self.playcount_int), self.playcount_intervals - 1)
                    playcount = float(playcount)
                    # playcount_offset = np.digitize(playcount, self.playcount_bins) - 1
                    # get the cluster
                    playcount_offset = self.playcount_cluster[playcount_index]
                    playcount_index = self.track_attr_mapper['playcount'] + \
                        playcount_offset
                    icm[playcount_index, track_index] = self.playcount_weight
                    playcount_index += 1
        return csr_matrix(icm)

    def build_iucm(self, test_dict, path='./data/playlists_final'):
        """
        test_dict is a dict like {pl_id : [tracks]}
        Do not add attributes if track is in test_dict[pl_id]
        Returns an item content matrix derived from user features
        Each items contains the feature of the playlist in which is present
        Rationale:
        two items are similar if the playlist in which they appear are similar:
        example: same user, similar creation time, similar titles.
        """
        iucm = lil_matrix((self.playlist_attrs_number, self.tracks_number))
        # for each playlist get its tracks and add its attributes
        # tracks of a playlist are the indices of urm
        # keep track of the index for cluster of created at
        index = 0
        for pl_id in self.playlists_final.keys():
            # check, some playlists are not in the training set
            if pl_id in self.train_final.keys():
                # get tracks_id of the current playlist
                tracks = self.train_final[pl_id]
                # clean from tracks in test_dict[pl_id]
                tracks = [x for x in tracks if not pl_id in test_dict.keys() or x not in test_dict[pl_id]]
                # get attributes of the playlist
                # get created at
                created_at_offset = self.created_at_cluster[index]
                created_at_index = self.playlist_attr_mapper['created_at'] + \
                    created_at_offset
                # get owner
                owner = self.playlists_final[pl_id]['owner']
                owner_index = self.playlist_attr_mapper['owner'][owner]
                # get title
                title = self.playlists_final[pl_id]['title']
                # array of word of the title
                title_array = parse_csv_array(title)
                title_index_array = [
                    self.playlist_attr_mapper['title'][x] for x in title_array]
                for track in tracks:
                    # get index
                    tr_index = self.get_track_index_from_id(track)
                    iucm[created_at_index, tr_index] += self.created_at_weight
                    iucm[owner_index, tr_index] += self.owner_weight
                    for title_index in title_index_array:
                        iucm[title_index, tr_index] += self.title_weight
        # apply tfidf
        # tfidftransform = TfidfTransformer()
        # icm = tfidftransform.fit_transform(iucm.transpose()).transpose()
        # scale all values
        return iucm

    def build_ucm(self, path='./data/playlists_final'):
        ucm = lil_matrix((self.playlist_attrs_number, self.playlists_number))
        # for each playlist get its tracks and add its attributes
        # tracks of a playlist are the indices of urm
        # keep track of the index for cluster of created at
        index = 0
        for pl_id in self.playlists_final.keys():
            # check, some playlists are not in the training set
            if pl_id in self.train_final.keys():
                # get attributes of the playlist
                # get created at
                created_at_offset = self.created_at_cluster[index]
                created_at_index = self.playlist_attr_mapper['created_at'] + \
                    created_at_offset
                duration_offset = self.playlist_duration_cluster[index]
                duration_index = self.playlist_attr_mapper['duration'] + duration_offset
                numtracks_offset = self.playlist_numtracks_cluster[index]
                numtracks_index = self.playlist_attr_mapper['numtracks'] + numtracks_offset
                # get owner
                owner = self.playlists_final[pl_id]['owner']
                owner_index = self.playlist_attr_mapper['owner'][owner]
                # get title
                title = self.playlists_final[pl_id]['title']
                # array of word of the title
                title_array = parse_csv_array(title)
                title_index_array = [
                    self.playlist_attr_mapper['title'][x] for x in title_array]
                pl_index = self.get_playlist_index_from_id(pl_id)
                ucm[created_at_index, pl_index] = self.created_at_weight
                ucm[duration_index, pl_index] = self.playlist_duration_weight
                ucm[numtracks_index, pl_index] = self.playlist_numtracks_weight
                ucm[owner_index, pl_index] = self.owner_weight
                for title_index in title_index_array:
                    ucm[title_index, pl_index] = self.title_weight
        return csr_matrix(ucm)

    def build_tags_matrix(self, path='./data/tracks_final.csv'):
        """
        Builds a (n_tags, n_tracks) sparse matrix out of the
        tags from the dataset
        """
        tags_matrix = lil_matrix((self.attrs_number, self.tracks_number))
        with open(path, newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter='\t')
            for row in reader:
                tags = parse_csv_array(row['tags'])
                # get index of this track
                track_index = self.track_id_mapper[row['track_id']]
                for tag in tags:
                    if tag in self.track_attr_mapper['tags']:
                        tag_index = self.track_attr_mapper['tags'][tag]
                        tags_matrix[tag_index, track_index] = 1
        return tags_matrix

    def build_artist_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start_artist = np.amin(list(self.track_attr_mapper['artist_id'].values()))
        final_artist = np.amax(list(self.track_attr_mapper['artist_id'].values()))+1
        return icm[start_artist:final_artist, :]

    def build_duration_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = self.track_attr_mapper['duration']
        end = start + self.duration_intervals
        return icm[start:end, :]

    def build_playcount_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = self.track_attr_mapper['playcount']
        end = start + self.playcount_intervals
        return icm[start:end, :]

    def build_album_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start_album = np.amin(list(self.track_attr_mapper['album'].values()))
        final_album = np.amax(list(self.track_attr_mapper['album'].values()))+1
        return icm[start_album:final_album, :]

    def build_tag_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start_tag = np.amin(list(self.track_attr_mapper['tags'].values()))
        end_tag = np.amax(list(self.track_attr_mapper['tags'].values()))+1
        return icm[start_tag:end_tag, :]

    def build_owner_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = np.amin(list(self.playlist_attr_mapper['owner'].values()))
        end = np.amax(list(self.playlist_attr_mapper['owner'].values()))+1
        return icm[start:end, :]

    def build_owner_item_matrix(self, ucm, urm):
        owners = self.build_owner_matrix(ucm).tocsc()
        # for playlist_index in range(urm.shape[0]):
        #     owner_index = owners.indices[owners.indptr[playlist_index]:owners.indptr[playlist_index+1]]
        #     # print("This should be 1!", owner_index.shape)
        #     if owner_index.shape[0]==1:
        #         owner_item_matrix[playlist_index] = owner_item[owner_index[0]]
        # this is a playlist playlist sim
        pl_owner_sim = owners.transpose().dot(owners)
        pl_owner_sim.data = np.ones_like(pl_owner_sim.data)
        owner_item_matrix = pl_owner_sim.dot(urm)
        print("OIM shape:", owner_item_matrix.shape)
        return owner_item_matrix.tocsr()

    def build_title_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = np.amin(list(self.playlist_attr_mapper['title'].values()))
        end = np.amax(list(self.playlist_attr_mapper['title'].values()))+1
        return icm[start:end, :]

    def build_created_at_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = self.playlist_attr_mapper['created_at']
        end = start + self.created_at_intervals
        return icm[start:end, :]

    def build_numtracks_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = self.playlist_attr_mapper['numtracks']
        end = start + self.playlist_numtracks_intervals
        return icm[start:end, :]

    def build_pl_duration_matrix(self, icm):
        """
        Returns the artist part of icm
        """
        start = self.playlist_attr_mapper['duration']
        end = start + self.playlist_duration_intervals
        return icm[start:end, :]

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
        return self.urm.copy()

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

    def add_playlist_to_icm(self, icm, urm, urm_weight):
        # for each track add the playlist attribute
        urm_weighted = urm * urm_weight
        return vstack([icm, urm_weighted], format='csr')

    def add_playlist_attr_to_icm(self, icm, test_dict):
        """
        Adds to the current icm the iucm, that is the
        item user content matrix
        for each tracks add the attribute of playlists in which is present
        """
        iucm = self.build_iucm(test_dict)
        return vstack([icm, iucm])

    def add_tracks_num_rating_to_icm(self, icm, urm):
        """
        returns the icm with the number of ratings for each track
        """
        # row vector with the number of ratings for each track
        num_ratings = urm.sum(axis=0)
        rating_cluster = KMeans(n_clusters=self.tracks_num_rating_cluster_size).fit_predict(np.reshape(num_ratings, (-1, 1)))
        ratings = lil_matrix((self.tracks_num_rating_cluster_size, icm.shape[1]))
        for i in range(rating_cluster.shape[0]):
            ratings[rating_cluster[i], i] = self.track_num_rating_weight
        ratings = ratings.tocsr()
        icm = vstack([icm, ratings], format='csr')

        return icm

    def add_playlist_num_rating_to_icm(self, ucm, urm):
        """
        returns the icm with the number of ratings for each track
        """
        # row vector with the number of ratings for each track
        num_ratings = urm.sum(axis=1)
        rating_cluster = KMeans(n_clusters=self.playlist_num_rating_cluster_size).fit_predict(np.reshape(num_ratings, (-1, 1)))
        ratings = lil_matrix((self.playlist_num_rating_cluster_size, ucm.shape[1]))
        for i in range(len(rating_cluster)):
            ratings[rating_cluster[i], i] = self.playlist_num_rating_weight
        ratings = ratings.tocsr()
        ucm = vstack([ucm, ratings], format='csr')

        return ucm


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


def build_tracks_mappers(path, dataset, load_tags=False, filter_tag=False):
    """
    Build the mappers of tracks
    """
    # attrs is a dict that contains for every attribute its different values
    # used for mappers
    attrs = {'artist_id': set(), 'album': set(), 'tags': set()}
    # used for couting frequency of each tag. key is the tag, value is the frequency
    tag_counter = {}
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
    durations = []  # array of duration for dividing it in bins
    playcounts = []  # the same as before
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
                if tag in tag_counter:
                    tag_counter[tag] += 1
                else:
                    tag_counter[tag] = 1
            track_id_mapper[row['track_id']] = track_index
            track_index_mapper[track_index] = row['track_id']
            # duration
            # duration is -1 if not present
            if row['duration'] is not None and row['duration'] != '':
                duration = float(row['duration'])
                # threshold for max and cleaning
                if duration != -1:  # and duration / 1000 < 700:
                    if duration < min_duration:
                        min_duration = duration
                    if duration > max_duration:
                        max_duration = duration
                    # durations.append(duration)
                    # take the log
                    durations.append(np.log(duration))
            if row['playcount'] is not None and row['playcount'] != '':
                playcount = float(row['playcount'])
                # threshold for max
                playcounts.append(np.log(playcount))
                if playcount < 5000:
                    if playcount < min_playcount:
                        min_playcount = playcount
                    if playcount > max_playcount:
                        max_playcount = playcount
            track_index += 1
    # set tag counter
    dataset.tag_counter = tag_counter
    # is a dictionary of dictionary
    # for each attrbute a dictionary of keys of attribute and their index
    mapper = {'artist_id': {}, 'album': {},
              'tags': {}, 'duration': 0, 'playcount': 0}
    attr_index = 0
    for v in attrs['artist_id']:
        mapper['artist_id'][v] = attr_index
        attr_index += 1
    for v in attrs['album']:
        mapper['album'][v] = attr_index
        attr_index += 1
    # load tags only if specified and only if higher than pop threshold
    if load_tags:
        for v in attrs['tags']:
            if filter_tag:
                if tag_counter[v] > dataset.pop_threshold:
                    mapper['tags'][v] = attr_index
                    attr_index += 1
            else:
                mapper['tags'][v] = attr_index
                attr_index += 1

    # compute ranges
    dataset.duration_int = (max_duration - min_duration) / \
        dataset.duration_intervals
    dataset.playcount_int = (
        max_playcount - min_playcount) / dataset.playcount_intervals
    # set min duration and min playcount
    dataset.min_duration = min_duration
    dataset.min_playcount = min_playcount
    # for dividing in equal ranges
    _, duration_bins = pd.qcut(
        durations, dataset.duration_intervals, retbins=True)
    _, playcount_bins = pd.qcut(
        playcounts, dataset.playcount_intervals, retbins=True)
    dataset.duration_bins = duration_bins
    dataset.playcount_bins = playcount_bins
    # set index of duration and playcount
    mapper['duration'] = attr_index
    mapper['playcount'] = attr_index + dataset.duration_intervals

    attr_index += dataset.duration_intervals + dataset.playcount_intervals + 1

    return track_id_mapper, track_index_mapper, mapper, attr_index


def build_tracks_mappers_clusters(path, dataset, load_tags=False, filter_tag=False):
    """
    Build the mappers of tracks
    """
    # attrs is a dict that contains for every attribute its different values
    # used for mappers
    attrs = {'artist_id': set(), 'album': set(), 'tags': set()}
    # used for couting frequency of each tag. key is tag, value is frequency
    tag_counter = {}
    # mapper from track id to column index. key is track id value is column
    track_id_mapper = {}
    # mapper from index to track id. key is column, value is id
    track_index_mapper = {}
    # this is the number of columns of the matrix of the matrix
    track_index = 0
    # duration and playcount attributes
    durations = []  # array of duration for dividing it in bins
    playcounts = []  # the same as before
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
                if tag in tag_counter:
                    tag_counter[tag] += 1
                else:
                    tag_counter[tag] = 1
            track_id_mapper[row['track_id']] = track_index
            track_index_mapper[track_index] = row['track_id']
            # duration
            # duration is -1 if not present
            if row['duration'] is not None and row['duration'] != '':
                duration = float(row['duration'])
                # threshold for max and cleaning
                if duration != -1:  # and duration / 1000 < 700:
                    durations.append(duration)
            if row['playcount'] is not None and row['playcount'] != '':
                playcount = float(row['playcount'])
                # threshold for max
                playcounts.append(playcount)
            track_index += 1
    # set tag counter
    dataset.tag_counter = tag_counter
    # is a dictionary of dictionary
    # for each attrbute a dictionary of keys of attribute and their index
    mapper = {'artist_id': {}, 'album': {},
              'tags': {}, 'duration': 0, 'playcount': 0}
    attr_index = 0
    for v in attrs['artist_id']:
        mapper['artist_id'][v] = attr_index
        attr_index += 1
    for v in attrs['album']:
        mapper['album'][v] = attr_index
        attr_index += 1
    # load tags only if specified and only if higher than pop threshold
    if load_tags:
        for v in attrs['tags']:
            if filter_tag:
                if tag_counter[v] > dataset.pop_threshold:
                    mapper['tags'][v] = attr_index
                    attr_index += 1
            else:
                mapper['tags'][v] = attr_index
                attr_index += 1

    #  Divide duration and playcount in cluster
    dataset.duration_cluster = KMeans(
        n_clusters=dataset.duration_intervals).fit_predict(np.reshape(durations, (-1, 1)))
    dataset.playcount_cluster = KMeans(
        n_clusters=dataset.playcount_intervals).fit_predict(np.reshape(playcounts, (-1, 1)))
    # set index of duration and playcount
    mapper['duration'] = attr_index
    mapper['playcount'] = attr_index + dataset.duration_intervals

    attr_index += dataset.duration_intervals + dataset.playcount_intervals + 1

    # normalize tag frequency
    max_freq = max([x for x in tag_counter.values()])
    for k in tag_counter:
        tag_counter[k] = tag_counter[k] / max_freq

    return track_id_mapper, track_index_mapper, mapper, attr_index, tag_counter


def build_tracks_mappers_clusters_ext(path, dataset, load_tags=False, filter_tag=False):
    """
    Build the mappers of tracks
    """
    # attrs is a dict that contains for every attribute its different values
    # used for mappers
    attrs = {'artist_id': set(), 'album': set(), 'tags': set()}
    # used for couting frequency of each tag. key is tag, value is frequency
    tag_counter = {}
    # mapper from track id to column index. key is track id value is column
    track_id_mapper = {}
    # mapper from index to track id. key is column, value is id
    track_index_mapper = {}
    # this is the number of columns of the matrix of the matrix
    track_index = 0
    # duration and playcount attributes
    durations = []  # array of duration for dividing it in bins
    playcounts = []  # the same as before

    # used for discriminating between the album of different artists
    # {'artist_id':set()}
    album_art = {}
    album_artist = {}
    album_artist_counter = {}

    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            artist_id = row['artist_id']
            attrs['artist_id'].add(artist_id)
            albums = parse_csv_array(row['album'])
            for album in albums:
                attrs['album'].add(album)
                if artist_id not in album_art.keys():
                    album_art[artist_id] = set()
                    album_artist[artist_id] = []
                    album_artist_counter[artist_id] = []
                album_art[artist_id].add(album)
                if album not in album_artist[artist_id]:
                    album_artist[artist_id].append(album)
                    album_artist_counter[artist_id].append(1)
                else:
                    album_artist_counter[artist_id][indexOf(album_artist[artist_id], album)] += 1 
            tags = parse_csv_array(row['tags'])
            for tag in tags:
                attrs['tags'].add(tag)
                if tag in tag_counter:
                    tag_counter[tag] += 1
                else:
                    tag_counter[tag] = 1
            track_id_mapper[row['track_id']] = track_index
            track_index_mapper[track_index] = row['track_id']
            # duration
            # duration is -1 if not present
            if row['duration'] is not None and row['duration'] != '':
                duration = float(row['duration'])
                # threshold for max and cleaning
                if duration != -1:  # and duration / 1000 < 700:
                    durations.append(duration)
            if row['playcount'] is not None and row['playcount'] != '':
                playcount = float(row['playcount'])
                if playcount != 0.0 and playcount != -1.0:
                    # threshold for max
                    playcounts.append(playcount)
            track_index += 1
    # set tag counter
    dataset.tag_counter = tag_counter
    # is a dictionary of dictionary
    # for each attrbute a dictionary of keys of attribute and their index
    mapper = {'artist_id': {}, 'album': {},
              'tags': {}, 'artist_album': {}, 'duration': 0, 'playcount': 0}
    attr_index = 0
    for v in attrs['artist_id']:
        mapper['artist_id'][v] = attr_index
        attr_index += 1
    for v in attrs['album']:
        mapper['album'][v] = attr_index
        attr_index += 1
    # add artist values to album to represent None album of each artist
    for v in attrs['artist_id']:
        mapper['album'][v] = attr_index
        attr_index += 1
    # load tags only if specified and only if higher than pop threshold
    if load_tags:
        for v in attrs['tags']:
            if filter_tag:
                if tag_counter[v] > dataset.pop_threshold:
                    mapper['tags'][v] = attr_index
                    attr_index += 1
            else:
                mapper['tags'][v] = attr_index
                attr_index += 1
    # for k in album_art.keys():
    #     for album in album_art[k]:
    #         if k not in mapper['artist_album'].keys():
    #             mapper['artist_album'][k] = {}
    #         mapper['artist_album'][k][album] = attr_index
    #         attr_index += 1

    #  Divide duration and playcount in cluster
    dataset.duration_cluster = KMeans(
        n_clusters=dataset.duration_intervals).fit_predict(np.reshape(durations, (-1, 1)))
    dataset.playcount_cluster = KMeans(
        n_clusters=dataset.playcount_intervals).fit_predict(np.reshape(playcounts, (-1, 1)))
    # set index of duration and playcount
    mapper['duration'] = attr_index
    mapper['playcount'] = attr_index + dataset.duration_intervals

    attr_index += dataset.duration_intervals + dataset.playcount_intervals + 1

    # normalize tag frequency
    max_freq = max([x for x in tag_counter.values()])
    for k in tag_counter:
        tag_counter[k] = tag_counter[k] / max_freq

    return track_id_mapper, track_index_mapper, mapper, attr_index, tag_counter, album_artist_counter, album_artist


def build_playlists_mappers(path, dataset):
    """
    Builds the mapper for playlist
    """
    attrs = {'title': set(), 'owner': set(), 'created_at': 0, 'numtracks':0, 'duration':0}
    # mapper from playlist id to row index
    playlist_id_mapper = {}
    # mapper from row index to playlist id
    playlists_index_mapper = {}
    # array of creation time for dividing it into cluster
    created_at = []
    numtracks = []
    duration = []
    playlist_index = 0
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            titles = parse_csv_array(row['title'])
            for title in titles:
                attrs['title'].add(title)
            owner = row['owner']
            attrs['owner'].add(owner)
            created_at.append(row['created_at'])
            numtracks.append(row['numtracks'])
            duration.append(row['duration'])
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
    mapper['created_at'] = attr_index
    mapper['duration'] = attr_index + dataset.created_at_intervals
    mapper['numtracks'] = attr_index + dataset.created_at_intervals + dataset.playlist_duration_intervals
    attr_index += dataset.created_at_intervals + dataset.playlist_duration_intervals + dataset.playlist_numtracks_intervals + 1
    dataset.created_at_cluster = KMeans(
        n_clusters=dataset.created_at_intervals).fit_predict(np.reshape(created_at, (-1, 1)))
    dataset.playlist_numtracks_cluster = KMeans(
        n_clusters=dataset.playlist_numtracks_intervals).fit_predict(np.reshape(numtracks, (-1, 1)))
    dataset.playlist_duration_cluster = KMeans(
        n_clusters=dataset.playlist_duration_intervals).fit_predict(np.reshape(duration, (-1, 1)))
    return playlist_id_mapper, playlists_index_mapper, mapper, attr_index


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


def most_popular_features(icm, topK):
    """
    Returns the row indices of the most topK most popular features.
    By most popular we mean those that appear the highest number
    of items.

    ICM: sparse (n_features, n_items) matrix
    topK: int

    Returns a topK-long array of indices
    """
    # Sum the columns to get the number of items each feature appears in
    popularity = csc_matrix(icm.sum(axis=1))
    # Get the indices of the topK ones
    data = popularity.data[popularity.indptr[0]:popularity.indptr[1]]
    popular_indices = np.argpartition(data,
                                      len(data) - topK)[-topK:]
    return popular_indices


def aggregate_features(icm, n_features, topK):
    """
    Performs feature aggregation as described in [1]

    ICM: sparse matrix - The features matrix with unitary entries.
    N_FEATURES: int - The number of features aggregated in each set.
    TOP_K: int - The number of top features to be aggregated.

    Returns a roughly (top_k ** n_features, icm.shape[1]) sparse matrix

    [1]: Daniele Grattarola et al 2017. Content-Based approaches for Cold-Start
         Job Recommendations
    """
    from itertools import product
    final = None
    topKICM = icm[most_popular_features(icm, topK)]
    # Build the list of tuples of feature indices to aggregate
    features_indices = product(range(topKICM.shape[0]), repeat=n_features)
    print('Aggregating features...')
    for i, t in enumerate(features_indices):
        if i % 1000 == 0:
            print('{:d} sets of features aggregated...'.format(i))
        # The features to aggregate must all distinct
        if len(t) != len(set(t)):
            continue
        # If all distinct, perform an element-wise logical AND
        # of the feature vectors
        aggr = np.ones((1, topKICM.shape[1]))
        for index in t:
            aggr = topKICM.getrow(index).multiply(aggr)
        aggr = csr_matrix(aggr)

        # Stack current aggregation over final matrix
        if final is None:
            final = aggr
        else:
            final = vstack((final, aggr), format='csr')
    return final


def build_aggregated_feature_space(icm, n_features, topK):
    """
    Builds a weighted aggregated icm from input parameter ICM.
    We perform feature aggregation first of the TOPK features in
    ICM, then we stack those aggregated features over the ICM and
    run TF-IDF rescaled to [0, 1].
    """
    from sklearn.feature_extraction.text import TfidfTransformer
    # agg_f_less = aggregate_features(icm, n_features - 1, topK)
    agg_f_full = aggregate_features(icm, n_features, topK)
    extended_icm = vstack((icm, agg_f_full), format='csr')
    print('Computing TF-IDF...')
    transformer = TfidfTransformer(norm='l2', use_idf=True,
                                   smooth_idf=True, sublinear_tf=False)
    tfidf = transformer.fit_transform(extended_icm)
    return tfidf

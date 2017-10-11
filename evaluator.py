import random
import math


class Evaluator(object):

    def __init__(self):
        # for each fold the dictionary of predictions
        self.test_dictionaries = []

        # target tracks is a list of lists, for each fold its target tracks
        self.target_tracks = []

        # min number of tracks for playlists
        self.min_playlist_size = 10

        # starts from -1 since get_fold adds 1 at the beginning
        self.current_fold_index = -1

        # evaluation for each fold
        self.evaluations = []

        self.folds = 0

    def cross_validation(self, folds, train_dataset):
        """
        Method for initializing the cross validation
        folds is the number of folds in which the dataset is divided.
        train dataset is a dict: {"playlist_id": [tracks]}
        This method creates folds dictionaries for testing.
        You have to call get_fold(fold_number, urm)
        to get the urm ready for one iteration of training
        """
        self.folds = folds
        # initialize test dictionary
        for i in range(0, folds):
            self.test_dictionaries.append({})
            self.target_tracks.append(list())
            self.evaluations.append(0)

        # get the size of training set
        training_set_size = sum(
            [1 for x in train_dataset.values()
             if len(x) >= self.min_playlist_size])

        fold_size = math.ceil(training_set_size / folds)

        # get the playlist with at least min_playlist_size elements
        playlists = [x for x in train_dataset.keys() if len(
            train_dataset[x]) >= self.min_playlist_size]

        # shuffle the list of playlists
        random.shuffle(playlists)

        # sublists is a list of lists. contains the folds
        sublists = [playlists[x:x + fold_size]
                    for x in range(0, len(playlists), fold_size)]
        sample_number = 5

        # sublist is a list of playlist, sample 5 tracks for each playlist
        fold_index = 0
        for sublist in sublists:
            current_tg_tracks = set()
            for pl in sublist:
                # sample from the tracks of that playlist in train dataset
                samples = random.sample(train_dataset[pl], sample_number)
                # add tracks to the correct fold
                self.test_dictionaries[fold_index][pl] = samples
                current_tg_tracks = current_tg_tracks.union(samples)
            self.target_tracks[fold_index] = list(current_tg_tracks)
            fold_index += 1

    def get_fold(self, dataset):
        """
        dataset is Dataset object
        Returns:
        the user rating matrix of the current fold, the target tracks
        and target playlist
        """
        self.current_fold_index = self.current_fold_index + 1
        current_fold = dataset.build_train_matrix()
        for k in self.test_dictionaries[self.current_fold_index].keys():
            playlist_index = dataset.get_playlist_index_from_id(k)
            for track in self.test_dictionaries[self.current_fold_index][k]:
                track_index = dataset.get_track_index_from_id(track)
                current_fold[playlist_index, track_index] = 0
        return current_fold, self.target_tracks[self.current_fold_index], self.test_dictionaries[self.current_fold_index].keys()

    def evaluate_fold(self, recommendation):
        """
        recommendation is the dictionary of recommendation {'playlist ': list}
        For each playlist in test_dictionary[current_fold] evaluate MAP
        """
        if self.current_fold_index < self.folds:
            cumulated_ap = 0
            for pl_id in self.test_dictionaries[self.current_fold_index].keys():
                # avg precision
                ap = 0
                item_number = 1
                relevant_items = 0
                for tr_id in recommendation[pl_id]:
                    if tr_id in self.test_dictionaries[self.current_fold_index][pl_id]:
                        relevant_items += 1
                        precision = relevant_items / item_number
                        ap = ap + precision
                    item_number += 1
                # should ap be divided by 5?????
                cumulated_ap = cumulated_ap + (ap / 5)
            map_at_five = cumulated_ap / \
                len(self.test_dictionaries[self.current_fold_index].keys())
            print("MAP@5: " + str(map_at_five))
            self.evaluations[self.current_fold_index] = map_at_five

    def get_mean_map(self):
        """
        Returns the mean map computed over each fold
        """
        return sum(self.evaluations) / self.folds

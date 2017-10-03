import csv


class Dataset():
    """docstring for Dataset"""

    def __init__(self):
        prefix = './data/'
        self.tracks_final = load_csv(prefix + 'tracks_final.csv', 'track_id')
        self.playlists_final = load_csv(prefix + 'playlists_final.csv',
                                        'playlist_id')
        self.train_final = load_csv(prefix + 'train_final.csv', 'playlist_id')
        self.target_playlists = load_csv(prefix + 'target_playlists.csv',
                                         'playlist_id')
        self.target_tracks = load_csv(prefix + 'target_tracks.csv', 'track_id')


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

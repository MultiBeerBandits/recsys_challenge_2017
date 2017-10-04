import csv
from scipy.sparse import lil_matrix


class Dataset():
    """docstring for Dataset"""

    def __init__(self):
        prefix = './data/'
        self.tracks_final = load_csv(prefix + 'tracks_final.csv', 'track_id')
        self.playlists_final = load_csv(prefix + 'playlists_final.csv',
                                        'playlist_id')
        self.target_playlists = load_csv(prefix + 'target_playlists.csv',
                                         'playlist_id')
        self.target_tracks = load_csv(prefix + 'target_tracks.csv', 'track_id')
        self.train_final = load_train_final(prefix + 'train_final.csv')


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


def build_train_final_matrix(dataset):
    rows = max(map(int, dataset.train_final.keys()))
    columns = max([max(map(int, x)) for x in dataset.train_final.values()])
    M = lil_matrix((rows + 1, columns + 1))
    print(M.shape)
    for k, v in dataset.train_final.items():
        for track in v:
            M[k, track] = 1
    return M

from loader import *
from scipy.sparse import *
import csv


H = 10


class Recommendation():
    """docstring for Recommendation"""

    def __init__(self):
        self.track_id = ''
        self.rating = 0.0

    def __cmp__(self, object):
        if object is None:
            return -1
        elif self.rating < object.rating:
            return -2
        elif self.rating == object.rating:
            return 0
        else:
            return 1

    def __lt__(self, object):
        if object is None:
            return False
        else:
            return self.rating < object.rating

    def __eq__(self, object):
        if object is None:
            return False
        else:
            return self.rating == object.rating

    def __str__(self):
        return ('{' + self.track_id + ', ' + self.rating + '}')


def main():
    dataset = Dataset()
    print('Dataset built.')
    M = build_train_final_matrix(dataset)
    print('Train final matrix built.')
    M_csc = csc_matrix(M)
    items_bias = lil_matrix((1, M_csc.shape[1]))

    # Iterate over columns to compute items bias
    for c_id in range(M_csc.shape[1]):
        playlists = M_csc.data[M_csc.indptr[c_id]:M_csc.indptr[c_id + 1]]
        # Items is a vector of ones, so its sum is equal to its size
        cum = playlists.size
        items_bias[0, c_id] = (cum / (cum + H))

    # Normalize M_csc subtracting item bias
    for c_id in range(M_csc.shape[1]):
        M_csc.data[M_csc.indptr[c_id]:
                   M_csc.indptr[c_id + 1]] -= items_bias[0, c_id]

    print('Computed item bias and normalized URM')
    M_csr = M_csc.tocsr()
    users_bias = lil_matrix((M_csr.shape[0], 1))
    # Iterate over rows to compute users bias
    for r_id in range(M_csr.shape[0]):
        tracks = M_csr.data[M_csr.indptr[r_id]:M_csr.indptr[r_id + 1]]
        cum = tracks.size
        users_bias[r_id, 0] = (cum / (cum + H))

    print('Computed user bias')
    # For each playlist in target_playlists
    rec = {}
    for pl_id in dataset.target_playlists.keys():
        # For each track in target_tracks
        rec.setdefault(pl_id, [Recommendation() for x in range(5)])
        for t_id in dataset.target_tracks.keys():
            if t_id not in dataset.train_final[pl_id]:
                r_ui = Recommendation()
                r_ui.rating = items_bias[0, int(
                    t_id)] + users_bias[int(pl_id), 0]
                r_ui.track_id = t_id
                min_rec = min(rec[pl_id])
                if r_ui > min_rec:
                    ind = rec[pl_id].index(min_rec)
                    rec[pl_id][ind] = r_ui

    print('Recommendation done.')
    with open('dict.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for k, v in rec:
            writer.writerow([k, v])


if __name__ == '__main__':
    main()

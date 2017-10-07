from loader import *
from scipy.sparse import *
import numpy as np
import csv

# shrink term
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
        return ('{' + str(self.track_id) + ', ' + str(self.rating) + '}')


def main():
    dataset = Dataset()
    print('Dataset built.')
    # TODO: call get_norm_urm
    norm_urm, users_bias, items_bias = get_norm_urm(dataset)
    # get urm, need later
    urm = dataset.build_train_matrix()
    # For each playlist in target_playlists
    rec = {}
    # estimated ratings
    r_tilde = lil_matrix((norm_urm.shape[0], norm_urm.shape[1]))
    i = 0
    for pl_id in dataset.target_playlists.keys():
        i = i + 1
        # For each track in target_tracks compute estimated rating
        pl_index = dataset.playlist_id_mapper[pl_id]
        #track_indeces = map(dataset.get_track_index_from_id,dataset.target_tracks.keys())
        #r_tilde[pl_index,track_indeces] = items_bias[0,track_indeces] + users_bias[pl_index,0]
        for t_id in dataset.target_tracks.keys():
            track_index = dataset.track_id_mapper[t_id]
            r_tilde[pl_index, track_index] = items_bias[0,
                                                        track_index] + users_bias[pl_index, 0]
        if i % 500 == 0:
            print("Processed: " + str(i))
        # todo
        row_clean = (r_tilde[pl_index].multiply(
            np.ones((1, r_tilde.shape[1])) - urm[pl_index])).tocsc()
        print(row_clean[0].toarray())
        max_indices = (row_clean[0].toarray()).argsort()[::-1][:5]
        # create empty list
        print(max_indices)
        rec[pl_id] = []
        print(row_clean[0, max_indices[0, len(max_indices[0])-1]])
        print(row_clean[0, max_indices[0, 0]])
        for ind in max_indices[0]:
            print(ind)
            print(row_clean[0,ind])
            tr_id = dataset.track_index_mapper[row_clean[0,ind]]
            rec[pl_id].append(tr_id)
        print(rec[pl_id])
    # now r_tilde is ready, multiply it by 1 - urm to get only element not present in each playlist
    # r_tilde_clean = r_tilde .multiply(1 - urm)
    print('Recommendation done.')
    with open('dict.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            'playlist_id', 'track_ids'])
        writer.writeheader()
        for k, v in rec.items():
            tracks_ids = ' '.join([str(x) for x in v])
            writer.writerow([k, tracks_ids])


def get_norm_urm(dataset):
    """
    Return normalized urm, user bias, item bias
    """
    M = dataset.build_train_matrix()
    print('Train final matrix built.')
    M_csc = csc_matrix(M)
    items_bias = lil_matrix((1, M_csc.shape[1]))

    # Iterate over columns to compute items bias
    # for c_id in range(M_csc.shape[1]):
    #    playlists = M_csc.data[M_csc.indptr[c_id]:M_csc.indptr[c_id + 1]]
    # Items is a vector of ones, so its sum is equal to its size
    #    cum = playlists.size
    #    items_bias[0, c_id] = (cum / (cum + H))
    item_sum = M_csc.sum(axis=0)
    items_bias = item_sum / (item_sum + H)
    # Normalize M_csc subtracting item bias
    for c_id in range(M_csc.shape[1]):
        M_csc.data[M_csc.indptr[c_id]:
                   M_csc.indptr[c_id + 1]] -= items_bias[0, c_id]
    print('Computed item bias and normalized URM')
    M_csr = M_csc.tocsr()
    # Iterate over rows to compute users bias
    # for r_id in range(M_csr.shape[0]):
    #    tracks = M_csr.data[M_csr.indptr[r_id]:M_csr.indptr[r_id + 1]]
    #    cum = tracks.size
    #    users_bias[r_id, 0] = (cum / (cum + H))
    user_sum = M_csr.sum(axis=1)
    user_bias = user_sum / (user_sum + H)
    print('Computed user bias')
    normalized_urm = M_csr - user_bias

    return normalized_urm, user_bias, items_bias


if __name__ == '__main__':
    main()

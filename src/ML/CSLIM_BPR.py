from multiprocessing import Pool
from scipy.sparse import *
from sklearn.linear_model import ElasticNet, SGDRegressor
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from src.utils.loader import *
from src.utils.evaluator import *
import random


class BPRCSLIM():
    """
    A BPR optimization algorithm for CSLIM model
    Please refer to:
    [1]: Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, \
         Lars Schmidt-Thieme: BPR: Bayesian Personalized Ranking \
         from Implicit Feedback. UAI 2009
    [2]: X. Ning, G. Karypis: Slim: Sparse linear methods for \
         top-n recommender systems. ICDM 2011.
    """

    def __init__(self, l_pos=1e-5, l_neg=1e-6, l_rate=0.05):

        # Stochastic gradient descent learning rate
        self.l_rate = l_rate

        # Regularization coefficient for relevant items weights
        self.l_pos = l_pos

        # Regularization coefficient for relevant items weights
        self.l_neg = l_neg

        # LearnBPR single iteration length. According to [1] the
        # number of Theta updates should be linear with number
        # of positive feedbacks. iteration_length is the linear
        # coefficient.
        self.iteration_length

        # Weights matrix
        self.Theta = None

        # Estimated ratings matrix
        self.R_hat = None

        # List of target playlists ID
        self.pl_id_list = None

        # List of target tracks ID
        self.tr_id_list = None

    def fit(self, urm, target_items, target_users, dataset):
        # Store target playlists and tracks
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)

        with Pool() as pool:
            tasks = [urm for i in range(os.cpu_count())]
            thetas = pool.map(learnBPR, tasks)

            # Reduce to the mean of the Thetas
            self.Theta = reduce(lambda x, y: x + y,
                                thetas,
                                lil_matrix((urm.shape[1], urm.shape[1])))
            self.Theta = self.Theta.tocsr()
            self.Theta.data /= len(thetas)

    def predict(self, at=5):
        """
        returns a dictionary of
        'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
        """
        recs = {}
        for i in range(0, self.R_hat.shape[0]):
            pl_id = self.pl_id_list[i]
            pl_row = self.R_hat.data[self.R_hat.indptr[i]:
                                     self.R_hat.indptr[i + 1]]
            # get top 5 indeces. argsort, flip and get first at-1 items
            sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
            track_cols = [self.R_hat.indices[self.R_hat.indptr[i] + x]
                          for x in sorted_row_idx]
            tracks_ids = [self.tr_id_list[x] for x in track_cols]
            recs[pl_id] = tracks_ids
        return recs


def randomInitTheta(n_items):
    # generate a random matrix with a given density
    return rand(n_items, n_items, density=0.05)


def create_sampler(urm):
    def gen_sample():
        # generate a random user
        u = int(random.random() * urm.shape[0])
        # sample from the indices of urm[u]
        indices = urm.indices[urm.indptr[u]:urm.indptr[u + 1]]
        idx = int(random.random() * len(indices))
        i = indices[idx]
        j = int(random.random() * urm.shape[1])
        while j in indices:
            j = int(random.random() * urm.shape[1])
        return u, i, j
    return gen_sample


def learnBPR(urm):
    """
    BPR-Optimizes cSLIM model

    urm: sparse-matrix - the user rating matrix
    """
    Theta = randomInitTheta(urm.shape[1]).tocsc()
    converge = False

    # Keep descending the gradient until the MAP stabilizes
    while not converge:
        iterate(urm, Theta)
        # Compute MAP
        # Is it stable?
        # converge = True


def iterate(urm, Theta):
    """
    Performs iteration_length * n_positive_feedbacks iterarions of stochastic
    gradient descent, random sampling with replacement a triple (user, positive
    feedback, negative feedback).

    urm: sparse-matrix - The user rating matrix
    Theta: sparse-matrix - The weights matrix to BPR-Optimize
    """

    # Number of iterations
    n_iter = self.iteration_length * urm.nnz
    # Get (u, i, j) sampler
    draw = create_sampler(urm)

    for n in range(n_iter):
        # sample from Ds
        u, i, j = draw()

        # get user row
        x_u = urm.getrow(u)
        w_i = Theta.getcol(i)
        w_j = Theta.getcol(j)
        x_uij = x_u.multiply(w_i - w_j).sum()

        dsigm = exp(-x_uij) / (1 + exp(x_uij))

        # compute i's regularization update vector
        #   first zero-out the Theta_ii entry
        w_i_reg_update = w_i.tolil(copy=True)
        w_i_reg_update[i] = 0
        w_i_reg_update = w_i_reg_update.tocsc()
        #   then apply regularization for positive feedbacks
        w_i_reg_update = self.l_pos * w_i_reg_update

        # compute j's regularization update vector
        #   first zero-out the Theta_ii entry
        w_j_reg_update = w_j.tolil(copy=True)
        w_j_reg_update[j] = 0
        w_j_reg_update = w_j_reg_update.tocsc()
        #   then apply regularization for positive feedbacks
        w_j_reg_update = self.l_neg * w_j_reg_update

        # Now we compute the derivative term of i's update
        #   first zero-out the x_ui entry
        dx_uij_i = x_u.tolil(copy=True)
        dx_uij_i[i] = 0
        #   then multiply for the sigmoid derivative
        dx_uij_i = dsigm * dx_uij_i.tocsr()

        # Now we compute the derivative term of j's update
        #   first zero-out the x_uj entry
        dx_uij_j = x_u.tolil(copy=True)
        dx_uij_j[j] = 0
        #   then multiply for the sigmoid derivative
        dx_uij_j = -dsigm * dx_uij_j.tocsr()

        # Apply stochastic gradient descent update
        Theta = Theta.tolil()
        Theta[:, i] += self.l_rate * (dx_uij_i + w_i_reg_update)
        Theta[:, j] += self.l_rate * (dx_uij_j + w_j_reg_update)


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    for i in range(0, 5):
        ubf = SLIM()
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        ubf.fit(urm, list(tg_tracks), list(tg_playlist), ds)
        recs = ubf.predict()
        ev.evaluate_fold(recs)
    map_at_five = ev.get_mean_map()
    print("MAP@5 Final", map_at_five)
    # export
    cslim = SLIM()
    urm = ds.build_train_matrix()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    cslim.fit(urm,
              tg_tracks,
              tg_playlist, ds)
    recs = cslim.predict()
    with open('submission_cslim.csv', mode='w', newline='') as out:
        fieldnames = ['playlist_id', 'track_ids']
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for k in tg_playlist:
            track_ids = ''
            for r in recs[k]:
                track_ids = track_ids + r + ' '
            writer.writerow({'playlist_id': k,
                             'track_ids': track_ids[:-1]})

from multiprocessing import Pool
from scipy.sparse import *
import numpy as np
import os
from src.utils.loader import *
from src.utils.evaluator import *
import random
from math import exp


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

    def __init__(self, l_pos=0.0025, l_neg=0.00025, l_rate=0.05, tol=1e-4):

        # Stochastic gradient descent learning rate
        self.l_rate = l_rate

        # Regularization coefficient for relevant items weights
        self.l_pos = l_pos

        # Regularization coefficient for relevant items weights
        self.l_neg = l_neg

        # LearnBPR tolerance on MAP. Stop iterating when MAP < tol
        self.tol = tol

        # LearnBPR single iteration length. According to [1] the
        # number of Theta updates should be linear with number
        # of positive feedbacks. iteration_length is the linear
        # coefficient.
        self.iteration_length = 5

        # Weights matrix
        self.Theta = None

        # Estimated ratings matrix
        self.R_hat = None

        # List of target playlists ID
        self.pl_id_list = None

        # List of target tracks ID
        self.tr_id_list = None

    def fit(self, urm, target_users, target_items, dataset):
        # Store target playlists and tracks
        self.pl_id_list = list(target_users)
        self.tr_id_list = list(target_items)

        # List target playlists row indices
        pl_indices = [dataset.get_playlist_index_from_id(x)
                      for x in self.pl_id_list]
        tr_indices = [dataset.get_track_index_from_id(x)
                      for x in self.tr_id_list]

        n_workers = 4  # os.cpu_count()
        with Pool(n_workers) as pool:
            print('Running {:d} workers...'.format(n_workers))
            tasks = [(urm,
                      pl_indices,
                      tr_indices,
                      self.tol,
                      self.l_pos,
                      self.l_neg,
                      self.l_rate,
                      self.iteration_length,
                      'worker_{:d}'.format(i))
                     for i in range(n_workers)]
            thetas = pool.map(_work, tasks)

            # Reduce to the mean of the Thetas
            for t in thetas:
                if self.Theta is None:
                    self.Theta = t
                else:
                    self.Theta += t
            self.Theta = self.Theta.tocsr()
            self.Theta.data /= len(thetas)

        self.R_hat = compute_R_hat(urm, self.Theta, pl_indices, tr_indices)

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
    return rand(n_items, n_items, density=0.001)


def create_sampler(urm):
    """
    Returns a closure on URM that samples uniformly a user u and
    two items i and j such that:
        i belongs to the set of positive feedbacks from u
        j belongs to the set of missing feedbacks from u
    """
    urm = csr_matrix(urm)

    def gen_sample():
        # generate a random user
        u = int(random.random() * urm.shape[0])
        while len(urm.data[urm.indptr[u]:urm.indptr[u + 1]]) == 0:
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


def _work(params):
    """Pool worker job. Just unpacks parameters and calls learnBPR"""
    urm = params[0]
    target_playlists = params[1]
    target_tracks = params[2]
    tol = params[3]
    l_pos = params[4]
    l_neg = params[5]
    l_rate = params[6]
    iteration_length = params[7]
    worker_id = params[8]

    return learnBPR(urm,
                    target_playlists,
                    target_tracks,
                    tol,
                    l_pos,
                    l_neg,
                    l_rate,
                    iteration_length,
                    worker_id)


def learnBPR(urm, target_playlists,
             target_tracks,
             tol,
             l_pos,
             l_neg,
             l_rate,
             iteration_length,
             worker_id):
    """
    BPR-Optimizes the cSLIM model

    urm: sparse-matrix - the user rating matrix
    """
    print('[ {:d} ] Starting LearnBPR with tol = {:f} ...'.format(
        os.getpid(), tol))

    # Check whether a previous Theta computation is available.
    # In case, start from there.
    if os.path.isfile('./data/cslim_bpr_theta_{}.npz'.format(worker_id)):
        print('[ {:d} ] A previous Theta matrix was found. Loading it...'
              .format(os.getpid()))
        Theta = load_sparse_matrix(
            './data/cslim_bpr_theta_{}.npz'.format(worker_id))
    else:
        # Otherwise initialize a new random one
        Theta = randomInitTheta(urm.shape[1]).tocsc()

    converge = False
    MAP_old = None
    iteration_count = 0

    # Keep descending the gradient until the MAP stabilizes
    while not converge:
        iterate(urm, Theta, l_pos, l_neg, l_rate, iteration_length, worker_id)
        # Compute R_hat
        R_hat = compute_R_hat(urm, Theta, target_playlists, target_tracks)
        # Clean URM from undesired playlists and tracks
        urm_test = urm.tolil()
        urm_test = urm_test[target_playlists].tolil()
        urm_test = urm_test[:, target_tracks].tocsr()
        # Compute MAP
        print('[ {:d} ] Evaluating MAP@5...'.format(os.getpid()))
        MAP_new = evaluate(urm_test, R_hat, target_playlists, at=5)
        print('[ {:d} ] Iteration {:d} scored MAP = {:f}'.format(
            os.getpid(), iteration_count, MAP_new))
        # Is it stable?
        # if MAP_old is None:
        #     MAP_old = MAP_new
        # else:
        #     deltaMAP = MAP_new - MAP_old
        #     if deltaMAP < tol:
        #         converge = True
        #     else:
        #         MAP_old = MAP_new

        # Assume it is stable after only one cycle
        converge = True
        iteration_count += 1
    return Theta


def iterate(urm, Theta, l_pos, l_neg, l_rate, iteration_length, worker_id):
    """
    Performs iteration_length * n_positive_feedbacks iterarions of stochastic
    gradient descent, random sampling with replacement a triple (user, positive
    feedback, negative feedback).

    urm: sparse-matrix - The user rating matrix
    Theta: sparse-matrix - The weights matrix to BPR-Optimize
    """
    n_iter = urm.nnz  # Number of iterations

    draw = create_sampler(urm)  # Get (u, i, j) sampler
    Theta = csc_matrix(Theta)

    for n in range(n_iter):
        if n % 100 == 0:
            print('[ {:d} ] Iteration step {:d} / {:d} ...'.format(
                  os.getpid(), n, n_iter))
        if n % 1000 == 0 and n != 0:
            print('[ {:d} ] Saving Theta on disk...'.format(os.getpid()))
            save_sparse_matrix('./data/cslim_bpr_theta_{}.npz'
                               .format(worker_id))
        # sample from Ds
        u, i, j = draw()

        x_u = urm.getrow(u).transpose()  # get user row
        w_i = Theta.getcol(i)
        w_j = Theta.getcol(j)
        x_uij = x_u.multiply(w_i - w_j).sum()

        dTheta_i = theta_positive_update(i, w_i, x_u, x_uij, l_pos)
        dTheta_j = theta_negative_update(j, w_j, x_u, x_uij, l_neg)

        # Apply stochastic gradient descent update
        dTheta_i.data *= l_rate
        dTheta_j.data *= l_rate
        Theta = Theta.tolil()
        Theta[:, i] = (w_i + dTheta_i)
        Theta[:, j] = (w_j + dTheta_j)
        Theta = Theta.tocsc()


def theta_positive_update(i, theta_i, x_u, x_uij, l_positive):
    # Evaluate sigmod derivative in x_uij
    dsigm = exp(-x_uij) / (1 + exp(-x_uij))

    # compute i's regularization update vector
    #   first zero-out the Theta_ii entry
    theta_i_reg_update = theta_i.copy()
    theta_i_reg_update[i] = 0
    #   then apply regularization for positive feedbacks
    theta_i_reg_update.data *= l_positive

    # Now we compute the derivative term of i's update
    #   first zero-out the x_ui entry
    dx_uij_i = x_u.copy()
    dx_uij_i[i] = 0
    #   then multiply for the sigmoid derivative
    dx_uij_i.data *= dsigm

    return dx_uij_i + theta_i_reg_update


def theta_negative_update(j, theta_j, x_u, x_uij, l_negative):
    # Evaluate sigmod derivative in x_uij
    dsigm = exp(-x_uij) / (1 + exp(-x_uij))

    # compute j's regularization update vector
    #   first zero-out the Theta_ii entry
    theta_j_reg_update = theta_j.copy()
    theta_j_reg_update[j] = 0
    #   then apply regularization for positive feedbacks
    theta_j_reg_update.data *= l_negative

    # Now we compute the derivative term of j's update
    #   first zero-out the x_uj entry
    dx_uij_j = x_u.copy()
    dx_uij_j[j] = 0
    #   then multiply for the sigmoid derivative
    dx_uij_j.data *= -dsigm

    return dx_uij_j + theta_j_reg_update


def compute_R_hat(urm, Theta, target_playlists, target_tracks):
    """
    Computes R_hat by dot product of URM with THETA.
    Then it keeps only the rows in TARGET_PLAYLISTS and
      the columns in TARGET_TRACKS
    The results is a sparse (TARGET_PLAYLISTS, TARGET_TRACKS) matrix
    """

    # Keep only target_playlists rows
    urm = urm[target_playlists]

    Theta = csr_matrix(Theta)
    R_hat = urm.dot(Theta).tolil()

    # Clean R_hat from already rated entries
    R_hat[urm.nonzero()] = 0

    # Keep only target_item columns
    R_hat = R_hat[:, target_tracks]
    return csr_matrix(R_hat)


def evaluate(urm_test, r_hat, target_users, at=5):
    """
    Evaluates the recommendations from r_hat by computing the MAP.

    target_users: list - row indices of the users we have to make
                  predictions for.

    Returns the MAP.
    """
    urm_test = csr_matrix(urm_test)
    cumulative_MAP = 0.0
    num_eval = 0

    for user_i in range(len(target_users)):
        relevant_items = urm_test[user_i].indices

        if len(relevant_items) > 0:
            recommended_items = top_k(r_hat, user_i, at=5)
            num_eval += 1

            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_MAP /= num_eval
    return cumulative_MAP


def top_k(r_hat, user_index, at=5):
    """
    Returns the column indices of R_HAT of the top-AT ratings
    for USER_INDEX
    """
    r_hat = csr_matrix(r_hat)
    user_row = r_hat.data[r_hat.indptr[user_index]:
                          r_hat.indptr[user_index + 1]]
    top_k_indices = np.argpartition(user_row, user_row.shape[0] - at)[-at:]
    return top_k_indices


def MAP(recommended_items, relevant_items):
    """
    Computes the AP for a single user.

    recommended_items: ndarray - column indices of the items recommended
    relevant_items: ndarray - column indices of the relevant items
                    for the user

    Returns the AP
    """
    is_relevant = np.in1d(recommended_items,
                          relevant_items,
                          assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(
        is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min(
        [relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


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


if __name__ == '__main__':
    ds = Dataset(load_tags=True, filter_tag=True)
    # export
    bprcslim = BPRCSLIM()
    urm = ds.build_train_matrix()
    # get icm
    ds.set_track_attr_weights(art_w=1,
                              alb_w=1,
                              dur_w=0.2,
                              playcount_w=0.2,
                              tags_w=0.2)
    icm = ds.build_icm() * np.sqrt(2.598883982624128)
    M = vstack([urm, icm]).tocsr()
    tg_playlist = list(ds.target_playlists.keys())
    tg_tracks = list(ds.target_tracks.keys())
    # Train the model with the best shrinkage found in cross-validation
    bprcslim.fit(M, tg_playlist, tg_tracks, ds)
    recs = bprcslim.predict()
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

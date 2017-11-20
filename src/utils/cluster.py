import numpy as np
from scipy.sparse import *
from sklearn.cluster import KMeans

def build_user_cluster(urm, icm, ucm, k):
    # k is the number of clusters

    # first build the user feature matrix: UxF (features of rated items)
    ufm = urm.dot(icm.transpose())

    # normalize dividing by rating of each user
    Nu = urm.sum(axis=1)
    Iu = np.copy(Nu)
    # save from divide by zero!
    Iu[Iu == 0] = 1
    # since we have to divide the upm get the reciprocal of this vector
    Iu = np.reciprocal(Iu)
    # multiply the ufm by Iu. Normalize UFM
    ufm = ufm.multiply(Iu)

    # add a column for the number of rating
    ufm = hstack([ufm, Nu], format='csr')

    # stack all matrices horizontally
    ucm = hstack([urm, ufm, ucm.transpose()], format='csr')

    k_m = KMeans(n_clusters=k)
    user_clusters = k_m.fit_predict(ucm)

    return user_clusters, k_m

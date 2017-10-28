# Preprocessing and feature extraction

from sklearn.feature_selection import *
from scipy.sparse import *
import numpy as np
import random


def get_icm_weighted_chi2(urm, icm, u_samples=1000, i_samples=1000):
    """
    Recall that the chi-square test measures
    dependence between stochastic variable
    so using this function “weeds out” the features that
    are the most likely to be independent of class
    and therefore irrelevant for classification.
    - Compute chi squared for each user
    - Average them
    - Apply weights derived from chi squared to each feature
    """
    # binarize icm
    icm_bin = icm.copy()
    icm_bin[icm.nonzero()] = 1
    # feature matrix. UxF
    f_mat = lil_matrix((urm.shape[0], icm_bin.shape[0]))
    # transpose icm
    X = icm_bin.transpose()
    urm = csr_matrix(urm)
    # start compute chi2
    for i in range(u_samples):
        u = random.randint(0, urm.shape[0] - 1)
        # rated items
        r_items = urm.indices[urm.indptr[u]:urm.indptr[u + 1]]
        items = list(range(0, urm.shape[1]))
        selected = np.random.choice(items, size=i_samples)
        item_list = np.append(r_items, selected)
        y = urm[u, item_list]
        X_view = X[item_list]
        y = np.reshape(y.todense(), (-1, 1))
        feat_weights, _ = chi2(X_view, y)
        f_mat[u] = csr_matrix(feat_weights)
        if i % 100 == 0:
            print("User: ", i)
    # average all feature weights
    f_mean = f_mat.mean(axis=0)
    # normalize dividing each feature by the max
    max = np.amax(f_mean)
    f_weights = np.multiply(f_mean, 1 / max)
    # multiply binarized icm by the weights, we obtain a new icm
    # the importance of a weight is how much it depends on 1 or 0 in URM
    icm_w = icm_bin.multiply(f_weights.transpose())
    return icm_w

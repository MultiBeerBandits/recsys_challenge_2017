from skopt import forest_minimize
from skopt.space import Integer
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.Ensemble.ensemble import Ensemble
from src.Ensemble.simEnsemble import SimEnsemble
from src.CBF.CBF_tfidf import ContentBasedFiltering
from src.UBF.UBF2 import UserBasedFiltering
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.Pop.popularity import Popularity
from src.MF.MF_BPR.MF_BPR_CBF import MF_BPR_CBF
from src.MF.MF_BPR.MF_BPR_KNN import MF_BPR_KNN
from src.IBF.IBF import ItemBasedFiltering
from src.CBF.CBF_MF import ContentBasedFiltering as CBF_AUG
from src.ML.BPRSLIM_ext import BPRSLIM
# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='ensemble.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)

ensemble = None
sim_ensemble = SimEnsemble()
ev = Evaluator()


def create_BPR_SLIM(urm, dataset):
    icm = dataset.build_icm()
    cslim_train = vstack((urm * 0.4, icm), format="csr")
    logFile = open("SLIM_BPR_Cython.txt", "a")

    bpr = SLIM_BPR_Cython(URM_train = cslim_train.tocsr(),
                          ev=ev,
                          recompile_cython=False,
                          positive_threshold=1,
                          sparse_weights=True,
                          epochs=1000,
                          validate_every_N_epochs=10,
                          logFile=logFile,
                          batch_size=1,
                          epochMultiplier=0.2,
                          sgd_mode='adagrad',
                          learning_rate=5e-9,
                          lambda_i=1e-3,
                          lambda_j=1e-5,
                          topK=300)
    return bpr



def objective(params):
    global ensemble, ev
    print("Current params", str(params))

    recs = ensemble.predict(params)
    map_at_five = ev.evaluate_fold(recs)
    # Make negative because we want to _minimize_ objective
    out = -map_at_five

    return out

def int_objective(params):
    global ensemble, ev
    print("Current params", str(params))

    recs = ensemble.predict_interleaved(params)
    map_at_five = ev.evaluate_fold(recs)
    # Make negative because we want to _minimize_ objective
    out = -map_at_five

    return out


def sim_objective(params):
    global sim_ensemble, ev
    print("Current params", str(params))

    recs = sim_ensemble.predict(params)
    map_at_five = ev.evaluate_fold(recs)

    # Make negative because we want to _minimize_ objective
    out = -map_at_five

    return out

def result(res):
    logging.info("MAP " + str(-res.fun))
    logging.info(str(res.x))

def linear_ensemble():
    global ensemble, ev
    space = [# Integer(0, 1000),  # MF_BPR_CBF
             # Integer(0, 1000),  # MF_BPR_KNN
             Integer(0, 1000),  # CBF
             Integer(0, 1000),  # UBF
             Integer(0, 1000),  # IBF
             # Integer(0, 1000),  # BPR_CSLIM
             Integer(0, 100)  # Pop
             ]
    # x0 = [1, 0, 0, 0, 0, 0]
    # x1 = [0, 1, 0, 0, 0, 0]
    # x2 = [0, 0, 1, 0, 0, 0]
    # x3 = [0, 0, 0, 1, 0, 0]
    # x4 = [0, 0, 0, 0, 1, 0]
    # x5 = [0, 0, 0, 0, 0, 1]

    # initial values are the single recommenders
    # x0 = [1000, 0, 0, 0, 0, 0]
    # x1 = [0, 1000, 0, 0, 0, 0]
    # x2 = [0, 0, 1000, 0, 0, 0]
    # x3 = [0, 0, 0, 1000, 0, 0]
    # x4 = [0, 0, 0, 0, 1000, 0]
    # x5 = [0, 0, 0, 0, 0, 1000]

    x0 = [1000, 0, 0, 0]
    x1 = [0, 1000, 0, 0]
    x2 = [0, 0, 1000, 0]
    x3 = [0, 0, 0, 100]
    #x6 = [0, 0, 0, 0, 0, 0, 1]

    x0s = [x0, x1, x2, x3]
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=False, weight_tag=False)
    ds.set_track_attr_weights_2(1.0, 1.0, 0.0, 0.0, 0.0,
                                1.0, 1.0, 0.0, 0.0)
    ds.set_playlist_attr_weights(1, 1, 1, 1, 1)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)

    # augment r_hat
    # cbf_aug = CBF_AUG()
    # cbf_aug.fit(urm, tg_playlist,
    #             tg_tracks,
    #             ds)

    # get R_hat
    #r_hat_aug = cbf_aug.getR_hat()

    # create all the models
    cbf = ContentBasedFiltering()
    ubf = UserBasedFiltering()
    ibf = ItemBasedFiltering()
    pop = Popularity(20)
    # mf_bpr = MF_BPR_CBF(r_hat_aug)
    # mf_bpr_knn = MF_BPR_KNN(r_hat_aug)
    # bpr = BPRSLIM(epochs=40,
    #               epochMultiplier=1,
    #               sgd_mode='rmsprop',
    #               learning_rate=5e-08,
    #               topK=300,
    #               urmSamplingChances=2 / 5,
    #               icmSamplingChances=3 / 5,
    #               urm_ext=R_hat)

    # add models to list of models
    #models = [mf_bpr, mf_bpr_knn, cbf, ubf, ibf, pop]
    models = [cbf, ubf, ibf, pop]

    # create the ensemble
    ensemble = Ensemble(models, normalize_ratings=True)

    # call fit on ensemble to fit all models
    ensemble.fit(urm, list(tg_tracks), list(tg_playlist), ds)

    # start optimization
    res = forest_minimize(objective,
                          space,
                          x0=x0s,
                          verbose=True,
                          n_random_starts=20,
                          n_calls=10000,
                          n_jobs=-1,
                          callback=result)

    # print optimal params
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['CBF', 'UBF', 'IBF', 'POP']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))


def opt_sim_ensemble():
    global sim_ensemble, ev
    space = [Real(0.0, 1.0),  # CBF
             Real(0.0, 1.0),  # IBF
             Real(0.0, 1.0),  # CSLIM
             ]
    x0 = [1, 0, 0]
    x1 = [0, 1, 0]
    x2 = [0, 0, 1]
    x0s = [x0, x1, x2]
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.5, 0.5, 0.5, 0.05, 0.05)
    ev = Evaluator()
    ev.cross_validation(4, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    sim_ensemble.fit(urm, list(tg_playlist), list(tg_tracks), ds)
    res = forest_minimize(sim_objective, space, x0=x0s, verbose=True, n_random_starts=20, n_calls=200, n_jobs=-1, callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['CBF', 'IBF', 'CSLIM']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))

def cluster_ensemble():
    """
    Ensemble in which we have different weights for each cluster of users
    """
    n_cluster = 20
    space = []
    for i in range(n_cluster):
        space.append((0.0, 1.0))

    res = forest_minimize(fit_cluster, space, verbose=True, n_calls=1000, n_jobs=-1, callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['1', '2', '3']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))

if __name__ == '__main__':
    linear_ensemble()

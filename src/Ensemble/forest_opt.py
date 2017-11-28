from skopt import forest_minimize
from skopt.space import Real
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.Ensemble.ensemble import Ensemble
from src.Ensemble.simEnsemble import SimEnsemble
from src.FWUM.UICF import xSquared
from src.CBF.CBF import ContentBasedFiltering
from src.UBF.UBF import UserBasedFiltering
from src.MF.iALS import IALS
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.Pop.popularity import Popularity
from src.ML.CSLIM_parallel import SLIM
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
    space = [Real(0.0, 1.0),  # XBF
             Real(0.0, 1.0),  # CBF
             Real(0.0, 1.0),  # UBF
             # Real(0.0, 1.0),  # IALS
             Real(0.0, 1.0),  # CSLIM
             Real(0.0, 1.0),  # Pop
             ]
    # x0 = [1, 0, 0, 0, 0, 0]
    # x1 = [0, 1, 0, 0, 0, 0]
    # x2 = [0, 0, 1, 0, 0, 0]
    # x3 = [0, 0, 0, 1, 0, 0]
    # x4 = [0, 0, 0, 0, 1, 0]
    # x5 = [0, 0, 0, 0, 0, 1]

    x0 = [1, 0, 0, 0, 0]
    x1 = [0, 1, 0, 0, 0]
    x2 = [0, 0, 1, 0, 0]
    x3 = [0, 0, 0, 1, 0]
    x4 = [0, 0, 0, 0, 1]

    x0s = [x0, x1, x2, x3, x4]
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.5, 0.5, 0.5, 0.05, 0.05)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)

    # create all the models
    cbf = ContentBasedFiltering()
    xbf = xSquared()
    # ials = IALS(500, 50, 1e-4, 800)
    ubf = UserBasedFiltering()
    # bpr_cslim = create_BPR_SLIM(urm, ds)
    pop = Popularity()
    cslim = SLIM()

    # add models to list of models
    models = [cslim, xbf, cbf, ubf, pop]

    # create the ensemble
    ensemble = Ensemble(models)

    # call fit on ensemble to fit all models
    ensemble.fit(urm, list(tg_tracks), list(tg_playlist), ds)

    # start optimization
    res = forest_minimize(objective,
                          space,
                          x0=x0s,
                          verbose=True,
                          n_random_starts=20,
                          n_calls=300,
                          n_jobs=-1,
                          callback=result)

    # print optimal params
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['CSLIM', 'XBF', 'CBF', 'UBF', 'POP']
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
    opt_sim_ensemble()

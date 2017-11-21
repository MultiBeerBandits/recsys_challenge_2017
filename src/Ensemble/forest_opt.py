from skopt import forest_minimize
from skopt.space import Real
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.Ensemble.ensemble import Ensemble
from src.Ensemble.simEnsemble import SimEnsemble
# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='ensemble.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)

ensemble = Ensemble()
sim_ensemble = SimEnsemble()
ev = Evaluator()

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
             Real(0.0, 1.0)  # IALS
             ]
    x0 = [1, 0, 0, 0]
    x1 = [0, 1, 0, 0]
    x2 = [0, 0, 1, 0]
    x3 = [0, 0, 0, 1]
    x0s = [x0, x1, x2, x3]
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 0.9, 0.2, 0.2, 0.2)
    ds.set_playlist_attr_weights(0.5, 0.5, 0.5, 0.05, 0.05)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    ensemble.fit(urm, list(tg_tracks), list(tg_playlist), ds)
    res = forest_minimize(objective, space, x0=x0s, verbose=True, n_random_starts=20, n_calls=200, n_jobs=-1, callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['XBF', 'CBF', 'UBF', 'IALS']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))

def opt_sim_ensemble():
    global sim_ensemble, ev
    space = [Real(0.0, 1.0),  # CBF
             Real(0.0, 1.0),  # IBF
             Real(0.0, 1.0),  # IALS
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
    params = ['XBF', 'CBF', 'UBF', 'IALS']
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

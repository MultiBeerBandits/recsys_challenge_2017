from skopt import forest_minimize
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.Ensemble.ensemble import fit, fit_cluster
# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='ensemble.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def objective(params):
    # unpack
    alfa, l1, l2 = params

    print("Current params", alfa, l1, l2)

    if alfa > 0 and l1 > 0 and l2 > 0:
        # create all and evaluate
        ds = Dataset()
        ev = Evaluator()
        ev.cross_validation(5, ds.train_final.copy())
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cslim = SLIM(l1, l2, alfa)
        cslim.fit(urm, tg_tracks, tg_playlist, ds)
        recs = cslim.predict()
        map_at_five = ev.evaluate_fold(recs)
        # Make negative because we want to _minimize_ objective
        out = -map_at_five

        return out
    else:
        return 1000

def result(res):
    logging.info("MAP " + str(-res.fun))
    logging.info(str(res.x))

def linear_ensemble():
    space = [(0.0, 1.0),  # XBF
         (0.0, 1.0),  # CBF
         (0.0, 1.0),  # UBF
         (0.0,1.0)  # IALS
        ]

    x0 = [0.6, 0.7, 0.2, 0.6]
    res = forest_minimize(fit, space, x0=x0, verbose=True, n_calls=50, n_jobs=-1, callback=result)
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
    linear_ensemble()

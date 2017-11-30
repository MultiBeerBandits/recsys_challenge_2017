from skopt import forest_minimize, gbrt_minimize
from skopt.space import Real, Integer
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.MF.MF_BPR.MF_BPR import MF_BPR
# Logging stuff
import logging

# Params
# MAP@5: 0.0401673829450349 Current params [2, 0.01, 0.001, 0.001, 0.005]

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='MF_BPR.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def objective(params):
    print("Current params", str(params))
    components, user_reg, pos_item_reg, neg_item_reg, l_rate = params
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    mf = MF_BPR()
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    best_map = 0.0
    best_epoch = 0
    for epoch in range(5):
        mf.fit(urm, ds, list(tg_playlist), list(tg_tracks), no_components=components * 100, l_rate=l_rate, user_reg=user_reg, pos_item_reg=pos_item_reg, neg_item_reg=neg_item_reg)
        recs = mf.predict_dot()
        map_new = ev.evaluate_fold(recs)
        if map_new > best_map:
            best_map = map_new
            best_epoch = epoch

    print ("Best epoch")

    # Make negative because we want to _minimize_ objective
    out = -best_map

    return out


def result(res):
    logging.info("MAP " + str(-res.fun))
    logging.info(str(res.x))


def opt_mf():
    # Parameter to optimize:
    # components, user_reg, item_reg, l_rate
    space = [Integer(1, 10),   # components
             Real(1e-9, 0.8),  # user_reg
             Real(1e-9, 0.8),  # pos_item_reg
             Real(1e-9, 0.8),  # neg_item_reg
             Real(1e-5, 1),    # l_rate
             ]
    x0 = [2, 1e-2, 1e-3, 1e-3, 5e-3]
    x1 = [3, 1e-2, 1e-3, 1e-2, 5e-3]
    x2 = [4, 1e-2, 1e-3, 1e-2, 5e-1]
    x3 = [4, 1e-2, 1e-2, 5e-3, 1e-1]
    x4 = [5, 1e-2, 1e-3, 1e-3, 5e-2]
    x0s = [x0, x1, x2, x3, x4]
    # get the current fold
    res = gbrt_minimize(objective, space, x0=x0s, verbose=True, n_random_starts=20, n_calls=1000, xi=0.01,n_jobs=-1, callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['components', 'user_reg', 'CSLIM']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))


if __name__ == '__main__':
    opt_mf()

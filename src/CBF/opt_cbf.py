# Optimize Content Based Filtering by weight selection

from skopt import forest_minimize
from skopt.space import Integer
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF_augmented import ContentBasedFiltering
# Logging stuff
import logging

"""BEST PARAMS FOUND:
0.1031783086017616
[1.0, 0.9
 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.0, 0.050000000000000003, 0.050000000000000003, 0.050000000000000003, 0.050000000000000003]
"""

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='cbf_param.log',
                    format='%(asctime)s %(message)s',

                    filemode='w',
                    level=logging.DEBUG)


def objective(params):
    artist, album, duration_tracks, playcount, tags, n_rating, created_at, owner, title, duration_playlist, alfa = params

    print("Current params", str(params))
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(artist / 100,
                              album / 100, duration_tracks / 100,
                              playcount / 100,
                              tags / 100,
                              n_rating / 100)
    ds.set_playlist_attr_weights(created_at / 100, owner / 100, title / 100, duration_playlist / 100)
    ev = Evaluator()
    ev.cross_validation(4, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    cbf = ContentBasedFiltering(alfa / 100)
    cbf.fit(urm, tg_playlist, tg_tracks, ds)
    recs = cbf.predict()
    map_at_five = ev.evaluate_fold(recs)
    # Make negative because we want to _minimize_ objective
    out = -map_at_five

    return out


def result(res):
    logging.info("MAP " + str(-res.fun))
    logging.info(str(res.x))


def opt_content_based():
    space = [Integer(0, 100),  # artist
             Integer(0, 100),  # album
             Integer(0, 100),  # duration_tracks
             Integer(0, 100),  # playcount
             Integer(0, 100),  # tags
             Integer(0, 100),  # n_rating
             Integer(0, 100),  # created_at
             Integer(0, 100),  # owner
             Integer(0, 100),  # title
             Integer(0, 100),  # duration_playlist
             Integer(0, 100)   # mix parameter, value of S_cbf
             ]

    x0 = [100, 90, 20, 20, 20, 0, 50, 50, 50, 50, 97]

    res = forest_minimize(objective, space, x0=x0, verbose=True,
                          n_random_starts=20, n_calls=1000, n_jobs=-1,
                          callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['CBF', 'IBF', 'CSLIM']
    for (p, x_) in zip(params, res.x):
      print('{}: {}'.format(p, x_))


if __name__ == '__main__':
    opt_content_based()

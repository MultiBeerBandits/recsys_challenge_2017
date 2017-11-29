# Optimize Content Based Filtering by weight selection

from skopt import forest_minimize, gbrt_minimize
from skopt.space import Integer
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF import ContentBasedFiltering
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

ds = Dataset(load_tags=True, filter_tag=True)

def objective(params):

    global ds

    # unpack params
    artist, album, inferred_album, duration_tracks, inferred_duration, playcount, inferred_playcount, tags, n_rating, urm_weight = params

    print("Current params", str(params))

    # set weights
    ds.set_track_attr_weights_2(artist / 100,
                                album / 100, duration_tracks / 100,
                                playcount / 100,
                                tags / 100,
                                n_rating / 100,
                                inferred_album / 100,
                                inferred_duration / 100,
                                inferred_playcount / 100)

    ev = Evaluator()
    ev.cross_validation(3, ds.train_final.copy())
    for i in range(3):
        urm, tg_tracks, tg_playlist = ev.get_fold(ds)
        cbf = ContentBasedFiltering()
        cbf.fit(urm, tg_playlist, tg_tracks, ds, urm_weight=urm_weight/100)
        recs = cbf.predict()
        map_at_five = ev.evaluate_fold(recs)
    # Make negative because we want to _minimize_ objective
    out = -ev.get_mean_map()

    return out


def result(res):
    logging.info("MAP " + str(-res.fun))
    logging.info(str(res.x))


def opt_content_based():
    space = [Integer(0, 100),  # artist
             Integer(0, 100),  # album
             Integer(0, 100),  # inferred album
             Integer(0, 100),  # duration_tracks
             Integer(0, 100),  # inferred duration
             Integer(0, 100),  # playcount
             Integer(0, 100),  # inferred playcount
             Integer(0, 100),  # tags
             Integer(0, 100),  # n_rating
             Integer(0, 100),  # urm weight
             ]

    x0 = [100, 90, 80, 20, 10, 20, 10, 10, 10, 50]

    res = gbrt_minimize(objective, space, x0=x0, verbose=True,
                        n_random_starts=20, n_calls=10000, n_jobs=-1,
                        callback=result)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['Artist', 'Album', 'inferred_album', 'duration_tracks', 'inferred_duration', 'playcount', 'inferred_playcount', 'tags', 'n_rating', 'urm_weight']
    for (p, x_) in zip(params, res.x):
      print('{}: {}'.format(p, x_))


if __name__ == '__main__':
    opt_content_based()

# Optimize Content Based Filtering by weight selection

from skopt import forest_minimize
from skopt.space import Real
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
import numpy as np
from src.CBF.CBF_augmented import ContentBasedFiltering
# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='ensemble.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def objective(params):
    params = np.array(params)
    params[params < 0] = 0
    params = list(params)
    artist, album, duration_tracks, playcount, tags, n_rating, created_at, owner, title, duration_playlist = params

    print("Current params", str(params))
    # get the current fold
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(artist,
                              album, duration_tracks,
                              playcount,
                              tags,
                              n_rating)
    ds.set_playlist_attr_weights(created_at, owner, title, duration_playlist)
    ev = Evaluator()
    ev.cross_validation(4, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    cbf = ContentBasedFiltering()
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
    space = [Real(-0.1, 1.0),  # artist
             Real(-0.1, 1.0),  # album
             Real(-0.1, 1.0),  # duration_tracks
             Real(-0.1, 1.0),  # playcount
             Real(-0.1, 1.0),  # tags
             Real(-0.1, 1.0),  # n_rating
             Real(-0.1, 1.0),  # created_at
             Real(-0.1, 1.0),  # owner
             Real(-0.1, 1.0),  # title
             Real(-0.1, 1.0),  # duration_playlist
             ]

    x0 = [1, 0.9, 0.2, 0.2, 0.2, 0, 0.5, 0.5, 0.5, 0.05]

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

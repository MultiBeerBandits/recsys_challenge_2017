from skopt import forest_minimize
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *
from src.ML.WARP import *
# Logging stuff
import logging

# write log to file, filemode = 'w' tells to write each time a new file
logging.basicConfig(filename='warp.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.DEBUG)


def objective(params):
    # unpack parameters
    epochs, l_rate, no_components, item_alpha = params
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    warp = WARP()
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    warp.fit(urm, ds, list(tg_playlist), list(tg_tracks), l_rate=l_rate,
             no_components=no_components, item_alpha=item_alpha, n_epochs=epochs)
    recs = warp.predict()
    map_at_five = ev.evaluate_fold(recs)
    logging.info("MAP@5 Final" + str(map_at_five) + "Params:" +
          str(epochs) + " " + str(l_rate) + " " + str(no_components) + " " + str(item_alpha))
    # return negative since we want to maximize it and forest wants to minimize
    return -map_at_five


def learn_hyperparams():
    # define the space
    space = [(70, 300),  # epochs
             (10**-3, 1.0, 'log-uniform'),  # learning_rate
             (100, 500),  # no_components
             (10**-5, 10**-3, 'log-uniform'),  # item_alpha
             ]
    best_result = forest_minimize(objective, space, n_calls=100,
                                  random_state=0,
                                  verbose=True)
    print('Maximimum p@k found: {:6.5f}'.format(-best_result.fun))
    print('Optimal parameters:')
    params = ['epochs', 'learning_rate', 'no_components', 'item_alpha']
    for (p, x_) in zip(params, best_result.x):
        print('{}: {}'.format(p, x_))


if __name__ == '__main__':
    learn_hyperparams()

from skopt import forest_minimize
from scipy.sparse import *
from src.ML.CSLIM_parallel import *
from src.utils.loader import *
from src.utils.evaluator import *


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


def main():
    space = [(1, 5), # alfa
         (1e-7, 1e-3, 'log-uniform'), # l1
         (1e-6, 1e-2), # l2
        ]
    # best individual
    # 2.598883982624128, 1e-05, 3.8223372852050046e-05
    x0 = [2.59, 1e-05, 3.822*1e-05]
    res = forest_minimize(objective, space, verbose=True, x0=x0, n_calls=10, y0=-0.1126)
    print('Maximimum p@k found: {:6.5f}'.format(-res.fun))
    print('Optimal parameters:')
    params = ['alfa', 'l1', 'l2']
    for (p, x_) in zip(params, res.x):
        print('{}: {}'.format(p, x_))


if __name__ == '__main__':
    main()

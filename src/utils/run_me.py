from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.utils.loader import *
from src.utils.evaluator import *
import numpy as np
import scipy.sparse as sps

def run_SLIM():

    # ds = Dataset()
    # ev = Evaluator()
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())

    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    test_urm = ev.get_test_matrix(0, ds)

    icm = ds.build_icm()
    cslim_train = sps.vstack((urm * 0.4, icm), format="csr")

    recommender = SLIM_BPR_Cython(cslim_train.tocsr(),
                                  ev=ev,
                                  dataset=ds,
                                  tg_tracks=tg_tracks,
                                  tg_playlist=tg_playlist,
                                  recompile_cython=False,
                                  positive_threshold=1,
                                  sparse_weights=True)

    logFile = open("SLIM_BPR_Cython.txt", "a")
    recommender.fit(epochs=1000,
                    validate_every_N_epochs=10,
                    URM_test=test_urm.tocsr(),
                    logFile=logFile,
                    batch_size=1,
                    epochMultiplier=0.2,
                    sgd_mode='adagrad',
                    learning_rate=5e-9,
                    lambda_i=1e-3,
                    lambda_j=1e-5,
                    topK=300)
        # ev.print_worst(ds)



    #results_run = recommender.evaluateRecommendations(URM_test, at=5)
    #print(results_run)


run_SLIM()
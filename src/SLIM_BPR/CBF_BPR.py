
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.utils.loader import *
from src.utils.evaluator import *
from src.CBF.CBF_tfidf import *
import numpy as np

"""
first run cslim to get initial sparse S matrix, then pass it to BPR
"""

def run():

    ds = Dataset()
    ev = Evaluator()
    ds = Dataset(load_tags=True, filter_tag=True)
    ds.set_track_attr_weights(1, 1, 0.2, 0.2, 0.2)
    ev = Evaluator()
    ev.cross_validation(5, ds.train_final.copy())
    urm, tg_tracks, tg_playlist = ev.get_fold(ds)
    test_urm = ev.get_test_matrix(0, ds)
    test_dict = ev.get_test_dict(0)
    # run cslim
    cbf = ContentBasedFiltering()
    cbf.fit(urm, tg_playlist, tg_tracks, ds, test_dict=test_dict)
    rec = cbf.predict()
    ev.evaluate_fold(rec)
    # get the similarity matrix
    S = cbf.getW().tocsr()
    # pass it to the bpr algorithm
    recommender = SLIM_BPR_Cython(urm.tocsr(), tg_playlist=tg_playlist, tg_tracks=tg_tracks, ev=ev, dataset=ds, recompile_cython=True, positive_threshold=1, sparse_weights=True)
    logFile = open("SLIM_BPR_Cython.txt", "a")
    recommender.fit(S=S, epochs=1000, validate_every_N_epochs=1, URM_test=test_urm.tocsr(),
                logFile=logFile, batch_size=1, sgd_mode='adagrad', learning_rate=5e-4, lambda_i=1e-4, lambda_j=1e-4, topK=500)
        # ev.print_worst(ds)



    #results_run = recommender.evaluateRecommendations(URM_test, at=5)
    #print(results_run)


run()
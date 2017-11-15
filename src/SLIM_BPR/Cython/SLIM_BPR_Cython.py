#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from src.SLIM_BPR.Recommender_utils import similarityMatrixTopK
from src.SLIM_BPR.SLIM_BPR_Python import SLIM_BPR_Python
import subprocess
import os
import sys
import numpy as np


class SLIM_BPR_Cython(SLIM_BPR_Python):

  def __init__(self, URM_train, ev=None, tg_tracks=None, dataset=None, tg_playlist=None, positive_threshold=3, recompile_cython=False, sparse_weights=False, sgd_mode='adagrad'):

    super(SLIM_BPR_Cython, self).__init__(URM_train,
                                          positive_threshold=positive_threshold,
                                          sparse_weights=sparse_weights)

    self.sgd_mode = sgd_mode
    self.evaluator = ev
    self.dataset = dataset
    self.pl_id_list = list(tg_playlist)
    self.tr_id_list = list(tg_tracks)

    if recompile_cython:
      print("Compiling in Cython")
      self.runCompilationScript()
      print("Compilation Complete")

  def fit(self, S=None, epochs=30, logFile=None, URM_test=None, minRatingsPerUser=1,
          batch_size=1000, validate_every_N_epochs=1, start_validation_after_N_epochs=0,
          lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05, topK=False, sgd_mode='adagrad'):

    # if topK!= False:
    #     raise ValueError("Nope, TopK è rotto da qualche parte")

    self.eligibleUsers = []

    self.W_sparse = S

    # Select only positive interactions
    URM_train_positive = self.URM_train.copy()

    # Useless since implicit >= self.positive_threshold
    URM_train_positive.data = URM_train_positive.data
    URM_train_positive.eliminate_zeros()

    for user_id in range(self.n_users):

      start_pos = URM_train_positive.indptr[user_id]
      end_pos = URM_train_positive.indptr[user_id + 1]

      if len(URM_train_positive.indices[start_pos:end_pos]) > 0:
        self.eligibleUsers.append(user_id)

    self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
    self.sgd_mode = sgd_mode

    # Import compiled module
    from src.SLIM_BPR.Cython.SLIM_BPR_Cython_Epoch2 import SLIM_BPR_Cython_Epoch

    self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                             self.sparse_weights,
                                             self.eligibleUsers,
                                             S=S,
                                             topK=topK,
                                             learning_rate=learning_rate,
                                             batch_size=1,
                                             sgd_mode=sgd_mode)

    # Cal super.fit to start training
    super(SLIM_BPR_Cython, self).fit_alreadyInitialized(epochs=epochs,
                                                        logFile=logFile,
                                                        URM_test=URM_test,
                                                        minRatingsPerUser=minRatingsPerUser,
                                                        batch_size=batch_size,
                                                        validate_every_N_epochs=validate_every_N_epochs,
                                                        start_validation_after_N_epochs=start_validation_after_N_epochs,
                                                        lambda_i=lambda_i,
                                                        lambda_j=lambda_j,
                                                        learning_rate=learning_rate,
                                                        topK=topK)

  def runCompilationScript(self):

    # Run compile script setting the working directory to ensure the compiled file are contained in the
    # appropriate subfolder and not the project root

    compiledModuleSubfolder = "/src/SLIM_BPR/Cython"
    fileToCompile_list = ['SLIM_BPR_Cython_Epoch2.pyx']

    for fileToCompile in fileToCompile_list:

      command = ['python',
                 'compileCython.py',
                 fileToCompile,
                 'build_ext',
                 '--inplace'
                 ]

      output = subprocess.check_output(
          ' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

      try:

        command = ['cython',
                   fileToCompile,
                   '-a'
                   ]

        output = subprocess.check_output(
            ' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

      except:
        pass

    print("Compiled module saved in subfolder: {}".format(
        compiledModuleSubfolder))

    # Command to run compilation script
    # python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

    # Command to generate html report
    #subprocess.call(["cython", "-a", "SLIM_BPR_Cython_Epoch.pyx"])

  def evaluateRecommendations(self, URM_test, minRatingsPerUser=None, at=5):
    print("Evaluating recommendations")
    # Compute prediction matrix (n_target_users X n_items)
    urm_red = self.URM_train[[self.dataset.get_playlist_index_from_id(x) for x in self.pl_id_list]]
    w_red = self.W_sparse[:,[self.dataset.get_track_index_from_id(x)
                                for x in self.tr_id_list]]
    self.R_hat = urm_red.dot(w_red).tocsr()
    print('R_hat evaluated...')
    urm_red = urm_red[:,[self.dataset.get_track_index_from_id(x)
                                for x in self.tr_id_list]]
    # Clean R_hat from already rated entries
    self.R_hat[urm_red.nonzero()] = 0
    self.R_hat.eliminate_zeros()
    """
    returns a dictionary of
    'pl_id': ['tr_1', 'tr_at'] for each playlist in target playlist
    """
    recs = {}
    for i in range(0, self.R_hat.shape[0]):
        pl_id = self.pl_id_list[i]
        pl_row = self.R_hat.data[self.R_hat.indptr[i]:
                                 self.R_hat.indptr[i + 1]]
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
        track_cols = [self.R_hat.indices[self.R_hat.indptr[i] + x]
                      for x in sorted_row_idx]
        tracks_ids = [self.tr_id_list[x] for x in track_cols]
        recs[pl_id] = tracks_ids
    return self.evaluator.evaluate_fold(recs)


  def epochIteration(self):

    self.S = self.cythonEpoch.epochIteration_Cython()

    if self.sparse_weights:
      # it was without T, I think it's correct to add T
      self.W_sparse = self.S.T
    else:
      self.W = self.S.T

  def writeCurrentConfig(self, currentEpoch, results_run, logFile):

    current_config = {'learn_rate': self.learning_rate,
                      'topK_similarity': self.topK,
                      'epoch': currentEpoch,
                      'sgd_mode': self.sgd_mode}

    print("Test case: {}\nResults {}\n".format(current_config, results_run))

    sys.stdout.flush()

    if (logFile != None):
      logFile.write("Test case: {}, Results {}\n".format(
          current_config, results_run))
      logFile.flush()

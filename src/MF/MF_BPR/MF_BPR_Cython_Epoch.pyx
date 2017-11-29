"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# for the adam implementation refer to: 
# http://ruder.io/optimizing-gradient-descent/index.html

import numpy as np
cimport numpy as np
import time
import sys
cimport cython
from libc.math cimport exp, sqrt, pow
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef class MF_BPR_Cython_Epoch:

    cdef int n_users
    cdef int n_items, num_factors
    cdef int numPositiveIteractions
    cdef long urm_nnz

    cdef int useRMSE, useBPR

    cdef int useAdaGrad, rmsprop, useAdam, useAdam2, sgd, useRmsprop2

    cdef float learning_rate, user_reg, positive_reg, negative_reg

    cdef int batch_size, sparse_weights
    cdef long training_step
    cdef long[:] training_step_users
    cdef long[:] training_step_items

    cdef long[:] eligibleUsers
    cdef long numEligibleUsers

    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser

    cdef long[:] target_users
    cdef int num_target_users
    cdef long[:] target_items
    cdef int num_target_items

    cdef int[:] URM_mask_indices, URM_mask_indptr
    cdef double [:] URM_data
    cdef long [:] row_nnz 
    cdef long [:] row_indices
    cdef long [:] shuffled_idx

    # ADAM
    cdef double[:,:] cache_m_user
    cdef double[:,:] cache_v_user

    cdef double[:,:] cache_m_item
    cdef double[:,:] cache_v_item

    cdef double[:] cache_m_user_2
    cdef double[:] cache_v_user_2

    cdef double[:] cache_m_item_2
    cdef double[:] cache_v_item_2

    # RMSPROP
    cdef double[:,:] rmsprop_cache_user
    cdef double[:,:] rmsprop_cache_item

    # RMSPROP version 2
    cdef double[:] rmsprop_cache_user_2
    cdef double[:] rmsprop_cache_item_2

    cdef double[:] sgd_cache
    cdef double[:] sgd_cache_user

    cdef double gamma

    cdef double[:,:] W, H

    # refer to Adam Paper
    cdef double beta1
    cdef double beta2
    cdef double epsilon

    # sample strategy
    cdef int target_prob
    # it's used in this way: 
    # decision = rand()%total_prob
    # if decision < target_prob : sample from target
    # else sample from all
    cdef int total_prob


    def __init__(self, URM_mask, eligibleUsers, num_factors, target_users=None, target_items=None,
                 learning_rate = 0.05, user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
                 batch_size = 1, sgd_mode='sgd', epoch_multiplier=1.0, opt_mode='bpr'):

        super(MF_BPR_Cython_Epoch, self).__init__()

        self.numPositiveIteractions = int(URM_mask.nnz * epoch_multiplier)
        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.num_factors = num_factors

        self.URM_mask_indices = URM_mask.indices
        self.URM_mask_indptr = URM_mask.indptr
        self.URM_data = URM_mask.data
        self.urm_nnz = len(URM_mask.data)

        # RMSE part
        self.row_nnz = np.diff(URM_mask.indptr).astype(np.int64)
        self.row_indices = np.repeat(np.arange(self.n_users), self.row_nnz).astype(np.int64)

        # W and H cannot be initialized as zero, otherwise the gradient will always be zero
        self.W = np.multiply(np.random.random((self.n_users, self.num_factors)), 0.1) # it was 0.1
        self.H = np.multiply(np.random.random((self.n_items, self.num_factors)), 0.1)

        # select optimization mode
        if opt_mode=='bpr':
            self.useBPR = True
        elif opt_mode=='rmse':
            self.useRMSE = True

        if sgd_mode=='sgd':
            self.sgd = True
        elif sgd_mode=='adagrad':
            self.useAdaGrad = True
        elif sgd_mode == 'rmsprop':
            self.rmsprop = True
        elif sgd_mode == 'adam':
            self.useAdam = True
        elif sgd_mode == 'adam2':
            self.useAdam2 = True
        elif sgd_mode == 'rmsprop2':
            self.useRmsprop2 = True
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd'. Provided value was '{}'".format(
                    sgd_mode))

        if self.useAdaGrad:
            self.sgd_cache = np.zeros((self.n_items), dtype=float)
            self.sgd_cache_user = np.zeros((self.n_users), dtype=float)
        
        # RMSPROP
        elif self.rmsprop:
            self.rmsprop_cache_item = np.zeros((self.n_items, self.num_factors), dtype=float)
            self.rmsprop_cache_user = np.zeros((self.n_users, self.num_factors), dtype=float)
            self.gamma = 0.9

        elif self.useRmsprop2:
            self.rmsprop_cache_item_2 = np.zeros((self.n_items), dtype=float)
            self.rmsprop_cache_user_2 = np.zeros((self.n_users), dtype=float)
            self.gamma = 0.9

        # Adam requirements
        elif self.useAdam:
            self.init_Adam()

        elif self.useAdam2:
            self.init_Adam2()

        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg


        if batch_size!=1:
            print("MiniBatch not implemented, reverting to default value 1")
        self.batch_size = 1

        self.eligibleUsers = eligibleUsers
        self.numEligibleUsers = len(eligibleUsers)

        if target_users is not None:
            self.target_users = target_users
            self.num_target_users = len(target_users)
        if target_items is not None:
            self.target_items = target_items
            self.num_target_items = len(target_items)
        self.training_step = 0

        # sample strategy
        self.target_prob = 5
        self.total_prob = 10


    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    def epochIteration_Cython(self, l_rate, reset_cache=True):

        # Get number of available interactions
        cdef long totalNumberOfBatch = int(self.numPositiveIteractions / self.batch_size) + 1


        cdef BPR_sample sample
        cdef long u, i, j, idx
        cdef long index, numCurrentBatch
        cdef double x_uij, sigmoid, error, rui

        cdef int numSeenItems

        # Variables for AdaGrad and RMSprop
        cdef double [:] sgd_cache
        cdef double cacheUpdate, cacheUpdate_i, cacheUpdate_j, cacheUpdate_u
        cdef float gamma

        cdef double H_i, H_j, W_u
        cdef double sigmoid_i, sigmoid_u, sigmoid_j
        cdef double update_u, update_i, update_j
        cdef double gtu, gti, gtj


        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        self.learning_rate = l_rate

        if reset_cache:
            self.training_step = 0

            if self.useAdaGrad:
                self.sgd_cache = np.zeros((self.n_items), dtype=float)
                self.sgd_cache_user = np.zeros((self.n_users), dtype=float)

            elif self.rmsprop:
                self.rmsprop_cache_item = np.zeros((self.n_items, self.num_factors), dtype=float)
                self.rmsprop_cache_user = np.zeros((self.n_users, self.num_factors), dtype=float)
                self.gamma = 0.9

            # Adam requires
            elif self.useAdam:
                self.init_Adam()

            elif self.useAdam2:
                self.init_Adam2()

        if self.useRMSE:
            self.shuffled_idx = np.random.permutation(self.urm_nnz).astype(np.int64)

        for numCurrentBatch in range(totalNumberOfBatch):

            if self.useBPR:
                # Uniform user sampling with replacement
                sample = self.sampleBatch_Cython()

                u = sample.user
                i = sample.pos_item
                j = sample.neg_item

            else:
                idx = self.shuffled_idx[numCurrentBatch % self.urm_nnz]
                rui = self.URM_data[idx]
                # get the row and col indices of x_ij
                u = self.row_indices[idx]
                i = self.URM_mask_indices[idx]

            # add training step
            if self.useAdam or self.useAdam2:
                self.training_step_users[u] += 1
                self.training_step_items[i] += 1

                if self.useBPR:
                    self.training_step_items[j] += 1


            x_uij = 0.0

            for index in range(self.num_factors):

                if self.useBPR:
                    x_uij += self.W[u,index] * (self.H[i,index] - self.H[j,index])
                
                else:
                    x_uij += self.W[u,index] * self.H[i,index]

            if self.useRMSE:
                error = rui - x_uij 


            # Use gradient of log(sigm(-x_uij))
            sigmoid = 1 / (1 + exp(x_uij))
            sigmoid_i = sigmoid
            sigmoid_j = sigmoid
            sigmoid_u = sigmoid

            #   OLD CODE, YOU MAY TRY TO USE IT
            # if self.useAdaGrad:
            #     cacheUpdate = gradient ** 2
            
            #     self.sgd_cache[i] += cacheUpdate
            #     self.sgd_cache[j] += cacheUpdate
            #     self.sgd_cache_user[u]+= cacheUpdate

            #     gradient_i = gradient / (sqrt(self.sgd_cache[i]) + 1e-8)
            #     gradient_j = gradient / (sqrt(self.sgd_cache[j]) + 1e-8)
            #     gradient_u = gradient / (sqrt(self.sgd_cache_user[u]) + 1e-8)
            
            # elif self.rmsprop:
            #     cacheUpdate_i = self.sgd_cache[i] * self.gamma + (1 - self.gamma) * gradient ** 2
            #     cacheUpdate_j = self.sgd_cache[j] * self.gamma + (1 - self.gamma) * gradient ** 2
            #     cacheUpdate_u = self.sgd_cache_user[u] * self.gamma + (1 - self.gamma) * gradient ** 2
            
            #     self.sgd_cache[i] = cacheUpdate_i
            #     self.sgd_cache[j] = cacheUpdate_j
            #     self.sgd_cache_user[u] = cacheUpdate_u
            
            #     gradient_i = gradient / (sqrt(self.sgd_cache[i]) + 1e-8)
            #     gradient_j = gradient / (sqrt(self.sgd_cache[j]) + 1e-8)
            #     gradient_u = gradient / (sqrt(self.sgd_cache_user[u]) + 1e-8)

            if self.useAdam2:
                sigmoid_u = self.get_adam_update_user2(sigmoid, u)
                sigmoid_i = self.get_adam_update_item2(sigmoid, i)
                sigmoid_j = self.get_adam_update_item2(sigmoid, j)

            elif self.useRmsprop2:

                sigmoid_u = self.get_rmsprop_update_user2(sigmoid, u)
                sigmoid_i = self.get_rmsprop_update_item2(sigmoid, i)
                sigmoid_j = self.get_rmsprop_update_item2(sigmoid, j)

            for index in range(self.num_factors):

                if self.useBPR:
                    # Copy original value to avoid messing up the updates
                    H_i = self.H[i, index]
                    H_j = self.H[j, index]
                    W_u = self.W[u, index]

                    # calculate gradients
                    gtu = (-sigmoid_u * ( H_i - H_j ) + self.user_reg * W_u)
                    gti = (-sigmoid_i * ( W_u ) + self.positive_reg * H_i)
                    gtj = (-sigmoid_j * (-W_u ) + self.negative_reg * H_j)


                    if self.useAdam:

                        # update of U params
                        update_u = self.get_adam_update_user(gtu, u, index)

                        # update of I params
                        update_i = self.get_adam_update_item(gti, i, index)

                        # update of J params
                        update_j = self.get_adam_update_item(gtj, j, index)

                    elif self.rmsprop:

                        # update of U params
                        update_u = self.get_rmsprop_update_user(gtu, u, index)

                        # update of I params
                        update_i = self.get_rmsprop_update_item(gti, i, index)

                        # update of J params
                        update_j = self.get_rmsprop_update_item(gtj, j, index)

                    else:
                        # the update is normal
                        update_u = gtu
                        update_j = gtj
                        update_i = gti

                    # Let's update
                    self.W[u, index] -= self.learning_rate * update_u
                    self.H[i, index] -= self.learning_rate * update_i
                    self.H[j, index] -= self.learning_rate * update_j
                
                else:

                    # Copy original value to avoid messing up the updates
                    H_i = self.H[i, index]
                    W_u = self.W[u, index]

                    # calculate gradients
                    gtu = (error * ( H_i ) + self.user_reg * W_u)
                    gti = (error * ( W_u ) + self.positive_reg * H_i)

                    # update of U params
                    update_u = self.get_adam_update_user(gtu, u, index)

                    # update of I params
                    update_i = self.get_adam_update_item(gti, i, index)

                    # Let's update
                    self.W[u, index] -= self.learning_rate * update_u
                    self.H[i, index] -= self.learning_rate * update_i



            if((numCurrentBatch%500000==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


    def get_W(self):

        return np.array(self.W)


    def get_H(self):
        return np.array(self.H)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_adam_update_user(self, float gradient, int u, int k):
        """
        Returns the update due to the Adam optimization technique
        """
        cdef double new_cache_m_value, new_cache_v_value, m_hat, v_hat
        new_cache_m_value = self.beta1 * self.cache_m_user[u, k] + (1 - self.beta1) * gradient
        new_cache_v_value = self.beta2 * self.cache_v_user[u, k] + (1 - self.beta2) * gradient * gradient

        # update caches
        self.cache_m_user[u, k] = new_cache_m_value
        self.cache_v_user[u, k] = new_cache_v_value

        # correction part
        m_hat = new_cache_m_value / (1 - pow(self.beta1, self.training_step_users[u]))
        v_hat = new_cache_v_value / (1 - pow(self.beta2, self.training_step_users[u]))

        return m_hat / (sqrt(v_hat) + self.epsilon)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_adam_update_item(self, float gradient, int i, int k):
        """
        Returns the update due to the Adam optimization technique
        """
        cdef double new_cache_m_value, new_cache_v_value, m_hat, v_hat
        new_cache_m_value = self.beta1 * self.cache_m_item[i, k] + (1 - self.beta1) * gradient
        new_cache_v_value = self.beta2 * self.cache_v_item[i, k] + (1 - self.beta2) * gradient * gradient

        # update caches
        self.cache_m_item[i, k] = new_cache_m_value
        self.cache_v_item[i, k] = new_cache_v_value

        # correction part
        m_hat = new_cache_m_value / (1 - pow(self.beta1, self.training_step_items[i]))
        v_hat = new_cache_v_value / (1 - pow(self.beta2, self.training_step_items[i]))

        return m_hat / (sqrt(v_hat) + self.epsilon)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_adam_update_user2(self, float gradient, int u):
        """
        Returns the update due to the Adam optimization technique
        """
        cdef double new_cache_m_value, new_cache_v_value, m_hat, v_hat
        new_cache_m_value = self.beta1 * self.cache_m_user_2[u] + (1 - self.beta1) * gradient
        new_cache_v_value = self.beta2 * self.cache_v_user_2[u] + (1 - self.beta2) * gradient * gradient

        # update caches
        self.cache_m_user_2[u] = new_cache_m_value
        self.cache_v_user_2[u] = new_cache_v_value

        # correction part
        m_hat = new_cache_m_value / (1 - pow(self.beta1, self.training_step_users[u]))
        v_hat = new_cache_v_value / (1 - pow(self.beta2, self.training_step_users[u]))

        return m_hat / (sqrt(v_hat) + self.epsilon)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_adam_update_item2(self, float gradient, int i):
        """
        Returns the update due to the Adam optimization technique
        """
        cdef double new_cache_m_value, new_cache_v_value, m_hat, v_hat
        new_cache_m_value = self.beta1 * self.cache_m_item_2[i] + (1 - self.beta1) * gradient
        new_cache_v_value = self.beta2 * self.cache_v_item_2[i] + (1 - self.beta2) * gradient * gradient

        # update caches
        self.cache_m_item_2[i] = new_cache_m_value
        self.cache_v_item_2[i] = new_cache_v_value

        # correction part
        m_hat = new_cache_m_value / (1 - pow(self.beta1, self.training_step_items[i]))
        v_hat = new_cache_v_value / (1 - pow(self.beta2, self.training_step_items[i]))

        return m_hat / (sqrt(v_hat) + self.epsilon)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_rmsprop_update_user(self, float gradient, int u, int k):
        """
        Returns the update due to the RMSprop optimization technique
        """
        cdef double new_cache_value
        new_cache_value = self.rmsprop_cache_user[u, k] * self.gamma + (1 - self.gamma) * gradient ** 2
        
        self.rmsprop_cache_user[u, k] = new_cache_value

        return gradient / (sqrt(new_cache_value + 1e-8))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_rmsprop_update_item(self, float gradient, int i, int k):
        """
        Returns the update due to the RMSprop optimization technique
        """
        cdef double new_cache_value
        new_cache_value = self.rmsprop_cache_item[i, k] * self.gamma + (1 - self.gamma) * gradient ** 2
        
        self.rmsprop_cache_item[i, k] = new_cache_value

        return gradient / (sqrt(new_cache_value + 1e-8))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_rmsprop_update_user2(self, float gradient, int u):
        """
        Returns the update due to the RMSprop optimization technique
        """
        cdef double new_cache_value
        new_cache_value = self.rmsprop_cache_user_2[u] * self.gamma + (1 - self.gamma) * gradient ** 2
        
        self.rmsprop_cache_user_2[u] = new_cache_value

        return gradient / (sqrt(new_cache_value) + 1e-8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef float get_rmsprop_update_item2(self, float gradient, int i):
        """
        Returns the update due to the RMSprop optimization technique
        """
        cdef double new_cache_value
        new_cache_value = self.rmsprop_cache_item_2[i] * self.gamma + (1 - self.gamma) * gradient ** 2
        
        self.rmsprop_cache_item_2[i] = new_cache_value

        return gradient / (sqrt(new_cache_value) + 1e-8)

    cdef void init_Adam(self):
        self.cache_m_user = np.zeros((self.n_users, self.num_factors), dtype=float)
        self.cache_v_user = np.zeros((self.n_users, self.num_factors), dtype=float)
        self.cache_m_item = np.zeros((self.n_items, self.num_factors), dtype=float)
        self.cache_v_item = np.zeros((self.n_items, self.num_factors), dtype=float)
        self.training_step_users = np.zeros((self.n_users), dtype=long)
        self.training_step_items = np.zeros((self.n_items), dtype=long)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    cdef void init_Adam2(self):
        self.cache_m_user_2 = np.zeros((self.n_users), dtype=float)
        self.cache_v_user_2 = np.zeros((self.n_users), dtype=float)
        self.cache_m_item_2 = np.zeros((self.n_items), dtype=float)
        self.cache_v_item_2 = np.zeros((self.n_items), dtype=float)
        self.training_step_users = np.zeros((self.n_users), dtype=long)
        self.training_step_items = np.zeros((self.n_items), dtype=long)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef BPR_sample sampleBatch_Cython(self):

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected

        # Warning: rand() returns an integer

        index = rand() % self.numEligibleUsers

        sample.user = self.eligibleUsers[index]

        self.seenItemsSampledUser = self.getSeenItems(sample.user)
        self.numSeenItemsSampledUser = len(self.seenItemsSampledUser)

        index = rand() % self.numSeenItemsSampledUser

        sample.pos_item = self.seenItemsSampledUser[index]


        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):
            sample.neg_item = rand() % self.n_items

            index = 0
            while index < self.numSeenItemsSampledUser and self.seenItemsSampledUser[index]!=sample.neg_item:
                index+=1

            if index == self.numSeenItemsSampledUser:
                negItemSelected = True

        return sample

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef BPR_sample sample_from_target(self):
        """
        Takes one of the two items from the target items
        """

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected
        cdef int decision

        # Warning: rand() returns an integer

        # decide if sample from target or from non target
        decision = rand() % self.total_prob

        if decision > self.target_prob:

            # sample from all
            index = rand() % self.numEligibleUsers

            sample.user = self.eligibleUsers[index]

        else:

            # sample from target
            index = rand() % self.num_target_users

            sample.user = self.target_users[index]

        self.seenItemsSampledUser = self.getSeenItems(sample.user)
        self.numSeenItemsSampledUser = len(self.seenItemsSampledUser)

        index = rand() % self.numSeenItemsSampledUser

        sample.pos_item = self.seenItemsSampledUser[index]


        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):
            sample.neg_item = rand() % self.n_items

            index = 0
            while index < self.numSeenItemsSampledUser and self.seenItemsSampledUser[index]!=sample.neg_item:
                index+=1

            if index == self.numSeenItemsSampledUser:
                negItemSelected = True

        return sample

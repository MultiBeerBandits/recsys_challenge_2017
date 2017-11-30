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

from src.SLIM_BPR.Recommender_utils import check_matrix
import numpy as np
cimport numpy as np
import time
import sys
import scipy.sparse as sp

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef class SLIM_BPR_Cython_Epoch:

    cdef int n_users
    cdef int n_items
    cdef int n_features
    cdef int m_rows, m_cols
    cdef int numPositiveIteractions
    cdef int topK
    cdef int useAdaGrad, rmsprop, momentum

    cdef float learning_rate

    cdef float urmSamplingChances, icmSamplingChances
    cdef long[:] eligibleUsers
    cdef long[:] eligibleFeatures
    
    cdef long numEligibleUsers
    cdef long numEligibleFeatures

    cdef int[:] M_indices, M_indptr

    cdef Sparse_Matrix_Tree_CSR S
    cdef M

    def __init__(self,
                 urm,
                 icm,
                 eligibleUsers,
                 eligibleFeatures,
                 learning_rate,
                 urmSamplingChances=0.5,
                 icmSamplingChances=0.5,
                 epochMultiplier=1,
                 topK=False,
                 sgd_mode='adagrad'):
        self.numPositiveIteractions = 0
        if urm is not None:
            urm = check_matrix(urm, 'csr')
            self.numPositiveIteractions += int(urm.nnz * epochMultiplier)
        if icm is not None:
            icm = check_matrix(icm, 'csr')
            self.numPositiveIteractions += int(icm.nnz * epochMultiplier)

        self.n_users = urm.shape[0]
        self.n_features = icm.shape[0]
        self.n_items = urm.shape[1]
        self.topK = topK
        self.urmSamplingChances = urmSamplingChances
        self.icmSamplingChances = icmSamplingChances

        # Stack urm and icm to optimize cSlim
        if urm is not None and icm is not None:
            self.M = sp.vstack((urm, icm), format='csr')
        elif urm is not None:
            self.M = urm
        else:
            self.M = icm

        self.m_rows = self.M.shape[0]
        self.m_cols = self.M.shape[1]

        self.M_indices = self.M.indices
        self.M_indptr = self.M.indptr

        self.S = Sparse_Matrix_Tree_CSR(self.m_cols, self.m_cols)

        if sgd_mode == 'adagrad':
            self.useAdaGrad = True
        elif sgd_mode == 'rmsprop':
            self.rmsprop = True
        elif sgd_mode == 'momentum':
            self.momentum = True
        elif sgd_mode == 'sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop'. Provided value was '{}'".format(
                    sgd_mode))

        self.learning_rate = learning_rate
        self.eligibleUsers = eligibleUsers
        self.eligibleFeatures = eligibleFeatures
        
        self.numEligibleUsers = len(eligibleUsers)
        self.numEligibleFeatures = len(eligibleFeatures)

    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.M_indices[self.M_indptr[index]:self.M_indptr[index + 1]]

    def epochIteration_Cython(self):
        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        cdef BPR_sample sample
        cdef int[:] seenItemsSampledRow
        cdef int numSeenItemsSampledRow
        cdef long i, j
        cdef long index, seenItem, numIteration, itemId
        cdef double x_uij, gradient, dS_i, dS_j, gradient_i, gradient_j

        cdef int numSeenItems
        cdef int printStep = 500000

        # Variables for AdaGrad and RMSprop
        cdef double [:] sgd_cache
        cdef double cacheUpdate, cacheUpdate_i, cacheUpdate_j
        cdef float gamma

        if self.useAdaGrad:
            sgd_cache = np.zeros((self.n_items), dtype=float)

        elif self.rmsprop:
            sgd_cache = np.zeros((self.n_items), dtype=float)
            gamma = 0.9

        elif self.momentum:
            sgd_cache = np.zeros((self.n_items), dtype=float)
            gamma = 1e-5

        for numIteration in range(self.numPositiveIteractions):
            sample = self.sampleBatch_Cython()

            i = sample.pos_item
            j = sample.neg_item

            seenItemsSampledRow = self.getSeenItems(sample.user)
            numSeenItemsSampledRow = len(seenItemsSampledRow)

            x_uij = 0.0

            # The difference is computed on the user_seen items
            index = 0
            while index < numSeenItemsSampledRow:
                seenItem = seenItemsSampledRow[index]
                index += 1
                x_uij += self.S.get_value(i, seenItem) - self.S.get_value(j, seenItem)

            gradient = 1 / (1 + exp(x_uij))

            if self.useAdaGrad:
                cacheUpdate = gradient ** 2

                sgd_cache[i] += cacheUpdate
                sgd_cache[j] += cacheUpdate

                gradient_i = gradient / (sqrt(sgd_cache[i]) + 1e-8)
                gradient_j = gradient / (sqrt(sgd_cache[j]) + 1e-8)

                dS_i = self.learning_rate * gradient_i
                dS_j = self.learning_rate * (-gradient_j)

            elif self.rmsprop:
                cacheUpdate_i = sgd_cache[i] * gamma + (1 - gamma) * gradient ** 2
                cacheUpdate_j = sgd_cache[j] * gamma + (1 - gamma) * gradient ** 2

                sgd_cache[i] = cacheUpdate_i
                sgd_cache[j] = cacheUpdate_j

                gradient_i = gradient / (sqrt(sgd_cache[i]) + 1e-8)
                gradient_j = gradient / (sqrt(sgd_cache[j]) + 1e-8)

                dS_i = self.learning_rate * gradient_i
                dS_j = self.learning_rate * (-gradient_j)

            elif self.momentum:
                dS_i = self.learning_rate * gradient + gamma * sgd_cache[i]
                dS_j = self.learning_rate * (-gradient) + gamma * sgd_cache[j]

                sgd_cache[i] = dS_i
                sgd_cache[j] = dS_j                
            
            else:
                dS_i = self.learning_rate * gradient
                dS_j = self.learning_rate * (-gradient)

            index = 0
            while index < numSeenItemsSampledRow:
                seenItem = seenItemsSampledRow[index]
                index += 1
                
                if seenItem != i:
                    self.S.add_value(i, seenItem, dS_i)
                if seenItem != j:
                    self.S.add_value(j, seenItem, dS_j)

            if((numIteration % printStep == 0 and not numIteration == 0) or
                numIteration == self.numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numIteration,
                    100.0 * float(numIteration) / self.numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numIteration + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        # Fill diagonal with zeros
        index = 0
        while index < self.n_items:
            self.S.add_value(index, index, -self.S.get_value(index, index))
            index += 1

        if self.topK == False:
            print("Return S matrix to python caller...")
            return self.S.get_scipy_csr(TopK = False)
        else:
            print("Return S matrix to python caller...")
            return self.S.get_scipy_csr(TopK=self.topK)

    def build_S(self,S):
        """
        S is scipy sparse matrix
        this method converts the matrix into a Sparse Matrix Tree CSR
        """
        # first transpose the matrix since here we need to handle it by row
        S = S.transpose()
        for r in range(S.shape[0]):
            # get all the element
            row = S.data[S.indptr[r]:S.indptr[r+1]]
            columns = S.indices[S.indptr[r]:S.indptr[r+1]]
            for i in range(row.shape[0]):
                self.S.add_value(r, columns[i], row[i])
        # rebalance the tree
        self.S.rebalance_tree()    


    cdef BPR_sample sampleBatch_Cython(self):

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected

        # Warning: rand() returns an integer, in order to avoid integer division we force the
        # denominator to be a float
        cdef double RAND_MAX_DOUBLE = RAND_MAX

        # Compute probability of picking from urm users. If a random sampling
        # [0, 1) is less than picking-urm probability than sample from URM,
        # otherwise sample from ICM
        cdef double urmProb = self.urmSamplingChances
        cdef double choice = rand() / RAND_MAX_DOUBLE
        if choice < urmProb:
            index = int(rand() / RAND_MAX_DOUBLE * self.numEligibleUsers)
            sample.user = self.eligibleUsers[index]
        else:
            # If index is from ICM we need to add to it the number of users in
            # URM because we are working on the matrix result of stacking the
            # URM onto ICM
            index = int(rand() / RAND_MAX_DOUBLE * self.numEligibleFeatures)
            sample.user = self.n_users + self.eligibleFeatures[index]

        seenItemsSampledRow = self.getSeenItems(sample.user)
        numSeenItemsSampledRow = len(seenItemsSampledRow)

        index = int(rand() / RAND_MAX_DOUBLE * numSeenItemsSampledRow)
        sample.pos_item = seenItemsSampledRow[index]

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):
            sample.neg_item = int(rand() / RAND_MAX_DOUBLE  * self.n_items)

            index = 0
            while index < numSeenItemsSampledRow and seenItemsSampledRow[index] != sample.neg_item:
                index+=1

            if index == numSeenItemsSampledRow:
                negItemSelected = True
        return sample


##################################################################################################################
#####################
#####################            SPARSE MATRIX
#####################
##################################################################################################################

import scipy.sparse as sps

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free#, qsort

# Declaring QSORT as "gil safe", appending "nogil" at the end of the declaration
# Otherwise I will not be able to pass the comparator function pointer
# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-function-in-cython
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


# Node struct
ctypedef struct matrix_element_tree_s:
    long column
    double data
    matrix_element_tree_s *higher
    matrix_element_tree_s *lower

ctypedef struct head_pointer_tree_s:
    matrix_element_tree_s *head


# Function to allocate a new node
cdef matrix_element_tree_s * pointer_new_matrix_element_tree_s(long column, double data, matrix_element_tree_s *higher,  matrix_element_tree_s *lower):

    cdef matrix_element_tree_s * new_element

    new_element = < matrix_element_tree_s * > malloc(sizeof(matrix_element_tree_s))
    new_element.column = column
    new_element.data = data
    new_element.higher = higher
    new_element.lower = lower

    return new_element


# Functions to compare structs to be used in C qsort
cdef int compare_struct_on_column(const void *a_input, const void *b_input):
    """
    The function compares the column contained in the two struct passed.
    If a.column > b.column returns >0  
    If a.column < b.column returns <0      
    
    :return int: a.column - b.column
    """

    cdef head_pointer_tree_s *a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s *b_casted = <head_pointer_tree_s *> b_input

    return a_casted.head.column  - b_casted.head.column



cdef int compare_struct_on_data(const void * a_input, const void * b_input):
    """
    The function compares the data contained in the two struct passed.
    If a.data > b.data returns >0  
    If a.data < b.data returns <0      
    
    :return int: +1 or -1
    """

    cdef head_pointer_tree_s * a_casted = <head_pointer_tree_s *> a_input
    cdef head_pointer_tree_s * b_casted = <head_pointer_tree_s *> b_input

    if (a_casted.head.data - b_casted.head.data) > 0.0:
        return +1
    else:
        return -1



#################################
#################################       CLASS DECLARATION
#################################

cdef class Sparse_Matrix_Tree_CSR:

    cdef long num_rows, num_cols

    # Array containing the struct (object, not pointer) corresponding to the root of the tree
    cdef head_pointer_tree_s* row_pointer

    def __init__(self, long num_rows, long num_cols):

        self.num_rows = num_rows
        self.num_cols = num_cols

        self.row_pointer = < head_pointer_tree_s *> malloc(self.num_rows * sizeof(head_pointer_tree_s))

        # Initialize all rows to empty
        for index in range(self.num_rows):
            self.row_pointer[index].head = NULL


    cpdef double add_value(self, long row, long col, double value):
        """
        The function adds a value to the specified cell. A new cell is created if necessary.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :param value: value to add
        :return double: resulting cell value
        """

        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError("Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                self.num_rows, self.num_cols, row, col))

        cdef matrix_element_tree_s* current_element, new_element, * old_element
        cdef int stopSearch = False


        # If the row is empty, create a new element
        if self.row_pointer[row].head == NULL:

            # row_pointer is a python object, so I need the object itself and not the address
            self.row_pointer[row].head = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True

        # If the cell exist, update its value
        if current_element.column == col:
            current_element.data += value

            return current_element.data


        # The cell is not found, create new Higher element
        elif current_element.column < col and current_element.higher == NULL:

            current_element.higher = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        # The cell is not found, create new Lower element
        elif current_element.column > col and current_element.lower == NULL:

            current_element.lower = pointer_new_matrix_element_tree_s(col, value, NULL, NULL)

            return value

        else:
            assert False, 'ERROR - Current insert operation is not implemented'




    cpdef double get_value(self, long row, long col):
        """
        The function returns the value of the specified cell.         
        
        :param row: cell coordinates
        :param col:  cell coordinates
        :return double: cell value
        """


        if row >= self.num_rows or col >= self.num_cols or row < 0 or col < 0:
            raise ValueError(
                "Cell is outside matrix. Matrix shape is ({},{}), coordinates given are ({},{})".format(
                    self.num_rows, self.num_cols, row, col))


        cdef matrix_element_tree_s* current_element
        cdef int stopSearch = False

        # If the row is empty, return default
        if self.row_pointer[row].head == NULL:
            return 0.0


        # If the row is not empty, look for the cell
        # row_pointer contains the struct itself, but I just want its address
        current_element = self.row_pointer[row].head

        # Follow the tree structure
        while not stopSearch:

            if current_element.column < col and current_element.higher != NULL:
                current_element = current_element.higher

            elif current_element.column > col and current_element.lower != NULL:
                current_element = current_element.lower

            else:
                stopSearch = True


        # If the cell exist, return its value
        if current_element.column == col:
            return current_element.data

        # The cell is not found, return default
        else:
            return 0.0




    cpdef get_scipy_csr(self, long TopK = False):
        """
        The function returns the current sparse matrix as a scipy_csr object         
   
        :return double: scipy_csr object
        """
        cdef int terminate
        cdef long row

        data = []
        indices = []
        indptr = []

        # Loop the rows
        for row in range(self.num_rows):

            #Always set indptr
            indptr.append(len(data))

            # row contains data
            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)


                # Flatten the tree data
                subtree_column, subtree_data = self.from_linked_list_to_python_list(self.row_pointer[row].head)
                data.extend(subtree_data)
                indices.extend(subtree_column)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)


        #Set terminal indptr
        indptr.append(len(data))

        return sps.csr_matrix((data, indices, indptr), shape=(self.num_rows, self.num_cols))



    cpdef rebalance_tree(self, long TopK = False):
        """
        The function builds a balanced binary tree from the current one, for all matrix rows
        
        :param TopK: either False or an integer number. Number of the highest elements to preserve
        """

        cdef long row

        #start_time = time.time()

        for row in range(self.num_rows):

            if self.row_pointer[row].head != NULL:

                # Flatten the data structure
                self.row_pointer[row].head = self.subtree_to_list_flat(self.row_pointer[row].head)

                if TopK:
                    self.row_pointer[row].head = self.topK_selection_from_list(self.row_pointer[row].head, TopK)

                # Rebuild the tree
                self.row_pointer[row].head = self.build_tree_from_list_flat(self.row_pointer[row].head)
















    cdef matrix_element_tree_s * subtree_to_list_flat(self, matrix_element_tree_s * root):
        """
        The function flatten the structure of the subtree whose root is passed as a paramether    
        The list is bidirectional and ordered with respect to the column
        The column ordering follows from the insertion policy
        
        :param root: tree root
        :return list, list: data and corresponding column. Empty list if root is None
        """

        if root == NULL:
            return NULL

        cdef matrix_element_tree_s *flat_list_head, *current_element

        # Flatten lower subtree
        flat_list_head = self.subtree_to_list_flat(root.lower)

        # If no lower elements exist, the head is the current element
        if flat_list_head == NULL:
            flat_list_head = root
            root.lower = NULL

        # Else move to the tail and add the subtree root
        else:
            current_element = flat_list_head
            while current_element.higher != NULL:
                current_element = current_element.higher

            # Attach the element with the bidirectional pointers
            current_element.higher = root
            root.lower = current_element

        # Flatten higher subtree and attach it to the tail of the flat list
        root.higher = self.subtree_to_list_flat(root.higher)

        # Attach the element with the bidirectional pointers
        if root.higher != NULL:
            root.higher.lower = root

        return flat_list_head



    cdef from_linked_list_to_python_list(self, matrix_element_tree_s * head):

        data = []
        column = []

        while head != NULL:
            data.append(head.data)
            column.append(head.column)

            head = head.higher

        return column, data



    cdef subtree_free_memory(self, matrix_element_tree_s* root):
        """
        The function frees all struct in the subtree whose root is passed as a parameter, root included 
        
        :param root: tree root
        """

        if root != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(root.higher)
            self.subtree_free_memory(root.lower)

            # Once the lower elements have been reached, start freeing from the bottom
            free(root)



    cdef list_free_memory(self, matrix_element_tree_s * head):
        """
        The function frees all struct in the list whose head is passed as a parameter, head included 
        
        :param head: list head
        """

        if head != NULL:
            # If the root exists, open recursion
            self.subtree_free_memory(head.higher)

            # Once the tail element have been reached, start freeing from them
            free(head)



    cdef matrix_element_tree_s* build_tree_from_list_flat(self, matrix_element_tree_s* flat_list_head):
        """
        The function builds a tree containing the passed data. This is the recursive function, the 
        data should be sorted by te caller
        To ensure the tree is balanced, data is sorted according to the column   
        
        :param row: row in which to create new tree
        :param column_vector: column coordinates 
        :param data_vector: cell data
        """

        if flat_list_head == NULL:
            return NULL


        cdef long list_length = 0
        cdef long middle_element_step = 0

        cdef matrix_element_tree_s *current_element, *middleElement, *tree_root

        current_element = flat_list_head
        middleElement = flat_list_head

        # Explore the flat list moving the middle elment every tho jumps
        while current_element != NULL:
            current_element = current_element.higher
            list_length += 1
            middle_element_step += 1

            if middle_element_step == 2:
                middleElement = middleElement.higher
                middle_element_step = 0

        tree_root = middleElement

        # To execute the recursion it is necessary to cut the flat list
        # The last of the lower elements will have to be a tail
        if middleElement.lower != NULL:
            middleElement.lower.higher = NULL

            tree_root.lower = self.build_tree_from_list_flat(flat_list_head)


        # The first of the higher elements will have to be a head
        if middleElement.higher != NULL:
            middleElement.higher.lower = NULL

            tree_root.higher = self.build_tree_from_list_flat(middleElement.higher)


        return tree_root




    cdef matrix_element_tree_s* topK_selection_from_list(self, matrix_element_tree_s* head, long TopK):
        """
        The function selects the topK highest elements in the given list 
        
        :param head: head of the list
        :param TopK: number of highest elements to preserve
        :return matrix_element_tree_s*: head of the new list
        """

        cdef head_pointer_tree_s *vector_pointer_to_list_elements
        cdef matrix_element_tree_s *current_element
        cdef long list_length, index, selected_count

        # Get list size
        current_element = head
        list_length = 0

        while current_element != NULL:
            list_length += 1
            current_element = current_element.higher


        # If list elements are not enough to perform a selection, return
        if list_length < TopK:
            return head

        # Allocate vector that will be used for sorting
        vector_pointer_to_list_elements = < head_pointer_tree_s *> malloc(list_length * sizeof(head_pointer_tree_s))

        # Fill vector wit pointers to list elements
        current_element = head
        for index in range(list_length):
            vector_pointer_to_list_elements[index].head = current_element
            current_element = current_element.higher


        # Sort array elements on their data field
        qsort(vector_pointer_to_list_elements, list_length, sizeof(head_pointer_tree_s), compare_struct_on_data)

        # Sort only the TopK according to their column field
        # Sort is from lower to higher, therefore the elements to be considered are from len-topK to len
        qsort(&vector_pointer_to_list_elements[list_length-TopK], TopK, sizeof(head_pointer_tree_s), compare_struct_on_column)


        # Rebuild list attaching the consecutive elements
        index = list_length-TopK

        # Detach last TopK element from previous ones
        vector_pointer_to_list_elements[index].head.lower = NULL

        while index<list_length-1:
            # Rearrange bidirectional pointers
            vector_pointer_to_list_elements[index+1].head.lower = vector_pointer_to_list_elements[index].head
            vector_pointer_to_list_elements[index].head.higher = vector_pointer_to_list_elements[index+1].head

            index += 1

        # Last element in vector will be the hew head
        vector_pointer_to_list_elements[list_length - 1].head.higher = NULL

        # Get hew list head
        current_element = vector_pointer_to_list_elements[list_length-TopK].head

        # If there are exactly enough elements to reach TopK, index == 0 will be the tail
        # Else, index will be the tail and the other elements will be removed
        index = list_length - TopK - 1
        if index > 0:

            index -= 1
            while index >= 0:
                free(vector_pointer_to_list_elements[index].head)
                index -= 1

        # Free array
        free(vector_pointer_to_list_elements)


        return current_element






##################################################################################################################
#####################
#####################            TEST FUNCTIONS
#####################
##################################################################################################################


    cpdef test_list_tee_conversion(self, long row):
        """
        The function tests the inner data structure conversion from tree to C linked list and back to tree
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head, *tree_root
        cdef matrix_element_tree_s *current_element, *previous_element

        head = self.subtree_to_list_flat(self.row_pointer[row].head)
        current_element = head

        cdef numElements_higher = 0
        cdef numElements_lower = 0

        while current_element != NULL:
            numElements_higher += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower += 1
            current_element = current_element.lower

        assert numElements_higher == numElements_lower, 'Bidirectional linked list not consistent.' \
                                                        ' From head to tail element count is {}, from tail to head is {}'.format(
                                                        numElements_higher, numElements_lower)

        print("Bidirectional list link - Passed")

        column_original, data_original = self.from_linked_list_to_python_list(head)

        assert numElements_higher == len(column_original), \
            'Data structure size inconsistent. LinkedList is {}, Python list is {}'.format(numElements_higher, len(column_original))

        for index in range(len(column_original)-1):
            assert column_original[index] < column_original[index+1],\
                'Columns not ordered correctly. Tree not flattened properly'

        print("Bidirectional list ordering - Passed")

        # Transform list into tree and back into list, as it is easy to test
        tree_root = self.build_tree_from_list_flat(head)
        head = self.subtree_to_list_flat(tree_root)

        cdef numElements_higher_after = 0
        cdef numElements_lower_after = 0

        current_element = head

        while current_element != NULL:
            numElements_higher_after += 1
            previous_element = current_element
            current_element = current_element.higher

        current_element = previous_element
        while current_element != NULL:
            numElements_lower_after += 1
            current_element = current_element.lower

        print("Bidirectional list from tree link - Passed")

        assert numElements_higher_after == numElements_lower_after, \
            'Bidirectional linked list after tree construction not consistent. ' \
            'From head to tail element count is {}, from tail to head is {}'.format(
            numElements_higher_after, numElements_lower_after)

        assert numElements_higher == numElements_higher_after, \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                numElements_higher, numElements_higher_after)

        column_after_tree, data_after_tree = self.from_linked_list_to_python_list(head)

        assert len(column_original) == len(column_after_tree), \
            'Data structure size inconsistent. Original length is {}, after tree conversion is {}'.format(
                len(column_original), len(column_after_tree))

        for index in range(len(column_original)):
            assert column_original[index] == column_after_tree[index],\
                'After tree construction columns are not ordered properly'
            assert data_original[index] == data_after_tree[index],\
                'After tree construction data content is changed'

        print("Bidirectional list from tree ordering - Passed")



    cpdef test_topK_from_list_selection(self, long row, long topK):
        """
        The function tests the topK selection from list
        
        :param row: row to use for testing
        """

        cdef matrix_element_tree_s *head

        head = self.subtree_to_list_flat(self.row_pointer[row].head)

        column_original, data_original = self.from_linked_list_to_python_list(head)

        head = self.topK_selection_from_list(head, topK)

        column_topK, data_topK = self.from_linked_list_to_python_list(head)

        assert len(column_topK) == len(data_topK),\
            "TopK data and column lists have different length. Columns length is {}, data is {}".format(len(column_topK), len(data_topK))
        assert len(column_topK) <= topK,\
            "TopK extracted list is longer than desired value. Desired is {}, while list is {}".format(topK, len(column_topK))

        print("TopK extracted length - Passed")

        # Sort with respect to the content to select topK
        idx_sorted = np.argsort(data_original)
        idx_sorted = np.flip(idx_sorted, axis=0)
        top_k_idx = idx_sorted[0:topK]

        column_topK_numpy = np.array(column_original)[top_k_idx]
        data_topK_numpy = np.array(data_original)[top_k_idx]

        # Sort with respect to the column to ensure it is ordered as the tree flattened list
        idx_sorted = np.argsort(column_topK_numpy)
        column_topK_numpy = column_topK_numpy[idx_sorted]
        data_topK_numpy = data_topK_numpy[idx_sorted]


        assert len(column_topK_numpy) <= len(column_topK),\
            "TopK extracted list and numpy one have different length. Extracted list lenght is {}, while numpy is {}".format(
                len(column_topK_numpy), len(column_topK))


        for index in range(len(column_topK)):

            assert column_topK[index] == column_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as column value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, column_topK[index], column_topK_numpy[index])

            assert data_topK[index] == data_topK_numpy[index], \
                "TopK extracted list and numpy one have different content at index {} as data value." \
                " Extracted list lenght is {}, while numpy is {}".format(index, data_topK[index], data_topK_numpy[index])

        print("TopK extracted content - Passed")






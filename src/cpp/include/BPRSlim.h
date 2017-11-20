//
// Created by leonardoarcari on 18/11/17.
//

#ifndef CPP_BPRSLIM_H
#define CPP_BPRSLIM_H

#include <eigen3/Eigen/SparseCore>

using namespace Eigen;

namespace mbb {

/**
 * BPR optimized SLIM, with stochastic gradient descent. 
 *  
 * @tparam     T data type of user rating matrix entries.
 */
template <typename T> class BPRSlim {
public:
  template <typename U, StorageOptions O = RowMajor>
  using SpMat = SparseMatrix<U, O>;

  /**
   * Default constructor. It's there only for allowing stack allocation
   * of this class in Cython. Don't use.
   */
  BPRSlim() = default;
  
  /**
   * BPRSlim constructor. Initializes the BPR optimization algorithm for
   * a SLIM model. The optimization is performed with stochastic gradient
   * descent, by random sampling with replacement a user u and two items
   * i and j such that i is a positive rated item and j is a negative rated
   * item. 
   *
   * @param[in]  urm              The URM
   * @param[in]  learningRate     The SGD learning rate
   * @param[in]  lpos             The regularization coefficient for
   *                              positive items
   * @param[in]  lneg             The regularization coefficient for
   *                              negative items
   * @param[in]  topK             The threshold for topK filtering the
   *                              parameters matrix
   * @param[in]  epochMultiplier  The multiplier of the epoch length. By
   *                              default an epoch is as long as the number
   *                              of positive items in the \p urm.
   */
  BPRSlim(SpMat<T> urm, double learningRate, double lpos, double lneg, int topK,
          double epochMultiplier);

  /**
   * Run the BPR optimization process. The optimization is performed with
   * stochastic gradient descent, by random sampling with replacement a
   * user u and two items i and j such that i is a positive rated item and
   * j is a negative rated item.
   * The algorithm runs \p epochs epochs, given that each epoch is
   * <n of non-zero entries of urm> * epochMultiplier
   *
   * @param[in]  epochs  Number of epochs of the SGD algorithm
   */
  void fit(int epochs);

  /**
   * Returns a deep-copy of the parameters matrix. Explicit zeros are
   * removed and return matrix is compressed.
   *
   * @return     The model parameters matrix.
   */
  Eigen::SparseMatrix<double, RowMajor> getParameters();

private:
  SpMat<T> urm;
  SpMat<double> W;
  double learningRate = 0.05;
  double lpos = 0.0025;
  double lneg = 0.00025;
  int topK;
  double epochMultiplier = 1;
  int epochLength;

  /**
   * Random sample with replacement a user index in the urm, and two items
   * i and j such that i is a positive item and j is a negative item.
   *
   * @return     A tuple of sampled user, i and j 
   */
  std::tuple<int, int, int> randomSample();

  /**
   * A single step of BPR-Optimization algorithm for SLIM.
   * Let x_uij = urm[\p userIndex, :].dot(w_i - w_j)
   *     dSigm = 1 / (1 + exp(x_uij))
   *     i = \p positiveIndex
   *     j = \p negativeIndex
   * 
   * The parameter matrix W is then updated as so:
   * W_ki = W_ki + alpha * (dsigm * urm_uk + l_pos * W_ki)
   * W_kj = W_kj + alpha * (-dsigm * x_uk + l_neg * W_kj)
   * 
   *
   * @param[in]  userIndex      The user index
   * @param[in]  positiveIndex  The positive item index
   * @param[in]  negativeIndex  The negative item index
   */
  void BPRStep(int userIndex, int positiveIndex, int negativeIndex);
};
} // namespace mbb

#endif // CPP_BPRSLIM_H

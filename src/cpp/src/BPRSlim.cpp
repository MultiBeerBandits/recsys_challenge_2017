//
// Created by leonardoarcari on 18/11/17.
//

#include "../include/BPRSlim.h"

#include <cmath>
#include <random>
#include <tuple>

namespace mbb {
template <typename T>
BPRSlim<T>::BPRSlim(SpMat<T> urm, double learningRate, double lpos, double lneg,
                    int topK, double epochMultiplier)
    : urm(urm), W(urm.cols(), urm.cols()), learningRate{learningRate},
      lpos{lpos}, lneg{lneg}, topK{topK}, epochMultiplier{epochMultiplier} {

  epochLength = static_cast<int>(urm.nonZeros() * epochMultiplier);
}

template <typename T> void BPRSlim<T>::fit(int epochs) {
  for (int i = 0; i < epochs; ++i) {
    for (int j = 0; j < epochLength; ++j) {
      // Random sample a user and a positive and a negative items
      int userIndex, positiveIndex, negativeIndex;
      std::tie(userIndex, positiveIndex, negativeIndex) = randomSample();
      // Perform a BPR optimization step
      BPRStep(userIndex, positiveIndex, negativeIndex);
    }
    // Remove explicit zeros
    W.setZero();
  }
  // Compress it
  W.makeCompressed();
}

template <typename T> std::tuple<int, int, int> BPRSlim<T>::randomSample() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> random(0, static_cast<int>(urm.rows()));

  int user = random(gen); // Sample random user index

  // Get items rated by user
  auto indPtr = urm.outerIndexPtr();
  int start = indPtr[user];
  int end = indPtr[user + 1];

  // Random sample among positive items
  std::uniform_int_distribution<> sampleI(start, end - 1);
  int i = urm.innerIndexPtr()[sampleI(gen)];

  // Random sample among negative items
  std::uniform_int_distribution<> sampleJ(0, static_cast<int>(urm.cols()));
  int j = sampleJ(gen);

  // Look for j in columns. If j is found, keep sampling.
  auto seenItemsStart = &urm.innerIndexPtr()[start];
  auto seenItemsEnd = &urm.innerIndexPtr()[end];
  while (std::find(seenItemsStart, seenItemsEnd, j) != seenItemsEnd) {
    j = sampleJ(gen);
  }

  return {user, i, j};
}

template <typename T>
void BPRSlim<T>::BPRStep(int userIndex, int positiveIndex, int negativeIndex) {
  using namespace Eigen;
  int u = userIndex, i = positiveIndex, j = negativeIndex;

  auto x_u = urm.row(u);
  // col(index) is available for CSC matrices only, while our W matrix is CSR.
  // Hence, we transpose it and get row(index) which is available.
  auto W_t = W.transpose();
  auto w_i = W_t.row(i);
  auto w_j = W_t.row(j);

  // Transpose (w_i - w_j) to make it column
  auto x_uij = x_u.dot((w_i - w_j).transpose());

  auto dSigm = 1 / (1 + std::exp(x_uij));
  // For each seen item, update W matrix
  for (typename SpMat<T>::InnerIterator it(urm, u); it; ++it) {
    auto x_uk = it.value();
    auto k = it.col();
    auto sigmTerm = dSigm * it.value();

    if (k != i) {
      // W_ki = W_ki + alpha * (dsigm * x_uk + l_pos * W_ki)
      W.coeffRef(k, i) +=
          learningRate * (dSigm * x_uk + lpos * W.coeffRef(k, i));
    }

    if (k != j) {
      // W_kj = W_kj + alpha * (-dsigm * x_uk + l_neg * W_kj)
      W.coeffRef(k, j) +=
          learningRate * (-dSigm * x_uk + lneg * W.coeffRef(k, j));
    }
  }
}

template <typename T>
Eigen::SparseMatrix<double, Eigen::RowMajor> BPRSlim<T>::getParameters() {
  // Make a deep copy, clean it, compress it and deliver it
  auto ret = Eigen::SparseMatrix<double, Eigen::RowMajor>(W);
  ret.setZero();
  ret.makeCompressed();
  return ret;
}
} // namespace mbb

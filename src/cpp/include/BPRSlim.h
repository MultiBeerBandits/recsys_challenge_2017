//
// Created by leonardoarcari on 18/11/17.
//

#ifndef CPP_BPRSLIM_H
#define CPP_BPRSLIM_H

#include <eigen3/Eigen/SparseCore>

namespace mbb {
template<typename T> class BPRSlim {
public:
  template<typename U,
      Eigen::StorageOptions O = Eigen::StorageOptions::RowMajor>
  using SpMat = Eigen::SparseMatrix<U, O>;

  BPRSlim() = default;
  BPRSlim(SpMat<T> urm, double learningRate, double lpos, double lneg, int topK,
          double epochMultiplier);
  void fit(int epochs);
  Eigen::SparseMatrix<double, Eigen::RowMajor> getParameters();

private:
  SpMat<T> urm;
  SpMat<double> W;
  double learningRate = 0.05;
  double lpos = 0.0025;
  double lneg = 0.00025;
  int topK;
  double epochMultiplier = 1;
  int epochLength;

  std::tuple<int, int, int> randomSample();
  void BPRStep(int userIndex, int positiveIndex, int negativeIndex);
};
} // end mbb namespace

#endif // CPP_BPRSLIM_H

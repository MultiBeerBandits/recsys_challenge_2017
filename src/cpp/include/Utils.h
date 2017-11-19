//
// Created by leonardoarcari on 18/11/17.
//

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <eigen3/Eigen/SparseCore>
#include <iostream>
#include <vector>

namespace mbb {
template<typename T>
Eigen::SparseMatrix<T, Eigen::RowMajor>
buildSparseMatrix(std::vector<int> const &indPtr, std::vector<T> const &data,
                  std::vector<int> const &indices, int n_rows, int n_cols) {
  using namespace Eigen;
  SparseMatrix<T, RowMajor> m(n_rows, n_cols);

  for (int i = 0; i < indPtr.size() - 1; ++i) {
    int start = indPtr[i];
    int end = indPtr[i + 1];
    for (int k = start; k < end; ++k) {
      m.insert(i, indices[k]) = data[k];
    }
  }

  m.makeCompressed();
  return m;
}
} // end of mbb namespace

#endif // CPP_UTILS_H

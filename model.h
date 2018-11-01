
#ifndef _MODEL_H_
#define _MODEL_H_

#include <vector>
#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DMatrixf;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> DVectorf;

//component-wise operations don't care Aliasing problem
// mat = relu(mat)
void relu(DMatrixf& mat);

// res = X * W + b
void linear(const DMatrixf& X, const DMatrixf& W, const DVectorf& b, DMatrixf& res);


class DnnModel {
public:
    int InitModel(const std::string& filepath);
    int MakeFeature(std::vector<int>& input_ids, DMatrixf& input_mat) const;
    int CalcOutput(const DMatrixf& input_mat, float& val) const;

private:
    int embedding_size_;       // 64
    std::vector<DVectorf> embedding_vec_;   // 1w * embedding_size_

    DMatrixf  trans_mat_;      // 64 * 256
    std::vector<DMatrixf> weights_;    // 256 * 256    一共10层，最后一层 256 *1 
    std::vector<DVectorf> bias_;      // 256 * 1    一共10层，最后一层 1 *1
};
#endif

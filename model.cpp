
#include "model.h"
#include <fstream>
#include <string>

void relu(DMatrixf& mat) {
      mat = mat.array().cwiseMax(float(0));
}
// res = X * W + b
void linear(const DMatrixf& X, const DMatrixf& W, const DVectorf& b, DMatrixf& res) {
      res.noalias() = X * W;
      res.rowwise() += b;
}

int split_string(std::vector<std::string>& str_vec, std::string& line, const std::string& delim) {
    if (line.empty()) {
        return -1;
    }
    char* strs = new char[line.length() + 1] ; //不要忘了
    strcpy(strs, line.c_str());
    char* d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while (p) {
        std::string s = p; //分割得到的字符串转换为string类型
        str_vec.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }
    return 0;
}

int DnnModel::InitModel(const std::string& filepath) {
    std::ifstream model_is(filepath);
    if (!model_is) {
        std::cout << "model file not exited" << std::endl;
        return -1;
    }
    embedding_size_ = 64;
    std::string line;
    try {
    //1. 前1万行是 embedding_vec_, 每行64个float数
    embedding_vec_.resize(10000);
    for (int i = 0; i < 10000; ++i) {
        getline(model_is, line);
        std::vector<std::string> str_vec;
        split_string(str_vec, line, ",");
        DVectorf temp(64);
        for (int j = 0; j < 64; ++j) {
            temp(j) = std::stof(str_vec[j]);
        }
        embedding_vec_[i] = temp;
    }

    //2. 读取矩阵trans_mat_， 64 * 256，一共64行，每行256个float数
    trans_mat_.resize(64, 256);
    for (int i = 0; i < 64; ++i) {
        getline(model_is, line);
        std::vector<std::string> str_vec;
        split_string(str_vec, line, ",");
        for (int j = 0; j < 256; ++j) {
            trans_mat_(i, j) = std::stof(str_vec[j]);
        }
    }

    //3. 读取DNN的weight 和bias
    weights_.resize(10);
    bias_.resize(10);
    for (int layer = 0; layer < 10; ++layer) {
         int idim = 256;
         int odim = 256;
         if (layer == 9) {
             odim = 1;
         }
         DMatrixf weight_lay(idim, odim);
         for (int i = 0; i < idim; ++i) {
             getline(model_is, line);
             std::vector<std::string> str_vec;
             split_string(str_vec, line, ",");
             for (int j = 0; j < odim; ++j) {
                 weight_lay(i, j) = std::stof(str_vec[j]);
             }
         }
         weights_[layer] = weight_lay;

         DVectorf bias_lay(odim);
         getline(model_is, line);
         std::vector<std::string> str_vec;
         split_string(str_vec, line, ",");
         for (int i = 0; i < odim; ++i) {
             bias_lay(i) = std::stof(str_vec[i]);
         }
         bias_[layer] = bias_lay;

    }
    } catch (std::invalid_argument& e) { 
        std::cout << "error line: " << line << std::endl;
        return -1;
    } catch (std::out_of_range& e) {
        std::cout << "error line: " << line << std::endl;
        return -1;
    }
    return 0;
}

int DnnModel::MakeFeature(std::vector<int>& input_ids, DMatrixf& input_mat) const {
    input_mat.resize(input_ids.size(), embedding_size_);
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const DVectorf& embedding = embedding_vec_[i];
        input_mat.row(i) = embedding;
    }
    return 0;
}


int DnnModel::CalcOutput(const DMatrixf& input_mat, float& predict_val) const {

    //1. 矩阵乘法
    DMatrixf temp_mat;
    temp_mat.noalias() = input_mat * trans_mat_;

    //2. dnn输入向量
    DMatrixf dnn_input(1, embedding_size_);
    dnn_input = temp_mat.colwise().sum();

    //3.计算10层DNN
    DMatrixf dnn_temp_mat;
    for (int i = 0; i < 10; ++i) {
        if (i % 2 == 0) {
            linear(dnn_input, weights_[i], bias_[i], dnn_temp_mat);
            if (i != 9) {
                relu(dnn_temp_mat);
            }
        } else {
            linear(dnn_temp_mat, weights_[i], bias_[i], dnn_input);
            if (i != 9) {
                relu(dnn_input);
            }
        }
    }

    predict_val = dnn_input(0, 0);
    return 0;
}

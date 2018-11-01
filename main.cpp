
#include <string>
#include <vector>
#include <thread>
#include <Eigen/Dense>
#include <stdlib.h>
#include <iostream>

#include "model.h"


DnnModel g_model;

void model_calc_func(int i) {

     //
     int test_case = 10;
     for (int i = 0; i < test_case; ++i) {
         //生成测试数据
         int id_count = 100;
         std::vector<int> input_ids(id_count, 0);
         for (int j =0; j < id_count; ++j) {
             input_ids[i] = rand() % 10000;
         }

         DMatrixf input_mat;
         g_model.MakeFeature(input_ids, input_mat);

         float val;
         g_model.CalcOutput(input_mat, val);
         std::cout << "predict: " << val << std::endl;
     }
}

int main() {
    int thread_num = 40;
    int res = g_model.InitModel("./data/model_data.txt");
    if (res != 0) {
        std::cout << "InitModel failed!" <<std::endl;
        return -1;
    }

    std::vector<std::thread>  thread_pool;
    for (int i = 0; i < thread_num; ++i) {
        thread_pool.push_back(std::thread(std::bind(&model_calc_func, i)));
    }
    for (auto& th: thread_pool) {
        th.join();
    }
    return 0;
}


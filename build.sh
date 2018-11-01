
g++ -g -O2 -W -fPIC -mavx -mavx2 -mfma -msse4.1 -msse4.2 -o test_dnn main.cpp model.cpp -I./eigen3 -std=c++11
